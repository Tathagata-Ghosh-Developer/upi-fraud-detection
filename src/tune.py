"""Stage 6: hyperparameter search with Ray Tune and the ASHA scheduler.

Random search proposes configurations; ASHA (asynchronous successive halving)
looks at every trial's validation PR-AUC after 50, 150, 450, ... boosting
rounds and stops the bottom two thirds at each rung. Most of the budget
therefore goes to promising configurations. Trials run in parallel, limited
by --gpus-per-trial (0.5 = two trials share one GPU). On a multi-GPU node or
a Ray cluster the same script simply runs more trials at once.

The best configuration is refit on the training window with early stopping
on validation and saved for the evaluation stage.
"""

from __future__ import annotations

import time

import numpy as np
import ray
import xgboost as xgb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from .baselines import save_scores
from .common import (FEATURE_ROLES, LABEL, PROJECT_ROOT, RESULTS_DIR, base_parser, feature_list, get_paths,
                     load_config, load_split, role_feature_set, setup_mlflow, stage_timer, update_json)
from .metrics import ranking_metrics
from .train import fit_xgb, predict, scale_pos_weight

SEARCH_SPACE = {
    "max_depth": tune.randint(4, 11),
    "learning_rate": tune.loguniform(0.02, 0.3),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.5, 1.0),
    "min_child_weight": tune.loguniform(1, 100),
    "reg_lambda": tune.loguniform(0.1, 20),
    "scale_pos_weight_mode": tune.choice(["full", "sqrt"]),
}
REPORT_EVERY = 10  # boosting rounds between reports to Tune


class ReportToTune(xgb.callback.TrainingCallback):
    """Send the best validation PR-AUC so far to Ray Tune every few rounds."""

    def __init__(self):
        self.history: list[float] = []

    def _report(self, rounds: int) -> None:
        best = int(np.argmax(self.history))
        tune.report({"val_aucpr": max(self.history), "best_round": best + 1, "boost_round": rounds})

    def after_iteration(self, model, epoch, evals_log) -> bool:
        self.history = list(evals_log["val"]["aucpr"])
        if (epoch + 1) % REPORT_EVERY == 0:
            self._report(epoch + 1)
        return False  # never stop training from here; ASHA and early stopping do that

    def after_training(self, model):
        if self.history and len(self.history) % REPORT_EVERY:
            self._report(len(self.history))
        return model


def build_params(cfg: dict, config: dict, y_train: np.ndarray) -> dict:
    params = {**cfg["xgboost"]["params"], "seed": cfg["seed"]}
    params.update({k: v for k, v in config.items() if k != "scale_pos_weight_mode"})
    params["scale_pos_weight"] = scale_pos_weight(y_train, config["scale_pos_weight_mode"])
    return params


def trainable(config, cfg, train, val, features):
    params = build_params(cfg, config, train[LABEL].to_numpy())
    fit_xgb(params, train, val, features, num_boost_round=cfg["tune"]["max_boost_rounds"],
            early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"], callbacks=[ReportToTune()])


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--num-samples", type=int, help="number of sampled configurations")
    parser.add_argument("--gpus-per-trial", type=float, help="GPU share per trial, e.g. 0.5 or 1")
    parser.add_argument("--cpus-per-trial", type=int)
    parser.add_argument("--time-budget-s", type=int, help="stop launching trials after this many seconds")
    parser.add_argument("--ray-address", default=None, help="'auto' to join an existing Ray cluster")
    parser.add_argument("--feature-set", default="primary", choices=FEATURE_ROLES,
                        help="primary (authorisation-time features) or strict (no balance columns)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["tune"]
    for key in ("num_samples", "gpus_per_trial", "cpus_per_trial", "time_budget_s"):
        if getattr(args, key) is not None:
            tcfg[key] = getattr(args, key)

    paths = get_paths(cfg)
    role = args.feature_set
    set_name = role_feature_set(cfg, role)
    model_name = f"xgb_tuned_{role}"
    features = feature_list(cfg, set_name)
    columns = ["txn_id", LABEL, *features]
    train, val, test = (load_split(cfg, s, columns) for s in ("train", "val", "test"))

    with stage_timer(f"tune ({role})"):
        # Workers import this package by name, so point them at the project root.
        ray.init(address=args.ray_address, include_dashboard=False, log_to_driver=False,
                 runtime_env={"env_vars": {"PYTHONPATH": str(PROJECT_ROOT)}})
        scheduler = ASHAScheduler(
            time_attr="boost_round",
            max_t=tcfg["max_boost_rounds"],
            grace_period=tcfg["grace_period"],
            reduction_factor=tcfg["reduction_factor"],
        )
        run = tune.with_parameters(trainable, cfg=cfg, train=train[[LABEL, *features]],
                                   val=val[[LABEL, *features]], features=features)
        tuner = tune.Tuner(
            tune.with_resources(run, {"cpu": tcfg["cpus_per_trial"], "gpu": tcfg["gpus_per_trial"]}),
            param_space=SEARCH_SPACE,
            tune_config=tune.TuneConfig(
                metric="val_aucpr",
                mode="max",
                scheduler=scheduler,
                # Seeded random search: the same configurations are proposed on every run.
                search_alg=BasicVariantGenerator(random_state=cfg["seed"]),
                num_samples=tcfg["num_samples"],
                time_budget_s=tcfg["time_budget_s"],
            ),
            run_config=tune.RunConfig(name=f"xgb_asha_{role}", storage_path=str(paths.ray_results), verbose=1),
        )
        results = tuner.fit()
        ray.shutdown()

        best = results.get_best_result(metric="val_aucpr", mode="max", scope="all")
        # Last report of each trial: val_aucpr is a running maximum, so this row holds
        # the trial's best score and the number of rounds it ran before stopping.
        trials = results.get_dataframe().rename(columns={"boost_round": "rounds_run"})
        keep = ["trial_id", "val_aucpr", "best_round", "rounds_run", "time_total_s",
                *[f"config/{k}" for k in SEARCH_SPACE]]
        trials = trials[keep].sort_values("val_aucpr", ascending=False)
        trials.round(5).to_csv(RESULTS_DIR / f"tune_trials_{role}.csv", index=False)
        print(trials.head(10).round(4).to_string(index=False))

    # Refit the winner on the training window and keep it for evaluation.
    with stage_timer(f"refit_best ({role})"):
        mlflow = setup_mlflow(cfg)
        params = build_params(cfg, best.config, train[LABEL].to_numpy())
        with mlflow.start_run(run_name=model_name) as parent:
            mlflow.set_tags({"stage": "tuning", "model": "xgboost", "feature_set": set_name})
            mlflow.log_params({**params, "num_trials": len(trials), "num_samples": tcfg["num_samples"],
                               "gpus_per_trial": tcfg["gpus_per_trial"], "scheduler": "ASHA"})
            for _, t in trials.iterrows():
                with mlflow.start_run(run_name=f"trial_{t['trial_id']}", nested=True):
                    mlflow.set_tags({"stage": "tuning_trial", "parent": parent.info.run_id})
                    mlflow.log_params({k.removeprefix("config/"): t[k] for k in trials.columns if k.startswith("config/")})
                    mlflow.log_metrics({"val_aucpr": t["val_aucpr"], "rounds_run": t["rounds_run"]})

            start = time.perf_counter()
            booster = fit_xgb(params, train, val, features, tcfg["max_boost_rounds"],
                              cfg["xgboost"]["early_stopping_rounds"])
            fit_seconds = time.perf_counter() - start
            booster.save_model(paths.models / f"{model_name}.json")
            for split, df in (("val", val), ("test", test)):
                save_scores(paths, model_name, split, df["txn_id"], predict(booster, df, features))
            val_metrics = ranking_metrics(val[LABEL].to_numpy(), predict(booster, val, features), cfg["evaluation"])
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({"fit_seconds": fit_seconds, "best_iteration": booster.best_iteration})

    update_json(RESULTS_DIR / "best_params.json", role, {
        "feature_set": set_name,
        "search_best_val_aucpr": float(best.metrics["val_aucpr"]),
        "refit_val_pr_auc": val_metrics["pr_auc"],
        "best_iteration": booster.best_iteration,
        "trials_completed": len(trials),
        "config": best.config,
        "xgboost_params": params,
    })
    print(f"best val PR-AUC (search) = {best.metrics['val_aucpr']:.4f}; refit = {val_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
