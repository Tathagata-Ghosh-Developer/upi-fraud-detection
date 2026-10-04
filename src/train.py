"""Stage 5: XGBoost on the GPU, plus a feature-set ablation.

Every ablation run uses the same default hyperparameters, early-stops on
validation PR-AUC and is then scored once on the test window. Only the
feature set changes, so differences between rows are due to the features.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from .baselines import save_scores
from .common import LABEL, RESULTS_DIR, base_parser, feature_list, get_paths, load_config, load_split, setup_mlflow, stage_timer
from .metrics import ranking_metrics


def scale_pos_weight(y: np.ndarray, mode: str) -> float:
    """Up-weight the rare fraud class. `full` makes both classes weigh the same in total."""
    ratio = float((y == 0).sum() / max(1, (y == 1).sum()))
    return {"full": ratio, "sqrt": float(np.sqrt(ratio)), "none": 1.0}[mode]


def fit_xgb(params: dict, train: pd.DataFrame, val: pd.DataFrame, features: list[str],
            num_boost_round: int, early_stopping_rounds: int, callbacks=None) -> xgb.Booster:
    # QuantileDMatrix bins features once, which saves GPU memory with the `hist` method.
    dtrain = xgb.QuantileDMatrix(train[features], label=train[LABEL])
    dval = xgb.QuantileDMatrix(val[features], label=val[LABEL], ref=dtrain)
    return xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
        verbose_eval=False,
    )


def predict(booster: xgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Score with the trees up to the early-stopping point."""
    return booster.predict(xgb.DMatrix(df[features]), iteration_range=(0, booster.best_iteration + 1))


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    mlflow = setup_mlflow(cfg)
    xcfg, ecfg = cfg["xgboost"], cfg["evaluation"]
    splits = {s: load_split(cfg, s) for s in ("train", "val", "test")}
    y_train = splits["train"][LABEL].to_numpy()

    rows = []
    with stage_timer("train_ablation"):
        for set_name in cfg["ablation"]:
            features = feature_list(cfg, set_name)
            params = {**xcfg["params"], "scale_pos_weight": scale_pos_weight(y_train, xcfg["scale_pos_weight"]),
                      "seed": cfg["seed"]}
            with mlflow.start_run(run_name=f"xgb_{set_name}"):
                mlflow.set_tags({"stage": "ablation", "model": "xgboost", "feature_set": set_name})
                mlflow.log_params({**params, "feature_set": set_name, "n_features": len(features)})
                start = time.perf_counter()
                booster = fit_xgb(params, splits["train"], splits["val"], features,
                                  xcfg["num_boost_round"], xcfg["early_stopping_rounds"])
                fit_seconds = time.perf_counter() - start

                row = {"feature_set": set_name, "n_features": len(features),
                       "best_iteration": booster.best_iteration, "fit_seconds": round(fit_seconds, 1)}
                for split in ("val", "test"):
                    df = splits[split]
                    score = predict(booster, df, features)
                    m = ranking_metrics(df[LABEL].to_numpy(), score, ecfg)
                    row.update({f"{split}_{k}": v for k, v in m.items() if k != "base_rate"})
                    save_scores(paths, f"xgb_{set_name}", split, df["txn_id"], score)
                mlflow.log_metrics({k: v for k, v in row.items() if k.startswith(("val_", "test_"))})
                mlflow.log_metrics({"fit_seconds": fit_seconds, "best_iteration": booster.best_iteration})
                booster.save_model(paths.models / f"xgb_{set_name}.json")
            rows.append(row)
            print(f"{set_name:30s} iters={row['best_iteration']:4d}  fit={fit_seconds:5.1f}s  "
                  f"val PR-AUC={row['val_pr_auc']:.4f}  test PR-AUC={row['test_pr_auc']:.4f}")

    table = pd.DataFrame(rows)
    table.round(4).to_csv(RESULTS_DIR / "ablation.csv", index=False)
    print(table[["feature_set", "val_pr_auc", "test_pr_auc", "test_roc_auc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
