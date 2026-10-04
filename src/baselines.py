"""Stage 4: baselines that any model has to beat.

1. PaySim's built-in rule, `isFlaggedFraud`, documented as flagging attempts to
   transfer more than 200,000 in one go (it fires on only 16 of 6.36M rows).
2. Logistic regression with balanced class weights, on the primary and the
   strict feature sets.
   Monetary features are heavy-tailed, so they get a signed log transform and
   standardisation first; otherwise a few huge balances dominate the fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .common import (FEATURE_ROLES, LABEL, base_parser, feature_list, get_paths, load_config, load_split,
                     role_feature_set, setup_mlflow, stage_timer)
from .metrics import ranking_metrics


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def save_scores(paths, model: str, split: str, txn_id, score) -> None:
    pd.DataFrame({"txn_id": txn_id, "score": score}).to_parquet(
        paths.scores / f"{model}__{split}.parquet", index=False
    )


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    mlflow = setup_mlflow(cfg)
    # The primary feature set is a superset of the strict one, so load it once.
    columns = ["txn_id", LABEL, "isFlaggedFraud", *feature_list(cfg, role_feature_set(cfg, "primary"))]
    data = {s: load_split(cfg, s, columns) for s in ("train", "val", "test")}

    with stage_timer("baselines"):
        # Rule baseline: no training, the score is the flag itself.
        with mlflow.start_run(run_name="baseline_isFlaggedFraud"):
            mlflow.set_tags({"stage": "baseline", "model": "rule"})
            for split in ("val", "test"):
                df = data[split]
                m = ranking_metrics(df[LABEL].to_numpy(), df["isFlaggedFraud"].to_numpy(), cfg["evaluation"])
                mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
                save_scores(paths, "rule_isFlaggedFraud", split, df["txn_id"], df["isFlaggedFraud"].astype(float))
                print(f"isFlaggedFraud {split}: flagged={int(df['isFlaggedFraud'].sum())}  PR-AUC={m['pr_auc']:.4f}")

        for role in FEATURE_ROLES:
            set_name = role_feature_set(cfg, role)
            features = feature_list(cfg, set_name)
            with mlflow.start_run(run_name=f"baseline_logreg_{role}"):
                mlflow.set_tags({"stage": "baseline", "model": "logistic_regression", "feature_set": set_name})
                model = make_pipeline(
                    FunctionTransformer(signed_log1p),
                    StandardScaler(),
                    LogisticRegression(class_weight="balanced", max_iter=1000),
                )
                train = data["train"]
                model.fit(train[features].to_numpy(), train[LABEL].to_numpy())
                mlflow.log_params({"feature_set": set_name, "class_weight": "balanced", "C": 1.0})
                for split in ("val", "test"):
                    df = data[split]
                    score = model.predict_proba(df[features].to_numpy())[:, 1]
                    m = ranking_metrics(df[LABEL].to_numpy(), score, cfg["evaluation"])
                    mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
                    save_scores(paths, f"logreg_{role}", split, df["txn_id"], score)
                    print(f"logreg_{role} {split}: PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
