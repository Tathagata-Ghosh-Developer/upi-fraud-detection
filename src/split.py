"""Stage 3: time-based train / validation / test split on `step` (hours).

Fraud models are always deployed on the future. A random split would let the
model train on transactions that happen after the ones it is tested on, and
would hide concept drift, so we cut the timeline instead:

    train       steps   1 .. train_end_step
    validation  steps   .. val_end_step     (early stopping, tuning, threshold choice)
    test        steps   .. test_end_step    (touched once, at the end)
    late_period steps   after test_end_step (stress test, see README)
"""

from __future__ import annotations

import dask.dataframe as dd

from .backend import to_pandas
from .common import LABEL, RESULTS_DIR, base_parser, get_paths, load_config, stage_timer, update_json

ID_COLUMNS = ["txn_id", "step", "day", "hour_of_day", "type", "amount", LABEL, "isFlaggedFraud"]


def model_columns(cfg: dict) -> list[str]:
    cols = list(ID_COLUMNS)
    for group in cfg["feature_groups"].values():
        cols += [c for c in group if c not in cols]
    cols += [c for c in cfg["raw_paysim_columns"] if c not in cols]
    return cols


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    s = cfg["split"]
    ranges = {
        "train": (1, s["train_end_step"]),
        "val": (s["train_end_step"] + 1, s["val_end_step"]),
        "test": (s["val_end_step"] + 1, s["test_end_step"]),
        "late_period": (s["test_end_step"] + 1, 10_000),
    }

    summary = {}
    with stage_timer("split"):
        paths.splits.mkdir(parents=True, exist_ok=True)
        ddf = dd.read_parquet(paths.features, columns=model_columns(cfg))
        for name, (lo, hi) in ranges.items():
            part = to_pandas(ddf[(ddf["step"] >= lo) & (ddf["step"] <= hi)].compute())
            part = part.sort_values(["step", "txn_id"]).reset_index(drop=True)
            # float64 -> float32 halves memory; precision loss is irrelevant for trees.
            float_cols = part.select_dtypes("float64").columns
            part[float_cols] = part[float_cols].astype("float32")
            part.to_parquet(paths.splits / f"{name}.parquet", index=False)

            n, fraud = len(part), int(part[LABEL].sum())
            summary[name] = {
                "steps": [int(part["step"].min()), int(part["step"].max())],
                "rows": n,
                "fraud": fraud,
                "fraud_rate": fraud / n,
            }
            print(f"{name:12s} steps {summary[name]['steps']}  rows={n:>9,}  fraud={fraud:>5,}  rate={fraud / n:.4%}")

    update_json(RESULTS_DIR / "data_summary.json", "splits", summary)


if __name__ == "__main__":
    main()
