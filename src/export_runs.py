"""Stage 9: export a compact summary of the MLflow runs and the run environment.

The MLflow store itself stays outside the repository; this CSV is the small,
reviewable record of what was run. Individual tuning trials are summarised in
results/tune_trials.csv and are left out here.
"""

from __future__ import annotations

import os
import platform
import subprocess
from importlib.metadata import version

from .common import RESULTS_DIR, base_parser, load_config, setup_mlflow, write_json

KEY_METRICS = ["val_pr_auc", "test_pr_auc", "test_roc_auc", "test_recall_at_precision_0.9",
               "test_recall_at_top_0.1pct", "test_recall_at_top_0.5pct", "fit_seconds", "best_iteration"]
PACKAGES = ["pandas", "numpy", "dask", "pyarrow", "xgboost", "ray", "mlflow", "scikit-learn", "shap"]


def environment() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9 if hasattr(os, "sysconf") else None
    return {
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": round(mem_gb, 1) if mem_gb else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in PACKAGES},
    }


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    mlflow = setup_mlflow(cfg)
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    runs = runs[runs["tags.stage"] != "tuning_trial"]
    # A stage rerun adds a new run with the same name; keep only the latest one.
    runs = runs.sort_values("start_time", ascending=False).drop_duplicates("tags.mlflow.runName")
    columns = {"tags.mlflow.runName": "run_name", "tags.stage": "stage", "tags.model": "model",
               "tags.feature_set": "feature_set", "start_time": "start_time"}
    columns.update({f"metrics.{m}": m for m in KEY_METRICS})
    table = runs[[c for c in columns if c in runs.columns]].rename(columns=columns)
    table = table.sort_values("start_time")
    table["start_time"] = table["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    table.round(4).to_csv(RESULTS_DIR / "mlflow_runs.csv", index=False)
    print(table.drop(columns="start_time").round(4).to_string(index=False))
    write_json(RESULTS_DIR / "environment.json", environment())


if __name__ == "__main__":
    main()
