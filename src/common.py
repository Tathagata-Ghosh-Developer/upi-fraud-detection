"""Shared helpers: configuration, paths, stage timing, data loading and MLflow setup."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# FRAUD_RESULTS redirects outputs, e.g. for a side-by-side run of the cuDF path.
RESULTS_DIR = Path(os.environ.get("FRAUD_RESULTS", PROJECT_ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"

LABEL = "isFraud"
TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    return parser


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    work_dir = os.environ.get("FRAUD_WORK", cfg["work_dir"])
    cfg["work_dir"] = str(Path(work_dir).expanduser().resolve())
    return cfg


def get_paths(cfg: dict) -> SimpleNamespace:
    """All on-disk locations. Everything large lives under work_dir."""
    work = Path(cfg["work_dir"])
    p = SimpleNamespace(
        work=work,
        raw_csv=work / "data" / "raw" / cfg["raw_csv_name"],
        parquet=work / "data" / "parquet",
        features=work / "data" / "features",
        splits=work / "data" / "splits",
        models=work / "models",
        scores=work / "scores",
        mlflow=work / "mlflow",
        ray_results=work / "ray_results",
    )
    for d in (p.models, p.scores, p.mlflow, p.ray_results, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return p


def feature_list(cfg: dict, set_name: str) -> list[str]:
    """Resolve an ablation set name into an ordered, de-duplicated feature list."""
    columns: list[str] = []
    for group in cfg["ablation"][set_name]:
        source = cfg[group] if group == "raw_paysim_columns" else cfg["feature_groups"][group]
        columns.extend(c for c in source if c not in columns)
    return columns


FEATURE_ROLES = ("primary", "strict")


def role_feature_set(cfg: dict, role: str) -> str:
    """Map a role ("primary" or "strict") to its ablation feature-set name."""
    return cfg[f"{role}_feature_set"]


def load_split(cfg: dict, name: str, columns: list[str] | None = None):
    import pandas as pd

    return pd.read_parquet(get_paths(cfg).splits / f"{name}.parquet", columns=columns)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def update_json(path: Path, key: str, value) -> None:
    """Read-modify-write one key of a JSON file shared by several stages."""
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    write_json(path, data)


@contextmanager
def stage_timer(stage: str):
    """Print and record the wall-clock time of a pipeline stage in results/timings.json."""
    start = time.perf_counter()
    print(f"[{stage}] started")
    yield
    elapsed = round(time.perf_counter() - start, 1)
    update_json(RESULTS_DIR / "timings.json", stage, elapsed)
    print(f"[{stage}] finished in {elapsed:.1f} s")


def setup_mlflow(cfg: dict):
    """Point MLflow at a local SQLite store under work_dir and select the experiment."""
    import mlflow

    paths = get_paths(cfg)
    mlflow.set_tracking_uri(f"sqlite:///{paths.mlflow / 'mlflow.db'}")
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(paths.mlflow / "artifacts").as_uri())
    mlflow.set_experiment(name)
    return mlflow
