"""Stage 2: feature engineering with Dask.

Two kinds of features:

* Row-wise features (time of day, ratios, flags) need only the row itself, so
  they are computed independently on each partition.
* Velocity features (how active was this account recently?) need an account's
  history. The data is hash-partitioned by account so that every transaction
  of an account lands in the same partition; each partition is then processed
  independently. This "shuffle, then compute locally" pattern is what lets the
  job scale out to more workers.

All partition functions use operations supported by both pandas and cuDF.
"""

from __future__ import annotations

import dask.dataframe as dd
import numpy as np

from .backend import configure_backend, to_pandas
from .common import RESULTS_DIR, TXN_TYPES, base_parser, get_paths, load_config, stage_timer, update_json


def add_window_features(df, key: str, prefix: str, windows: list[int]):
    """Count and total amount of `key`'s transactions in each look-back window.

    For a transaction at hour t, the w-hour window covers hours t-w .. t-1.
    PaySim timestamps are whole hours, so transactions inside the same hour
    cannot be ordered; leaving the current hour out guarantees that no
    feature uses information from the future.

    Implementation (a sweep line, built only from sort and cumulative sum):
    each (account, hour) with n transactions emits +n at hour+1, when it
    enters the window, and -n at hour+w+1, when it leaves. A running sum of
    these deltas, read at hour t, is exactly the number of transactions in
    the window.
    """
    per_hour = df.groupby([key, "step"])["amount"].agg(["count", "sum"]).reset_index()
    queries = per_hour[[key, "step"]].rename(columns={"step": "time"})
    queries["d_cnt"] = 0
    queries["d_amt"] = 0.0
    queries["is_query"] = 1

    for w in windows:
        enters = per_hour[[key]].copy()
        enters["time"] = per_hour["step"] + 1
        enters["d_cnt"] = per_hour["count"]
        enters["d_amt"] = per_hour["sum"]
        enters["is_query"] = 0

        leaves = enters.copy()
        leaves["time"] = per_hour["step"] + w + 1
        leaves["d_cnt"] = -per_hour["count"]
        leaves["d_amt"] = -per_hour["sum"]

        events = _concat([enters, leaves, queries]).sort_values([key, "time", "is_query"])
        events["cnt"] = events.groupby(key)["d_cnt"].cumsum()
        events["amt"] = events.groupby(key)["d_amt"].cumsum()

        result = events[events["is_query"] == 1][[key, "time", "cnt", "amt"]]
        result = result.rename(
            columns={"time": "step", "cnt": f"{prefix}_cnt_{w}h", "amt": f"{prefix}_amt_{w}h"}
        )
        per_hour = per_hour.merge(result, on=[key, "step"], how="left")

    feature_cols = [c for c in per_hour.columns if c.startswith(f"{prefix}_")]
    out = df.merge(per_hour[[key, "step", *feature_cols]], on=[key, "step"], how="left")
    for c in feature_cols:
        # Float rounding in the running sum can leave residues such as -1e-9;
        # rounding removes them and "+ 0.0" turns the resulting -0.0 into 0.0.
        out[c] = (out[c].round(2) + 0.0).astype("float32")
    return out


def _concat(frames):
    """pandas.concat or cudf.concat, matching the type of the input frames."""
    if hasattr(frames[0], "to_pandas"):
        import cudf

        return cudf.concat(frames, ignore_index=True)
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def add_row_features(df):
    """Features that depend only on the transaction itself."""
    amount = df["amount"]
    old_org, new_org = df["oldbalanceOrg"], df["newbalanceOrig"]
    old_dest, new_dest = df["oldbalanceDest"], df["newbalanceDest"]

    # Time: hour of day on a circle, so 23:00 and 00:00 are neighbours.
    hour = df["step"] % 24
    df["hour_of_day"] = hour.astype("int8")
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["day"] = ((df["step"] - 1) // 24).astype("int16")

    # The transaction itself.
    df["amount_log"] = np.log1p(amount).astype("float32")
    # fillna: string comparisons on missing values return NA (none in PaySim, but
    # Dask probes the function with placeholder rows to infer output types).
    for t in TXN_TYPES:
        df[f"type_{t}"] = (df["type"] == t).fillna(False).astype("int8")
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").fillna(False).astype("int8")

    # Payer balance before the transaction (known to the remitter bank).
    df["amount_to_orig_balance"] = (amount / (old_org + 1.0)).astype("float32")
    df["orig_balance_is_zero"] = (old_org == 0).astype("int8")

    # Post-transaction and payee balances. In PaySim these encode simulator
    # bookkeeping that differs for fraud rows; see README "Balance leakage".
    df["amount_to_dest_balance"] = (amount / (old_dest + 1.0)).astype("float32")
    df["orig_balance_error"] = (old_org - amount - new_org).astype("float32")
    df["dest_balance_error"] = (old_dest + amount - new_dest).astype("float32")
    df["dest_balances_both_zero"] = ((old_dest == 0) & (new_dest == 0) & (amount > 0)).astype("int8")
    return df


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf", "auto"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    engine = configure_backend(args.engine)
    windows = cfg["features"]["windows_hours"]
    nparts = cfg["features"]["npartitions"]

    with stage_timer(f"features ({engine})"):
        ddf = dd.read_parquet(paths.parquet)
        ddf = ddf.shuffle(on="nameOrig", npartitions=nparts)
        ddf = ddf.map_partitions(add_window_features, key="nameOrig", prefix="orig", windows=windows)
        ddf = ddf.shuffle(on="nameDest", npartitions=nparts)
        ddf = ddf.map_partitions(add_window_features, key="nameDest", prefix="dest", windows=windows)
        ddf = ddf.map_partitions(add_row_features)
        ddf.to_parquet(paths.features, write_index=False, overwrite=True)

    # How often does an account actually have recent history?
    w = max(windows)
    check = to_pandas(dd.read_parquet(paths.features, columns=[f"orig_cnt_{w}h", f"dest_cnt_{w}h"]).compute())
    coverage = {
        "engine": engine,
        f"share_rows_with_orig_history_{w}h": float((check[f"orig_cnt_{w}h"] > 0).mean()),
        f"share_rows_with_dest_history_{w}h": float((check[f"dest_cnt_{w}h"] > 0).mean()),
    }
    print(coverage)
    update_json(RESULTS_DIR / "data_summary.json", "feature_coverage", coverage)


if __name__ == "__main__":
    main()
