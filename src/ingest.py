"""Stage 1: read the raw PaySim CSV with Dask and write partitioned Parquet.

Parquet is columnar and typed, so later stages read only the columns they need
and skip CSV parsing. A stable `txn_id` (row number in the original file) is
added so results can always be traced back to a source row.
"""

from __future__ import annotations

import dask.dataframe as dd

from .backend import configure_backend, to_pandas
from .common import LABEL, RESULTS_DIR, base_parser, get_paths, load_config, stage_timer, write_json

DTYPES = {
    "step": "int16",
    "type": "string",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

# Published figures for the Kaggle release of PaySim; the run fails loudly on a mismatch.
EXPECTED_ROWS = 6_362_620
EXPECTED_FRAUD = 8_213


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf", "auto"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    engine = configure_backend(args.engine)

    with stage_timer(f"ingest ({engine})"):
        ddf = dd.read_csv(paths.raw_csv, dtype=DTYPES, blocksize=cfg["ingest"]["blocksize"])
        # Global row number: a running count of ones across partitions (which keep file order).
        ddf = ddf.assign(_one=1)
        ddf["txn_id"] = ddf["_one"].cumsum() - 1
        ddf = ddf.drop(columns="_one")
        ddf.to_parquet(paths.parquet, write_index=False, overwrite=True)

        cols = ["step", "type", LABEL, "isFlaggedFraud"]
        # The summary is tiny; do it in pandas whatever the engine.
        df = to_pandas(dd.read_parquet(paths.parquet, columns=cols).compute())
        n_rows, n_fraud = len(df), int(df[LABEL].sum())
        by_type = df.groupby("type")[LABEL].agg(["count", "sum"])
        summary = {
            "rows": n_rows,
            "fraud": n_fraud,
            "fraud_rate": n_fraud / n_rows,
            "is_flagged_fraud": int(df["isFlaggedFraud"].sum()),
            "steps": [int(df["step"].min()), int(df["step"].max())],
            "parquet_partitions": ddf.npartitions,
            "by_type": {
                str(t): {"rows": int(r["count"]), "fraud": int(r["sum"])}
                for t, r in by_type.iterrows()
            },
        }

    print(f"rows={n_rows:,}  fraud={n_fraud:,}  rate={n_fraud / n_rows:.4%}")
    print(by_type.rename(columns={"count": "rows", "sum": "fraud"}).to_string())
    write_json(RESULTS_DIR / "data_summary.json", summary)
    if (n_rows, n_fraud) != (EXPECTED_ROWS, EXPECTED_FRAUD):
        raise SystemExit(f"Unexpected dataset: {n_rows} rows / {n_fraud} fraud")


if __name__ == "__main__":
    main()
