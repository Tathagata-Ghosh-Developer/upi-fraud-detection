"""Check that two feature-stage outputs (e.g. pandas vs cuDF engine) agree.

Usage: python scripts/compare_feature_outputs.py DIR_A DIR_B
Rows are matched on txn_id. Columns are read one at a time to keep memory low.
Numeric columns are compared with a small tolerance, because GPU and CPU
floating-point sums can differ in the last bits.
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_column(path: str, column: str) -> pd.Series:
    df = pd.read_parquet(path, columns=["txn_id", column])
    return df.sort_values("txn_id")[column].reset_index(drop=True)


def main(dir_a: str, dir_b: str) -> None:
    cols_a = set(pq.ParquetDataset(dir_a).schema.names)
    cols_b = set(pq.ParquetDataset(dir_b).schema.names)
    common = sorted((cols_a & cols_b) - {"txn_id"})
    ids_a = np.sort(pd.read_parquet(dir_a, columns=["txn_id"])["txn_id"].to_numpy())
    ids_b = np.sort(pd.read_parquet(dir_b, columns=["txn_id"])["txn_id"].to_numpy())
    assert np.array_equal(ids_a, ids_b), "the two outputs contain different transactions"

    mismatched = []
    for col in common:
        x, y = read_column(dir_a, col), read_column(dir_b, col)
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            same = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=1e-5, atol=1e-2, equal_nan=True)
        else:
            same = bool((x.astype(str).to_numpy() == y.astype(str).to_numpy()).all())
        if not same:
            mismatched.append(col)
    print(f"rows={len(ids_a):,} columns compared={len(common)} "
          f"only_in_a={sorted(cols_a - cols_b)} only_in_b={sorted(cols_b - cols_a)} mismatched={mismatched}")
    sys.exit(1 if mismatched or cols_a != cols_b else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
