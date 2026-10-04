"""Choose the DataFrame library that backs each Dask partition.

pandas is the default and runs everywhere. On a Linux machine with RAPIDS
installed, `--engine cudf` makes every Dask partition a cuDF (GPU) DataFrame.
The partition functions in features.py only use operations both libraries
support, so the same code runs on either backend.
"""

from __future__ import annotations

import dask


def cudf_available() -> bool:
    try:
        import cudf  # noqa: F401
        import dask_cudf  # noqa: F401
    except ImportError:
        return False
    return True


def configure_backend(engine: str) -> str:
    """Set Dask's DataFrame backend and return the engine actually used."""
    if engine == "auto":
        engine = "cudf" if cudf_available() else "pandas"
    if engine == "cudf":
        if not cudf_available():
            print("cuDF / dask-cudf not importable; falling back to pandas")
            engine = "pandas"
        else:
            dask.config.set({"dataframe.backend": "cudf"})
    print(f"DataFrame engine: {engine}")
    return engine


def to_pandas(df):
    """Convert a cuDF frame to pandas; return pandas frames unchanged."""
    return df.to_pandas() if hasattr(df, "to_pandas") else df
