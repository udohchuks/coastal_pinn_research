"""Atomic CSV / pickle / NetCDF I/O.

Atomicity: every write goes to <path>.tmp then os.replace(). On Windows
os.replace is atomic if both paths are on the same volume.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV atomically (via .tmp + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def write_pickle_atomic(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


def write_netcdf_atomic(ds: xr.Dataset, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    ds.to_netcdf(tmp, mode="w")
    os.replace(tmp, p)


def read_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def read_netcdf(path: str | Path) -> xr.Dataset:
    return xr.open_dataset(path)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)