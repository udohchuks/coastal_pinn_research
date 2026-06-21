"""Wide-table schema and validation.

PINN_COLUMNS is the single source of truth for the output table's columns.
PINN_REQUIRED_COLUMNS excludes R_sediment_m_yr (placeholder, currently NaN).
validate_schema raises SchemaError if a DataFrame doesn't conform.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from coastal_pinn.exceptions import SchemaError


PINN_COLUMNS: list[str] = [
    "region",
    "timestamp",
    "easting_m",
    "northing_m",
    "h_m",
    "u_mag_m_s",
    "W_m",
    "W_dir_deg",
    "depth_at_shore_m",
    "R_sediment_m_yr",
]

PINN_REQUIRED_COLUMNS: list[str] = [c for c in PINN_COLUMNS if c != "R_sediment_m_yr"]


_DTYPE_CHECKS: dict[str, set[str]] = {
    "region": {"object", "string"},
    "timestamp": {"datetime64[ns, UTC]", "datetime64[ns]"},
    "easting_m": {"float64", "float32"},
    "northing_m": {"float64", "float32"},
    "h_m": {"float64", "float32"},
    "u_mag_m_s": {"float64", "float32"},
    "W_m": {"float64", "float32"},
    "W_dir_deg": {"float64", "float32"},
    "depth_at_shore_m": {"float64", "float32"},
    "R_sediment_m_yr": {"float64", "float32"},
}


def validate_schema(df: pd.DataFrame, *, allow_missing: Iterable[str] = ()) -> None:
    """Raise SchemaError if df is missing required columns or has wrong dtypes.

    Timestamp must be tz-aware UTC; this is enforced here as the contract
    every fetch_* function upholds, and the reconcile step relies on.
    """
    missing = set(PINN_REQUIRED_COLUMNS) - set(df.columns) - set(allow_missing)
    if missing:
        raise SchemaError(f"wide table missing required columns: {sorted(missing)}")

    # Check timestamp tz-awareness
    if "timestamp" in df.columns and len(df) > 0:
        ts_dtype = str(df["timestamp"].dtype)
        # xarray/NetCDF can produce datetime64[ns] without tz; require UTC awareness.
        # Accept any tz-aware dtype (datetime64[ns, UTC], datetime64[us, UTC], etc.)
        if "UTC" not in ts_dtype and "tz" not in ts_dtype.lower():
            raise SchemaError(
                "timestamp column must be tz-aware (UTC). "
                f"Got dtype={ts_dtype}. Use pd.to_datetime(..., utc=True) at the fetch boundary."
            )

    # region must be string-like (object, string, or str alias)
    if "region" in df.columns and not pd.api.types.is_string_dtype(df["region"]):
        raise SchemaError(f"column 'region' must be string-like; got {df['region'].dtype}")

    # numeric columns must be numeric
    NUMERIC_COLS = [c for c in PINN_COLUMNS if c not in ("region", "timestamp")]
    for col in NUMERIC_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"column {col!r} must be numeric; got {df[col].dtype}")


def ensure_utc(ts: "pd.Timestamp | pd.DatetimeIndex | pd.Series") -> "pd.Timestamp | pd.DatetimeIndex | pd.Series":
    """Coerce a Timestamp/DatetimeIndex/Series to UTC tz-aware at ns resolution.

    This is the standard helper every fetch_* function uses to normalize
    its timestamps before returning. merge_asof will throw a hard error
    if tz-aware and tz-naive DatetimeIndexes are mixed, or if two tz-aware
    columns differ in dtype resolution (e.g. ns vs us); this normalizes
    everything to datetime64[ns, UTC].
    """
    NS = "ns"
    if isinstance(ts, pd.Timestamp):
        localized = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        return localized.as_unit(NS)
    if isinstance(ts, pd.DatetimeIndex):
        localized = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        return localized.as_unit(NS)
    if isinstance(ts, pd.Series):
        localized = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
        return localized.dt.as_unit(NS)
    raise TypeError(f"ensure_utc: unsupported type {type(ts).__name__}")