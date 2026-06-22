"""Wide-table schema and validation.

PINN_COLUMNS is the single source of truth for the output table's columns
under the new multi-transect One-Line Model. Each row is one observation
at one (transect, date) pair.

The schema is per-transect (NOT per-date). For ~600 transects x 372 dates,
the table has ~150,000 rows.

Columns:
    region              str
    timestamp           datetime, UTC
    transect_id         int, 0..N-1
    along_shore_x_m     float, transect's along-shore position from
                        baseline origin
    cross_shore_S_m     float, target: observed shoreline cross-shore
                        distance from inland baseline
    h_m                 float, sea level at this transect and time
                        (interpolated from Copernicus PHY)
    u_mag_m_s           float, current speed at this transect and time
                        (interpolated from Copernicus PHY)
    W_m                 float, significant wave height at this transect
                        and time (interpolated from Copernicus WAM)
    W_dir_deg           float, wave direction (meteorological, 0-360)
                        at this transect and time
    W_longshore         float, DERIVED: W_m * sin(2*theta) where theta
                        is the angle of wave incidence relative to the
                        local shore-normal. CERC longshore transport
                        factor.
    depth_m             float, GEBCO depth at this transect (static)
    R_sediment_m_yr     float, LEARNED by network closure. NaN at fetch
                        time; the model infers it.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from coastal_pinn.exceptions import SchemaError


PINN_COLUMNS: list[str] = [
    "region",
    "timestamp",
    "transect_id",
    "along_shore_x_m",
    "cross_shore_S_m",
    "h_m",
    "u_mag_m_s",
    "W_m",
    "W_dir_deg",
    "W_longshore",
    "depth_m",
    "R_sediment_m_yr",
]

# R_sediment_m_yr is allowed to be NaN; the others are required.
PINN_REQUIRED_COLUMNS: list[str] = [c for c in PINN_COLUMNS if c != "R_sediment_m_yr"]


_DTYPE_CHECKS: dict[str, set[str]] = {
    "region": {"object", "string"},
    "timestamp": {"datetime64[ns, UTC]", "datetime64[ns]"},
    "transect_id": {"int64", "int32", "int"},
    "along_shore_x_m": {"float64", "float32"},
    "cross_shore_S_m": {"float64", "float32"},
    "h_m": {"float64", "float32"},
    "u_mag_m_s": {"float64", "float32"},
    "W_m": {"float64", "float32"},
    "W_dir_deg": {"float64", "float32"},
    "W_longshore": {"float64", "float32"},
    "depth_m": {"float64", "float32"},
    "R_sediment_m_yr": {"float64", "float32"},
}


def validate_schema(df: pd.DataFrame, *, allow_missing: Iterable[str] = ()) -> None:
    """Raise SchemaError if df doesn't conform to the PINN wide-table schema."""
    missing = set(PINN_REQUIRED_COLUMNS) - set(df.columns) - set(allow_missing)
    if missing:
        raise SchemaError(f"wide table missing required columns: {sorted(missing)}")

    if "timestamp" in df.columns and len(df) > 0:
        ts_dtype = str(df["timestamp"].dtype)
        if "UTC" not in ts_dtype and "tz" not in ts_dtype.lower():
            raise SchemaError(
                "timestamp column must be tz-aware (UTC). "
                f"Got dtype={ts_dtype}. Use pd.to_datetime(..., utc=True) at the fetch boundary."
            )

    if "region" in df.columns and not pd.api.types.is_string_dtype(df["region"]):
        raise SchemaError(f"column 'region' must be string-like; got {df['region'].dtype}")

    NUMERIC_COLS = [c for c in PINN_COLUMNS if c not in ("region", "timestamp", "transect_id")]
    for col in NUMERIC_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"column {col!r} must be numeric; got {df[col].dtype}")

    if "transect_id" in df.columns and not pd.api.types.is_integer_dtype(df["transect_id"]):
        raise SchemaError(
            f"column 'transect_id' must be integer; got {df['transect_id'].dtype}"
        )


def ensure_utc(ts: "pd.Timestamp | pd.DatetimeIndex | pd.Series") -> "pd.Timestamp | pd.DatetimeIndex | pd.Series":
    """Coerce a Timestamp/DatetimeIndex/Series to UTC tz-aware at ns resolution."""
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
