"""Tests for coastal_pinn.core.schema — wide-table validation."""

from __future__ import annotations

import pandas as pd
import pytest

from coastal_pinn.core.schema import (
    PINN_COLUMNS,
    PINN_REQUIRED_COLUMNS,
    ensure_utc,
    validate_schema,
)
from coastal_pinn.exceptions import SchemaError


def _good_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["keta"] * 3,
        "timestamp": pd.to_datetime(["2018-01-01", "2018-01-15", "2018-02-01"], utc=True),
        "easting_m": [945281.0, 945270.0, 945260.0],
        "northing_m": [657084.0, 657080.0, 657071.0],
        "h_m": [0.1, 0.2, 0.15],
        "u_mag_m_s": [0.1, 0.12, 0.11],
        "W_m": [1.0, 1.1, 0.9],
        "W_dir_deg": [200.0, 210.0, 195.0],
        "depth_at_shore_m": [-10.0, -10.0, -10.0],
        "R_sediment_m_yr": [0.0, 0.0, 0.0],
    })


def test_validate_good_df_passes():
    validate_schema(_good_df())


def test_validate_rejects_naive_timestamp():
    df = _good_df()
    df["timestamp"] = pd.to_datetime(["2018-01-01", "2018-01-15", "2018-02-01"])
    with pytest.raises(SchemaError, match="UTC"):
        validate_schema(df)


def test_validate_rejects_missing_column():
    df = _good_df().drop(columns=["W_m"])
    with pytest.raises(SchemaError, match="W_m"):
        validate_schema(df)


def test_validate_accepts_missing_R_when_allowed():
    df = _good_df().drop(columns=["R_sediment_m_yr"])
    # R_sediment_m_yr is in PINN_COLUMNS but not PINN_REQUIRED_COLUMNS;
    # since it's optional in the wide table, this should still pass.
    validate_schema(df)


def test_validate_rejects_non_numeric():
    df = _good_df()
    df["h_m"] = ["a", "b", "c"]
    with pytest.raises(SchemaError, match="h_m"):
        validate_schema(df)


def test_ensure_utc_naive_to_aware():
    ts = pd.Timestamp("2024-01-01")
    out = ensure_utc(ts)
    assert out.tzinfo is not None
    assert str(out.tzinfo) == "UTC"


def test_ensure_utc_idempotent_on_aware():
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    out = ensure_utc(ts)
    assert str(out.tzinfo) == "UTC"


def test_ensure_utc_series():
    s = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-15"]))
    out = ensure_utc(s)
    assert out.dt.tz is not None
    assert str(out.dt.tz) == "UTC"


def test_pinn_required_columns_excludes_R():
    assert "R_sediment_m_yr" in PINN_COLUMNS
    assert "R_sediment_m_yr" not in PINN_REQUIRED_COLUMNS