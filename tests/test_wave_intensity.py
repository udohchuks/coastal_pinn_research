"""Tests for coastal_pinn.sources.wave_intensity (post-fetch transformation only)."""

from __future__ import annotations

import xarray as xr

from coastal_pinn import PipelineConfig
from coastal_pinn.sources.wave_intensity import _to_dataframe


def test_waves_columns(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    assert set(df.columns) >= {"region", "timestamp", "W_m", "W_dir_deg"}
    assert (df["region"] == "keta").all()


def test_waves_timestamps_are_utc(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_waves_resampled_to_daily(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    # 8 3-hourly samples per day, 5 days -> at most 5 daily rows
    assert len(df) <= 5


def test_waves_direction_wrapped_to_0_360(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    assert (df["W_dir_deg"] >= 0).all()
    assert (df["W_dir_deg"] < 360).all()


def test_waves_direction_circular_mean(keta_config):
    """Verify that circular mean of directions is computed correctly, particularly near boundaries."""
    import numpy as np
    import pandas as pd
    times = pd.date_range("2018-01-01 00:00:00", periods=2, freq="12h")
    # two points: 350 deg and 10 deg. Linear mean would be 180. Circular mean is 0 (or 360).
    ds = xr.Dataset(
        {
            "shgt": (("time", "latitude", "longitude"),
                     np.array([[[1.0]], [[1.0]]]).astype("float32")),
            "mwd":  (("time", "latitude", "longitude"),
                     np.array([[[350.0]], [[10.0]]]).astype("float32")),
        },
        coords={"time": times, "latitude": [5.9], "longitude": [1.0]},
    )
    df = _to_dataframe(ds, keta_config)
    # The output should have W_dir_deg = 0.0 (or very close to it, within floating point tolerance)
    assert len(df) == 1
    val = df.loc[0, "W_dir_deg"]
    assert abs(val - 0.0) < 1e-4 or abs(val - 360.0) < 1e-4