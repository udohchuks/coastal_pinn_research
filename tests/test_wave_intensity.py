"""Tests for coastal_pinn.sources.wave_intensity (post-fetch transformation only).

The implementation now lives in coastal_pinn.sources.wam (Copernicus WAM);
this re-export keeps the import path stable.
"""

from __future__ import annotations

import xarray as xr

from coastal_pinn import PipelineConfig
from coastal_pinn.sources.wave_intensity import _to_dataframe


def test_waves_columns(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    assert set(df.columns) >= {"region", "timestamp", "transect_id", "W_m", "W_dir_deg"}
    assert (df["region"] == "keta").all()


def test_waves_timestamps_are_utc(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_waves_resampled_to_daily(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    # 8 3-hourly samples per day, 5 days -> at most 5 daily rows per transect
    n_transects = df["transect_id"].nunique()
    n_unique_days = df["timestamp"].dt.date.nunique()
    assert n_unique_days <= 5
    # rows = transects * days
    assert len(df) == n_transects * n_unique_days


def test_waves_direction_wrapped_to_0_360(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    df = _to_dataframe(ds, keta_config)
    assert (df["W_dir_deg"] >= 0).all()
    assert (df["W_dir_deg"] < 360).all()


def test_waves_direction_circular_mean(keta_config):
    """Verify that circular mean of directions is computed correctly, particularly near boundaries."""
    import numpy as np
    import pandas as pd
    from coastal_pinn.sources.transects import generate_transects
    from coastal_pinn.core.coords import utm_to_lonlat
    # 8 time points (3-hourly) so the daily resample has 8 substeps per day.
    # All 8 substeps in a single day: 4 at 350, 4 at 10 -> circular mean 0.
    times = pd.date_range("2018-01-01 00:00:00", periods=8, freq="3h")
    # Use the transect lons as data lons. The fetcher now clamps query
    # lons to the data range, so no wide-lon workaround is needed.
    transects = generate_transects(keta_config.region)
    transect_lons, _ = utm_to_lonlat(
        transects["origin_x"].tolist(), transects["origin_y"].tolist(),
        keta_config.region.utm_zone,
    )
    lons = np.sort(transect_lons)
    n_lons = len(lons)
    vhm0 = np.ones((8, 1, n_lons), dtype="float32")
    vmdr = np.zeros((8, 1, n_lons), dtype="float32")
    # First 4 substeps: 350; last 4 substeps: 10
    vmdr[0:4, 0, :] = 350.0
    vmdr[4:8, 0, :] = 10.0
    ds = xr.Dataset(
        {
            "VHM0": (("time", "latitude", "longitude"), vhm0),
            "VMDR":  (("time", "latitude", "longitude"), vmdr),
        },
        coords={"time": times, "latitude": [6.05], "longitude": lons},
    )
    df = _to_dataframe(ds, keta_config)
    # Single daily row per transect; circular mean of [350]*4 + [10]*4 = 0
    n_transects = df["transect_id"].nunique()
    assert n_transects >= 1
    for _, row in df.iterrows():
        v = row["W_dir_deg"]
        assert not np.isnan(v), f"NaN W_dir_deg for transect {row['transect_id']}"
        assert abs(v - 0.0) < 1e-4 or abs(v - 360.0) < 1e-4
