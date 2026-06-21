"""Tests for coastal_pinn.sources.sea_level (post-fetch transformation only).

Verifies that the cached NetCDF is correctly reduced to a daily UTC-aware
DataFrame with h_m, u_east_m_s, u_north_m_s columns.
"""

from __future__ import annotations

import xarray as xr

from coastal_pinn import PipelineConfig
from coastal_pinn.sources.sea_level import _read_cached, _to_dataframe


def test_sea_level_to_dataframe_columns(keta_config, real_shape_sea_level_nc):
    ds = _read_cached(real_shape_sea_level_nc)
    df = _to_dataframe(ds, keta_config)
    assert set(df.columns) >= {"region", "timestamp", "h_m",
                               "u_east_m_s", "u_north_m_s"}
    assert (df["region"] == "keta").all()


def test_sea_level_timestamps_are_utc(keta_config, real_shape_sea_level_nc):
    ds = _read_cached(real_shape_sea_level_nc)
    df = _to_dataframe(ds, keta_config)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_sea_level_resampled_to_daily(keta_config, real_shape_sea_level_nc):
    ds = _read_cached(real_shape_sea_level_nc)
    df = _to_dataframe(ds, keta_config)
    # 5 days of hourly data -> at most 5 daily rows
    assert len(df) <= 5
    # days are unique
    assert df["timestamp"].dt.date.nunique() == len(df)


def test_sea_level_missing_zos_raises(keta_config, tmp_cache_root):
    import numpy as np
    import pandas as pd
    # Build the dataset fully in memory and close it before passing to the
    # fetcher; on Windows xarray holds the file lock until close().
    times = pd.date_range("2018-01-01", periods=4, freq="h")  # naive ns
    ds = xr.Dataset(
        {"uo": ("time", np.zeros(4)), "vo": ("time", np.zeros(4))},
        coords={"time": times, "latitude": [5.0], "longitude": [0.5]},
    )
    # serialize then reopen via file path so the in-memory dataset can be GC'd
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp.close()
    try:
        ds.to_netcdf(tmp.name)
        ds.close()
        ds2 = xr.open_dataset(tmp.name)
        from coastal_pinn.exceptions import SourceUnavailable
        import pytest
        with pytest.raises(SourceUnavailable):
            _to_dataframe(ds2, keta_config)
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass


def test_sea_level_depth_collapse_averaging(keta_config):
    """Verify that multiple depth layers are collapsed via mean, not just ignored."""
    import numpy as np
    import pandas as pd
    times = pd.date_range("2018-01-01", periods=2, freq="h")
    # depth layers: one has value 10, the other has value 20 -> mean should be 15
    ds = xr.Dataset(
        {
            "zos": (("time", "depth", "latitude", "longitude"),
                    np.array([[[[10.0]], [[20.0]]], [[[10.0]], [[20.0]]]]).astype("float32")),
            "uo":  (("time", "depth", "latitude", "longitude"),
                    np.array([[[[10.0]], [[20.0]]], [[[10.0]], [[20.0]]]]).astype("float32")),
            "vo":  (("time", "depth", "latitude", "longitude"),
                    np.array([[[[10.0]], [[20.0]]], [[[10.0]], [[20.0]]]]).astype("float32")),
        },
        coords={"time": times, "depth": [0.0, 5.0], "latitude": [5.9], "longitude": [1.0]},
    )
    df = _to_dataframe(ds, keta_config)
    assert (df["h_m"] == 15.0).all()
    assert (df["u_east_m_s"] == 15.0).all()
    assert (df["u_north_m_s"] == 15.0).all()