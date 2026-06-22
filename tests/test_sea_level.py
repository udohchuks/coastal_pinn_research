"""Tests for coastal_pinn.sources.sea_level (post-fetch transformation only).

Verifies that the cached NetCDF is correctly reduced to a per-(transect, day)
UTC-aware DataFrame with h_m, u_east_m_s, u_north_m_s, transect_id columns.
"""

from __future__ import annotations

import xarray as xr

from coastal_pinn import PipelineConfig
from coastal_pinn.sources.sea_level import _read_cached, _to_dataframe


def test_sea_level_to_dataframe_columns(keta_config, real_shape_sea_level_nc):
    ds = _read_cached(real_shape_sea_level_nc)
    df = _to_dataframe(ds, keta_config)
    assert set(df.columns) >= {"region", "timestamp", "transect_id",
                               "h_m", "u_east_m_s", "u_north_m_s"}
    assert (df["region"] == "keta").all()


def test_sea_level_timestamps_are_utc(keta_config, real_shape_sea_level_nc):
    ds = _read_cached(real_shape_sea_level_nc)
    df = _to_dataframe(ds, keta_config)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_sea_level_has_per_transect_rows(keta_config, real_shape_sea_level_nc):
    ds = _read_cached(real_shape_sea_level_nc)
    df = _to_dataframe(ds, keta_config)
    # Per-transect output: more than just per-date
    n_transects = df["transect_id"].nunique()
    n_days = df["timestamp"].dt.date.nunique()
    # at least one row per (transect, day) pair
    assert len(df) >= n_transects * n_days


def test_sea_level_missing_zos_raises(keta_config, tmp_cache_root):
    import numpy as np
    import pandas as pd
    times = pd.date_range("2018-01-01", periods=4, freq="h")
    ds = xr.Dataset(
        {"uo": ("time", np.zeros(4)), "vo": ("time", np.zeros(4))},
        coords={"time": times, "latitude": [5.0], "longitude": [0.5]},
    )
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
    from coastal_pinn.sources.transects import generate_transects
    from coastal_pinn.core.coords import utm_to_lonlat
    times = pd.date_range("2018-01-01", periods=2, freq="h")
    # depth layers: one has value 10, the other has value 20 -> mean should be 15.
    # Use the transect lons as data lons. The fetcher now clamps query lons
    # to the data range, so no monkey-patching is needed.
    transects = generate_transects(keta_config.region)
    transect_lons, _ = utm_to_lonlat(
        transects["origin_x"].tolist(), transects["origin_y"].tolist(),
        keta_config.region.utm_zone,
    )
    # Sort lons ascending (xarray requires monotonic coords)
    lons = np.sort(transect_lons)
    n_lons = len(lons)
    val = np.zeros((2, 2, 1, n_lons), dtype="float32")
    val[0, 0, 0, :] = 10.0
    val[0, 1, 0, :] = 20.0
    val[1, 0, 0, :] = 10.0
    val[1, 1, 0, :] = 20.0
    ds = xr.Dataset(
        {
            "zos": (("time", "depth", "latitude", "longitude"), val),
            "uo":  (("time", "depth", "latitude", "longitude"), val),
            "vo":  (("time", "depth", "latitude", "longitude"), val),
        },
        coords={"time": times, "depth": [0.0, 5.0], "latitude": [6.05], "longitude": lons},
    )
    df = _to_dataframe(ds, keta_config)
    # depth is collapsed to mean 15.0 before per-transect interpolation
    # All transects should get 15.0 (query lons are clamped to data range)
    assert np.allclose(df["h_m"], 15.0)
    assert np.allclose(df["u_east_m_s"], 15.0)
    assert np.allclose(df["u_north_m_s"], 15.0)
