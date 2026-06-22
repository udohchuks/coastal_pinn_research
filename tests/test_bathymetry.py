"""Tests for coastal_pinn.sources.bathymetry (post-fetch transformation only).

The network fetch is not exercised here. Tests verify _extract_per_transect
against a real-shape NetCDF produced by the test fixture.
"""

from __future__ import annotations

import xarray as xr

from coastal_pinn import PipelineConfig, Region
from coastal_pinn.sources.bathymetry import _extract_per_transect


def test_extract_per_transect_columns(keta_config, real_shape_bathy_nc):
    ds = xr.open_dataset(real_shape_bathy_nc)
    df = _extract_per_transect(ds, keta_config)
    assert set(df.columns) >= {"region", "transect_id", "depth_m"}
    assert (df["region"] == "keta").all()
    # Should have one row per transect
    n_transects = len(df)
    assert n_transects > 1
    assert df["transect_id"].nunique() == n_transects


def test_extract_per_transect_depths_in_range(keta_config, real_shape_bathy_nc):
    ds = xr.open_dataset(real_shape_bathy_nc)
    df = _extract_per_transect(ds, keta_config)
    # Sampling is at the seaward (offshore) end of each transect, so depth_m
    # should be seafloor: negative (below MSL), deepening offshore.
    assert df["depth_m"].notna().all()
    assert (df["depth_m"] < 0).all()
    # Should span a reasonable range across the along-shore array
    assert df["depth_m"].max() - df["depth_m"].min() > 1.0


def test_extract_per_transect_no_null_depths(keta_config, real_shape_bathy_nc):
    ds = xr.open_dataset(real_shape_bathy_nc)
    df = _extract_per_transect(ds, keta_config)
    assert df["depth_m"].notna().all()
