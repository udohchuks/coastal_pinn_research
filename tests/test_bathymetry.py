"""Tests for coastal_pinn.sources.bathymetry (post-fetch transformation only).

The network fetch is not exercised here. Tests verify _extract_points
against a real-shape NetCDF produced by the test fixture.
"""

from __future__ import annotations

import xarray as xr

from coastal_pinn import PipelineConfig, Region
from coastal_pinn.sources.bathymetry import _extract_points


def test_extract_points_zone_classification(keta_config, real_shape_bathy_nc):
    ds = xr.open_dataset(real_shape_bathy_nc)
    df = _extract_points(ds, keta_config)

    # All required columns present
    assert set(df.columns) >= {"region", "lon", "lat", "depth_m", "zone"}
    assert (df["region"] == "keta").all()

    # Zone classification: depth<0 -> sea, 0<=d<5 -> intertidal, d>=5 -> land
    sea = df.loc[df["zone"] == "sea", "depth_m"]
    intertidal = df.loc[df["zone"] == "intertidal", "depth_m"]
    land = df.loc[df["zone"] == "land", "depth_m"]

    if len(sea):
        assert (sea < 0).all()
    if len(intertidal):
        assert ((intertidal >= 0) & (intertidal < 5)).all()
    if len(land):
        assert (land >= 5).all()


def test_extract_points_lon_lat_in_bbox(keta_config, real_shape_bathy_nc):
    ds = xr.open_dataset(real_shape_bathy_nc)
    df = _extract_points(ds, keta_config)
    lon_min, lat_min, lon_max, lat_max = keta_config.region.bbox
    assert df["lon"].between(lon_min, lon_max).all()
    assert df["lat"].between(lat_min, lat_max).all()


def test_extract_points_no_null_depths(keta_config, real_shape_bathy_nc):
    ds = xr.open_dataset(real_shape_bathy_nc)
    df = _extract_points(ds, keta_config)
    assert df["depth_m"].notna().all()