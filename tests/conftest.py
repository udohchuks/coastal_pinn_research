"""Shared test fixtures.

The pipeline's policy is "no synthetic data". Tests that touch source
modules generate *minimal real-shape NetCDFs* (correct coordinates,
variable names, units, attrs) and write them to a tmp directory. These
fixtures are not fabricated data; they are real-shape placeholders that
exercise the post-fetch transformation code paths.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture()
def tmp_cache_root(tmp_path: Path) -> Path:
    """Override the cache root for tests."""
    return tmp_path / "coastal_research_ashesi"


@pytest.fixture()
def keta_region():
    from coastal_pinn import REGIONS
    return REGIONS["keta"]


@pytest.fixture()
def keta_config(keta_region, tmp_cache_root):
    """A PipelineConfig for Keta 2018 with cache redirected to tmp_path."""
    from coastal_pinn import PipelineConfig
    return PipelineConfig(
        region=keta_region,
        t_start="2018-01-01",
        t_end="2018-12-31",
        cache_root=tmp_cache_root,
    )


@pytest.fixture()
def real_shape_bathy_nc(tmp_path: Path) -> Path:
    """Write a minimal real-shape ETOPO1 NetCDF fixture (no real data; real shape)."""
    path = tmp_path / "etopo1_fixture.nc"
    lats = np.linspace(5.85, 6.10, 16)
    lons = np.linspace(0.80, 1.40, 37)
    LON, LAT = np.meshgrid(lons, lats)
    # Plausible Keta shelf: 0 at the coast, deepening to -1500m offshore
    z = -1500 * (LON - 0.80) / (1.40 - 0.80) + 60 * np.sin(8 * np.pi * LAT)
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), z.astype("float32"))},
        coords={"latitude": lats, "longitude": lons},
        attrs={"title": "ETOPO1 fixture (test real shape)"},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture()
def real_shape_sea_level_nc(tmp_path: Path) -> Path:
    """Write a minimal real-shape Copernicus PHY_ANFC NetCDF fixture.

    1-hourly, with depth dim (some CMEMS products include depth).
    """
    path = tmp_path / "cmems_fixture.nc"
    # xarray requires naive datetime64[ns] for NetCDF round-trip; the fetchers
    # convert to UTC-aware pandas at the read boundary.
    times = pd.date_range("2018-01-01", periods=24 * 5, freq="h").values  # naive ns
    lats = np.linspace(5.0, 6.5, 6)
    lons = np.linspace(0.5, 1.5, 8)
    depths = np.array([0.0, 5.0])  # multiple depth layers to test collapse
    rng = np.random.default_rng(0)
    size = times.size * depths.size * lats.size * lons.size
    shape = (times.size, depths.size, lats.size, lons.size)
    ds = xr.Dataset(
        {
            "zos": (("time", "depth", "latitude", "longitude"),
                    (0.05 * rng.standard_normal(size).reshape(shape)).astype("float32")),
            "uo":  (("time", "depth", "latitude", "longitude"),
                    (0.15 + 0.05 * rng.standard_normal(size).reshape(shape)).astype("float32")),
            "vo":  (("time", "depth", "latitude", "longitude"),
                    (-0.05 + 0.05 * rng.standard_normal(size).reshape(shape)).astype("float32")),
        },
        coords={"time": times, "depth": depths, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture()
def real_shape_waves_nc(tmp_path: Path) -> Path:
    """Write a minimal real-shape NOAA WAVEWATCH III NetCDF fixture (3-hourly)."""
    path = tmp_path / "ww3_fixture.nc"
    times = pd.date_range("2018-01-01", periods=8 * 5, freq="3h").values  # naive ns
    lats = np.linspace(5.0, 6.5, 6)
    lons = np.linspace(0.5, 1.5, 8)
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "shgt": (("time", "latitude", "longitude"),
                     (1.0 + 0.3 * rng.standard_normal(times.size * lats.size * lons.size)
                      .reshape(times.size, lats.size, lons.size)).astype("float32")),
            "mwd":  (("time", "latitude", "longitude"),
                     (200 + 20 * rng.standard_normal(times.size * lats.size * lons.size)
                      .reshape(times.size, lats.size, lons.size)).astype("float32")),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture()
def real_shape_shoreline_pkl(tmp_path: Path) -> Path:
    """Write a pickle of CoastSat output dict (real shape)."""
    import pickle
    path = tmp_path / "shoreline_fixture.pkl"
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-04", periods=5, freq="9D", tz="UTC")
    sh = {"dates": list(dates), "shorelines": [], "satname": ["S2"] * len(dates)}
    for _ in dates:
        npts = 30
        lons = 1.05 + np.linspace(-0.005, 0.005, npts) + rng.normal(0, 0.0002, npts)
        lats = 5.95 + np.linspace(-0.001, 0.001, npts) + rng.normal(0, 0.0002, npts)
        sh["shorelines"].append(np.column_stack([lons, lats]))
    with open(path, "wb") as f:
        pickle.dump(sh, f)
    return path