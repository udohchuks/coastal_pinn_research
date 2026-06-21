"""Bathymetry source: GEBCO 2026 Global via NOAA CoastWatch ERDDAP.

Open-access, no auth. Returns a depth grid in (lon, lat, depth_m, zone)
form. Append-only cache to data_dir/bathymetry/.

GEBCO 2026 in NOAA ERDDAP is exposed as the `etopo1` grid (1-arc-minute
global relief, blended GEBCO/SRTM). This is the standard free path for
GEBCO-style depth data outside the GEBCO website's own subsetter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from coastal_pinn.config import PipelineConfig
from coastal_pinn.core.io import write_netcdf_atomic, read_netcdf
from coastal_pinn.core.paths import data_path, download_path
from coastal_pinn.exceptions import SourceUnavailable


# NOAA CoastWatch / AOML ERDDAP endpoints for the 1-arc-minute blended GEBCO/SRTM grid.
# Open-access; no credentials required. Returns a NetCDF file.
# Multiple mirrors are tried sequentially in case of server timeouts or issues.
ERDDAP_SOURCES = [
    ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo1.nc", "z"),
    ("https://cwcgom.aoml.noaa.gov/erddap/griddap/etopo180.nc", "altitude"),
    ("https://cwcgom.aoml.noaa.gov/erddap/griddap/etopo360.nc", "altitude"),
]



def fetch_bathymetry(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch GEBCO 2026 / ETOPO1 depth grid for cfg.region.bbox.

    Returns a long-format DataFrame with columns:
        region    (str)
        lon       (float, degrees)
        lat       (float, degrees)
        depth_m   (float, m; negative = below MSL)
        zone      ('sea' | 'intertidal' | 'land')

    Caches the raw NetCDF to cfg.data_dir/bathymetry/. Append-only: if a
    cached file exists for this (region, time-window) pair, it is reused
    and no network call is made. The bathymetry has no time axis, so the
    time window is encoded in the cache filename for provenance only.
    """
    if not cfg.bathymetry_enabled:
        raise SourceUnavailable("bathymetry", "disabled in config")

    cache = data_path(cfg, "bathymetry", suffix="nc")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_bathymetry(cfg, cache)
        except Exception as e:
            raise SourceUnavailable("bathymetry",
                f"failed to download GEBCO/ETOPO1 for {cfg.region.name}: {e}", cause=e) from e

    ds = read_netcdf(cache)
    try:
        df = _extract_points(ds, cfg)
    finally:
        ds.close()
    return df


def _download_bathymetry(cfg: PipelineConfig, out_path: Path) -> None:
    """Download the ETOPO1 grid restricted to cfg.region.bbox."""
    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox

    errors = []
    for base_url, var_name in ERDDAP_SOURCES:
        # Determine coordinate range normalization
        is_360 = "etopo360" in base_url
        if is_360:
            q_lon_min = lon_min % 360
            q_lon_max = lon_max % 360
        else:
            q_lon_min = (lon_min + 180) % 360 - 180
            q_lon_max = (lon_max + 180) % 360 - 180

        # Ensure correct ordering
        if q_lon_min > q_lon_max:
            q_lon_min, q_lon_max = q_lon_max, q_lon_min

        url = (
            f"{base_url}?"
            f"{var_name}%5B({lat_min}):({lat_max})%5D%5B({q_lon_min}):({q_lon_max})%5D"
        )
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_bytes(r.content)
            import os
            os.replace(tmp_path, out_path)
            return
        except Exception as e:
            errors.append(f"{base_url} ({var_name}): {e}")

    raise RuntimeError("All bathymetry ERDDAP sources failed:\n" + "\n".join(errors))



def _extract_points(ds: xr.Dataset, cfg: PipelineConfig) -> pd.DataFrame:
    """Convert the (lat, lon, z) grid to a long DataFrame with a 'zone' label.

    Zone rules (depth_m is elevation, negative = below MSL):
        sea:        depth_m <  0
        intertidal: 0 <= depth_m <  5
        land:       depth_m >= 5
    """
    # Find the elevation variable name (etopo1 uses 'z'; gebco may differ)
    var_candidates = ["z", "elevation", "altitude", "depth"]
    var = next((v for v in var_candidates if v in ds.data_vars), None)
    if var is None:
        raise SourceUnavailable("bathymetry",
            f"could not find depth variable in dataset; data_vars={list(ds.data_vars)}")

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    z = ds[var].values  # shape: (lat, lon)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lon = lon_grid.ravel()
    flat_lat = lat_grid.ravel()
    flat_z = np.asarray(z).ravel()

    df = pd.DataFrame({
        "region": cfg.region.name,
        "lon": flat_lon.astype(float),
        "lat": flat_lat.astype(float),
        "depth_m": flat_z.astype(float),
    })
    df["zone"] = np.where(df["depth_m"] < 0, "sea",
                  np.where(df["depth_m"] < 5, "intertidal", "land"))
    return df.dropna(subset=["depth_m"]).reset_index(drop=True)