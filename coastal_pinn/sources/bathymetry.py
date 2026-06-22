"""Bathymetry source: GEBCO 2026 Global via NOAA CoastWatch ERDDAP.

Open-access, no auth. Returns depth values interpolated to the (lon, lat)
of each transect generated from cfg.region.baseline. This replaces the
v1 scalar `depth_at_shore_m` with a per-transect `depth_m` profile,
which lets the PINN learn spatially-varying depth effects on the
closure R_theta.

Returns a DataFrame with columns:
    region       str
    transect_id  int
    depth_m      float, elevation relative to MSL (m, +above, -below)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from coastal_pinn.config import PipelineConfig
from coastal_pinn.core.coords import transects_to_lonlat
from coastal_pinn.core.io import write_netcdf_atomic, read_netcdf
from coastal_pinn.core.paths import data_path
from coastal_pinn.exceptions import SourceUnavailable
from coastal_pinn.sources.transects import generate_transects


# NOAA CoastWatch / AOML ERDDAP endpoints for the 1-arc-minute blended
# GEBCO/SRTM grid. Open-access; no credentials required. Returns a NetCDF.
# Multiple mirrors tried sequentially in case of server issues.
ERDDAP_SOURCES = [
    ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo1.nc", "z"),
    ("https://cwcgom.aoml.noaa.gov/erddap/griddap/etopo180.nc", "altitude"),
    ("https://cwcgom.aoml.noaa.gov/erddap/griddap/etopo360.nc", "altitude"),
]


def fetch_bathymetry(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch GEBCO / ETOPO1 and interpolate to per-transect depths.

    Returns a long-format DataFrame with columns:
        region, transect_id, depth_m

    Cached at cfg.data_dir/bathymetry/. Append-only.
    """
    if not cfg.bathymetry_enabled:
        raise SourceUnavailable("bathymetry", "disabled in config")

    cache = data_path(cfg, "bathymetry", suffix="nc")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print("[bathymetry      ] downloading GEBCO from NOAA ERDDAP...", flush=True)
        try:
            _download_bathymetry(cfg, cache)
        except Exception as e:
            raise SourceUnavailable("bathymetry",
                f"failed to download GEBCO/ETOPO1 for {cfg.region.name}: {e}",
                cause=e) from e
        print("[bathymetry      ] download complete, interpolating to transects...", flush=True)
    else:
        print("[bathymetry      ] cache hit, interpolating to transects...", flush=True)

    ds = read_netcdf(cache)
    try:
        df = _extract_per_transect(ds, cfg)
    finally:
        ds.close()
    return df


def _download_bathymetry(cfg: PipelineConfig, out_path: Path) -> None:
    """Download the ETOPO1 grid restricted to cfg.region.bbox."""
    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox

    errors = []
    for base_url, var_name in ERDDAP_SOURCES:
        is_360 = "etopo360" in base_url
        if is_360:
            q_lon_min = lon_min % 360
            q_lon_max = lon_max % 360
        else:
            q_lon_min = (lon_min + 180) % 360 - 180
            q_lon_max = (lon_max + 180) % 360 - 180

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


def _extract_per_transect(ds: xr.Dataset, cfg: PipelineConfig) -> pd.DataFrame:
    """Interpolate the (lat, lon) depth grid to per-transect values.

    Returns a DataFrame with one row per transect:
        region, transect_id, depth_m
    """
    var_candidates = ["z", "elevation", "altitude", "depth"]
    var = next((v for v in var_candidates if v in ds.data_vars), None)
    if var is None:
        raise SourceUnavailable("bathymetry",
            f"could not find depth variable in dataset; data_vars={list(ds.data_vars)}")

    transects_df = generate_transects(cfg.region)
    transects_ll = transects_to_lonlat(transects_df, cfg.region.utm_zone)

    if cfg.region.baseline is not None and len(cfg.region.baseline) >= 1:
        baseline_lat = float(cfg.region.baseline[0][1])
    else:
        baseline_lat = float(transects_ll["origin_lat"].values[0])
    raw_lons = transects_ll["origin_lon"].values
    raw_lats = np.full(len(transects_df), baseline_lat)
    from coastal_pinn.core.coords import clamp_query_to_data_range, safe_interp
    clamped_lons, clamped_lats = clamp_query_to_data_range(raw_lons, raw_lats, ds)
    lon_pts = xr.DataArray(clamped_lons, dims="points")
    lat_pts = xr.DataArray(clamped_lats, dims="points")

    sampled = safe_interp(ds[var], lon_pts, lat_pts)
    depths = np.asarray(sampled.values, dtype=float).ravel()

    df = pd.DataFrame({
        "region": cfg.region.name,
        "transect_id": transects_df["transect_id"].values,
        "depth_m": depths,
    })
    # Fill any NaN with the regional mean (defensive)
    if df["depth_m"].isna().any():
        df["depth_m"] = df["depth_m"].fillna(df["depth_m"].mean())
    return df.reset_index(drop=True)
