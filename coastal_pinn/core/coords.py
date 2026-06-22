"""Coordinate conversions: lon/lat <-> UTM, transect utilities, field interpolation."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
import pyproj
import xarray as xr


@lru_cache(maxsize=16)
def make_transformer(utm_zone: str) -> tuple[pyproj.Transformer, pyproj.Transformer]:
    """Build a (lonlat_to_utm, utm_to_lonlat) pair for the given UTM zone string.

    Cached: creating Transformers is expensive (~50 ms each) and the same
    zone is reused across all sources and transect computations.
    """
    import re
    m = re.fullmatch(r"\s*(\d+)\s*([NnSs])\s*", utm_zone)
    if not m:
        raise ValueError(f"invalid UTM zone: {utm_zone!r}")
    num = int(m.group(1))
    hemi = m.group(2).upper()
    base = 32600 if hemi == "N" else 32700
    epsg = base + num
    fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return fwd, inv


def lonlat_to_utm(lon: Iterable[float], lat: Iterable[float], utm_zone: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized lon/lat (WGS84) -> UTM easting/northing (m)."""
    fwd, _ = make_transformer(utm_zone)
    x, y = fwd.transform(list(lon), list(lat))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def utm_to_lonlat(easting: Iterable[float], northing: Iterable[float], utm_zone: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized UTM easting/northing -> lon/lat (WGS84)."""
    _, inv = make_transformer(utm_zone)
    lon, lat = inv.transform(list(easting), list(northing))
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def transects_to_lonlat(transects: pd.DataFrame, utm_zone: str
                         ) -> pd.DataFrame:
    """Convert transect origin (and end) from UTM to (lon, lat) for fetcher use.

    Adds columns: origin_lon, origin_lat, end_lon, end_lat.
    """
    out = transects.copy()
    out["origin_lon"], out["origin_lat"] = utm_to_lonlat(
        out["origin_x"].tolist(), out["origin_y"].tolist(), utm_zone
    )
    out["end_lon"], out["end_lat"] = utm_to_lonlat(
        out["end_x"].tolist(), out["end_y"].tolist(), utm_zone
    )
    return out


def clamp_query_to_data_range(
    query_lons: np.ndarray,
    query_lats: np.ndarray,
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp query lon/lat values to the data's coordinate range.

    UTM-to-lonlat conversion introduces small floating-point drift that
    can put query points just outside the data's discrete grid (e.g.,
    1.40000001 vs 1.39999999). xarray.interp returns NaN for points
    outside the data range. This function clamps the query to the data's
    min/max for each coordinate, ensuring boundary transects get the
    edge value instead of NaN.
    """
    lon_min = float(np.nanmin(ds["longitude"].values))
    lon_max = float(np.nanmax(ds["longitude"].values))
    lat_min = float(np.nanmin(ds["latitude"].values))
    lat_max = float(np.nanmax(ds["latitude"].values))
    clamped_lons = np.clip(query_lons, lon_min, lon_max)
    clamped_lats = np.clip(query_lats, lat_min, lat_max)
    return clamped_lons, clamped_lats


def safe_interp(
    var: xr.DataArray,
    lon_pts: xr.DataArray,
    lat_pts: xr.DataArray,
) -> xr.DataArray:
    """Interpolate with linear method, fall back to nearest if ANY NaN.

    When the data has too few grid cells for linear interpolation (e.g.,
    a single cell in one dimension), linear interp returns all NaN.
    This function tries linear first, and if the result has NaNs,
    retries with method='nearest'.

    If nearest also has NaNs (e.g., the query point coincides with a
    land grid cell that has NaN data), the data is forward-filled along
    the spatial dimensions before retrying nearest. This ensures that
    land NaN values get the nearest ocean value.

    Optimization: the NaN check uses .values to avoid triggering a
    dask compute, and short-circuits via .any() on the flattened array.
    The forward-fill path is skipped when the data has no NaN cells.
    """
    result = var.interp(longitude=lon_pts, latitude=lat_pts, method="linear")
    # .values forces computation (for dask); .any() short-circuits on first True
    if not result.isnull().values.any():
        return result

    # Linear has NaNs — try nearest
    result = var.interp(longitude=lon_pts, latitude=lat_pts, method="nearest")
    if not result.isnull().values.any():
        return result

    # Nearest also has NaNs — only happens when query points coincide with
    # NaN grid cells (e.g., land). Skip forward-fill if data has no NaN
    # cells at all (shouldn't reach here, but defensive).
    has_nan_cells = var.isnull().values.any()
    if not has_nan_cells:
        return result

    # Forward-fill the data along spatial dims so that land NaN values
    # inherit the nearest ocean value, then retry nearest.
    filled = var
    if "latitude" in filled.dims:
        filled = filled.ffill(dim="latitude").bfill(dim="latitude")
    if "longitude" in filled.dims:
        filled = filled.ffill(dim="longitude").bfill(dim="longitude")
    result = filled.interp(longitude=lon_pts, latitude=lat_pts, method="nearest")
    return result


def interp_field_to_transects(
    ds: xr.Dataset,
    var_name: str,
    transects_lonlat: pd.DataFrame,
) -> np.ndarray:
    """Interpolate a (time, lat, lon) field to per-transect values.

    Returns an array of shape (T, N) where T is the number of timesteps
    in the dataset and N is the number of transects. For static (no-time)
    fields, returns shape (N,).
    """
    if var_name not in ds.data_vars:
        raise KeyError(f"variable {var_name!r} not in dataset; have {list(ds.data_vars)}")
    var = ds[var_name]
    lons = transects_lonlat["origin_lon"].values
    lats = transects_lonlat["origin_lat"].values
    if "time" in var.dims:
        # (time, lat, lon) -> (time, N)
        sampled = var.interp(longitude=xr.DataArray(lons, dims="points"),
                              latitude=xr.DataArray(lats, dims="points"),
                              method="linear")
        return np.asarray(sampled.values, dtype=float)
    else:
        # static field (lat, lon) -> (N,)
        sampled = var.interp(longitude=xr.DataArray(lons, dims="points"),
                              latitude=xr.DataArray(lats, dims="points"),
                              method="linear")
        return np.asarray(sampled.values, dtype=float)


def compute_wave_longshore(
    W_m: np.ndarray,
    W_dir_deg: np.ndarray,
    shore_normal_deg: np.ndarray,
) -> np.ndarray:
    """Compute the CERC longshore wave component: W * sin(2*theta).

    W_m: (T, N) or (N,) array of wave heights.
    W_dir_deg: same shape, wave direction (meteorological, direction
        waves come FROM, 0=N, 90=E, 180=S, 270=W).
    shore_normal_deg: (N,) array, direction the transect points
        (seaward, degrees CCW from East).

    Returns the same shape as W_m: W * sin(2*theta) where theta is the
    angle between the wave direction and the shore-normal.

    Note: meteorological convention (waves coming FROM) means we convert
    the wave direction to "direction the wave is GOING TO" by adding 180,
    then measure the angle to the shore-normal.
    """
    W_dir_to = (W_dir_deg + 180.0) % 360.0
    # Make sure shore_normal is broadcast to W shape
    sn = np.asarray(shore_normal_deg, dtype=float)
    while sn.ndim < W_dir_to.ndim:
        sn = sn[np.newaxis, ...]
    theta = np.deg2rad(W_dir_to - sn)
    return W_m * np.sin(2.0 * theta)


def depth_at_shore(bathy_df: pd.DataFrame, *, depth_column: str = "depth_m",
                   zone_column: str = "zone") -> float:
    """DEPRECATED in v2: scalar depth for backward compatibility.

    In the new per-transect schema, depth is interpolated per transect
    (see interp_field_to_transects). This function is kept for any
    external scripts that still use it.
    """
    if zone_column not in bathy_df.columns or depth_column not in bathy_df.columns:
        return 0.0
    mask = bathy_df[zone_column].isin(["intertidal", "sea"])
    if not mask.any():
        return 0.0
    return float(bathy_df.loc[mask, depth_column].mean())
