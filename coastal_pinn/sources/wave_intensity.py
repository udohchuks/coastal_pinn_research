"""Wave intensity source: NOAA WAVEWATCH III via NOAA ERDDAP.

Open-access, no auth. WAVEWATCH III is NOAA's global wave model; the
NOAA CoastWatch ERDDAP exposes a global grid of significant wave height
and mean wave direction.

For Gulf-of-Guinea regions, we use the global WAVEWATCH III grid (0.5°,
3-hourly, 2005-present). The pipeline subsets to cfg.region.bbox and
resamples to daily means.

Returns a DataFrame with columns:
    region       (str)
    timestamp    (pd.Timestamp, UTC, daily)
    W_m          (float, significant wave height, m)
    W_dir_deg    (float, mean wave direction, deg)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from coastal_pinn.config import PipelineConfig
from coastal_pinn.core.io import write_netcdf_atomic, read_netcdf
from coastal_pinn.core.paths import data_path
from coastal_pinn.core.schema import ensure_utc
from coastal_pinn.exceptions import SourceUnavailable


# NOAA CoastWatch / PacIOOS ERDDAP sources for WAVEWATCH III global wave height + direction.
# Open-access, no credentials required. Multiple mirrors are tried sequentially.
ERDDAP_SOURCES = [
    ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncep_global_wave.nc", "shgt", "mwd", False),
    ("https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.nc", "Thgt", "Tdir", True),
]



def fetch_wave_intensity(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch NOAA WAVEWATCH III wave height and direction for cfg.region.bbox.

    Caches the raw NetCDF to cfg.data_dir/wave_intensity/. Append-only.
    """
    if not cfg.wave_intensity_enabled:
        raise SourceUnavailable("wave_intensity", "disabled in config")

    cache = data_path(cfg, "wave_intensity", suffix="nc")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_waves(cfg, cache)
        except Exception as e:
            raise SourceUnavailable("wave_intensity",
                f"failed to download NOAA WAVEWATCH III for {cfg.region.name}: {e}",
                cause=e) from e

    ds = read_netcdf(cache)
    try:
        return _to_dataframe(ds, cfg)
    finally:
        ds.close()


def _download_waves(cfg: PipelineConfig, out_path: Path) -> None:
    """Download the WAVEWATCH III grid restricted to cfg.region.bbox + time window.

    ERDDAP griddap subset syntax:
        <var>[lo:hi][lo:hi]  for 2-D (lat, lon)
        <var>[t0:t1][lo:hi][lo:hi]  for 3-D (time, lat, lon)
    """
    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox
    t0 = cfg.t_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    t1 = cfg.t_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    errors = []
    for base_url, height_var, dir_var, has_depth in ERDDAP_SOURCES:
        if has_depth:
            url = (
                f"{base_url}?"
                f"{height_var}%5B({t0}):({t1})%5D%5B(0.0):(0.0)%5D%5B({lat_min}):({lat_max})%5D%5B({lon_min}):({lon_max})%5D,"
                f"{dir_var}%5B({t0}):({t1})%5D%5B(0.0):(0.0)%5D%5B({lat_min}):({lat_max})%5D%5B({lon_min}):({lon_max})%5D"
            )
        else:
            url = (
                f"{base_url}?"
                f"{height_var}%5B({t0}):({t1})%5D%5B({lat_min}):({lat_max})%5D%5B({lon_min}):({lon_max})%5D,"
                f"{dir_var}%5B({t0}):({t1})%5D%5B({lat_min}):({lat_max})%5D%5B({lon_min}):({lon_max})%5D"
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
            errors.append(f"{base_url}: {e}")

    raise RuntimeError("All wave intensity ERDDAP sources failed:\n" + "\n".join(errors))



def _to_dataframe(ds: xr.Dataset, cfg: PipelineConfig) -> pd.DataFrame:
    """Reduce the (time, lat, lon) cube to a daily-mean DataFrame."""
    # Find active variable names depending on which mirror was used
    var_height = next((v for v in ["shgt", "Thgt", "significant_wave_height"] if v in ds.data_vars), None)
    var_dir = next((v for v in ["mwd", "Tdir", "mean_wave_direction", "wave_direction"] if v in ds.data_vars), None)

    if var_height is None or var_dir is None:
        raise SourceUnavailable("wave_intensity",
            f"missing wave variables in cached dataset (vars={list(ds.data_vars)})")

    h = ds[var_height].mean(dim=("latitude", "longitude"), skipna=True)

    # Circular mean for wave direction over spatial dimensions
    rad = np.deg2rad(ds[var_dir])
    sin_spatial = np.sin(rad).mean(dim=("latitude", "longitude"), skipna=True)
    cos_spatial = np.cos(rad).mean(dim=("latitude", "longitude"), skipna=True)

    # Time axis must be UTC tz-aware for merge_asof downstream
    time = pd.to_datetime(ds["time"].values, utc=True)

    df = pd.DataFrame({
        "timestamp": ensure_utc(pd.DatetimeIndex(time)),
        "W_m": np.asarray(h.values).ravel(),
        "sin_dir": np.asarray(sin_spatial.values).ravel(),
        "cos_dir": np.asarray(cos_spatial.values).ravel(),
    })

    # Resample to daily means
    daily = df.set_index("timestamp").resample("D").mean().reset_index()
    daily["region"] = cfg.region.name
    daily["timestamp"] = ensure_utc(daily["timestamp"])

    # Reconstruct wave direction using circular mean from averaged sine/cosine components
    daily["W_dir_deg"] = np.rad2deg(np.arctan2(daily["sin_dir"], daily["cos_dir"])) % 360.0

    return daily[["region", "timestamp", "W_m", "W_dir_deg"]]