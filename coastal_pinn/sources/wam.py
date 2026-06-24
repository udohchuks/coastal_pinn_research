"""Wave source: Copernicus Marine WAM (replaces NOAA WAVEWATCH III).

Uses the official copernicusmarine Python client with the WAM global
wave model. Reanalysis is used where available; analysis/forecast for
the most recent period. Resolution: 0.2 deg reanalysis / 0.083 deg
analysis, 3-hourly.

The reanalysis product ID is `cmems_mod_glo_wav_my_0.2deg_PT3H-i` (0.2 deg).
The analysis/forecast product ID is `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i`
(0.083 deg). Resolution differs between the two products.

Variables:
    VHM0: spectral significant wave height (m)
    VMDR: mean wave direction from (deg, meteorological convention)

v2 (per-transect): like sea_level.py, the (time, lat, lon) cube is
interpolated to the (lon, lat) of each transect generated from
cfg.region.baseline. This preserves along-shore variation in W and
W_dir. The significant wave height W feeds the Yates et al. (2009)
wave energy E = W**2 / 16 (derived downstream in reconcile()).

Returns a DataFrame with columns:
    region       str
    timestamp    pd.Timestamp, UTC
    transect_id  int
    W_m          float, significant wave height
    W_dir_deg    float, mean wave direction (meteorological, 0-360)
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from coastal_pinn.config import PipelineConfig
from coastal_pinn.core.coords import transects_to_lonlat
from coastal_pinn.core.paths import data_path
from coastal_pinn.core.schema import ensure_utc
from coastal_pinn.exceptions import MissingCredentials, SourceUnavailable
from coastal_pinn.sources.sea_level import _missing_credentials_message, _read_credentials
from coastal_pinn.sources.transects import generate_transects


WAM_REANALYSIS_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
WAM_ANALYSIS_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"

# Cutoff for switching from reanalysis to analysis/forecast (matches
# the convention used in sea_level.py for the PHY reanalysis cutoff).
WAM_REANALYSIS_END = datetime.datetime(2022, 6, 1, tzinfo=datetime.timezone.utc)


def fetch_wave_intensity(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch Copernicus WAM wave height and direction, per transect.

    Kept the function name `fetch_wave_intensity` to match the PipelineConfig
    field name; the underlying source is now Copernicus WAM, not WAVEWATCH III.
    """
    if not cfg.wave_intensity_enabled:
        raise SourceUnavailable("wave_intensity", "disabled in config")

    creds = _read_credentials()
    if creds is None:
        raise MissingCredentials(_missing_credentials_message())

    cache = data_path(cfg, "wave_intensity", suffix="nc")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print("[wave_intensity  ] downloading Copernicus WAM (VHM0, VMDR)...", flush=True)
        try:
            _download_wam(cfg, creds, cache)
        except Exception as e:
            raise SourceUnavailable("wave_intensity",
                f"failed to download Copernicus WAM for {cfg.region.name}: {e}",
                cause=e) from e
        print("[wave_intensity  ] download complete, interpolating to transects...", flush=True)
    else:
        print("[wave_intensity  ] cache hit, interpolating to transects...", flush=True)

    ds = xr.open_dataset(cache)
    try:
        return _to_dataframe(ds, cfg)
    finally:
        ds.close()


def _download_wam(cfg: PipelineConfig, creds: tuple[str, str], out_path: Path) -> None:
    """Call copernicusmarine.subset for VHM0 and VMDR.

    Handles the multi-year case where the time window spans the
    reanalysis/analysis cutoff (~2022-06-01) by downloading from both
    products and merging along the time dimension.
    """
    import copernicusmarine

    user, pwd = creds
    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox

    tmp_path = out_path.with_suffix(".tmp.nc")
    if tmp_path.exists():
        tmp_path.unlink()

    start_dt = cfg.t_start_dt
    end_dt = cfg.t_end_dt
    cutoff = pd.Timestamp("2022-06-01", tz="UTC")

    variables = ["VHM0", "VMDR"]
    spans_cutoff = (start_dt < cutoff) and (end_dt > cutoff)

    if spans_cutoff:
        part1 = out_path.with_suffix(".part1.nc")
        part2 = out_path.with_suffix(".part2.nc")
        for p in (part1, part2):
            if p.exists():
                p.unlink()
        copernicusmarine.subset(
            dataset_id=WAM_REANALYSIS_ID,
            variables=variables,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min,  maximum_latitude=lat_max,
            start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
            output_filename=str(part1),
            username=user, password=pwd,
        )
        copernicusmarine.subset(
            dataset_id=WAM_ANALYSIS_ID,
            variables=variables,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min,  maximum_latitude=lat_max,
            start_datetime=cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            output_filename=str(part2),
            username=user, password=pwd,
        )
        ds1 = xr.open_dataset(part1)
        ds2 = xr.open_dataset(part2)
        try:
            merged = xr.concat([ds1, ds2], dim="time")
            _, unique_idx = np.unique(merged["time"].values, return_index=True)
            merged = merged.isel(time=np.sort(unique_idx))
            merged.to_netcdf(tmp_path)
        finally:
            ds1.close()
            ds2.close()
            for p in (part1, part2):
                try: p.unlink()
                except OSError: pass
    else:
        dataset_id = WAM_REANALYSIS_ID if start_dt < cutoff else WAM_ANALYSIS_ID
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min,  maximum_latitude=lat_max,
            start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            output_filename=str(tmp_path),
            username=user, password=pwd,
        )
    os.replace(tmp_path, out_path)


def _to_dataframe(ds: xr.Dataset, cfg: PipelineConfig) -> pd.DataFrame:
    """Reduce the (time, lat, lon) cube to per-transect values.

    Resamples to daily means (the cube is 3-hourly). For direction,
    uses circular mean via sin/cos components.
    """
    for v in ["VHM0", "VMDR"]:
        if v not in ds.data_vars:
            raise SourceUnavailable("wave_intensity",
                f"missing variable {v!r} in cached dataset (vars={list(ds.data_vars)})")

    # Sample waves at each transect's seaward end (open water), not at the
    # inland baseline origin (see sea_level.py for rationale).
    transects_df = generate_transects(cfg.region)
    from coastal_pinn.core.coords import transect_sample_points
    raw_lons, raw_lats = transect_sample_points(transects_df, cfg.region.utm_zone)
    from coastal_pinn.core.coords import clamp_query_to_data_range
    clamped_lons, clamped_lats = clamp_query_to_data_range(raw_lons, raw_lats, ds)
    lon_pts = xr.DataArray(clamped_lons, dims="points")
    lat_pts = xr.DataArray(clamped_lats, dims="points")

    # Interpolate
    from coastal_pinn.core.coords import safe_interp
    h = safe_interp(ds["VHM0"], lon_pts, lat_pts)
    d = safe_interp(ds["VMDR"], lon_pts, lat_pts)

    # Daily resample using xarray before flattening
    d_rad = np.deg2rad(d)
    sin_d = np.sin(d_rad)
    cos_d = np.cos(d_rad)
    
    h_daily = h.resample(time="1D").mean()
    sin_d_daily = sin_d.resample(time="1D").mean()
    cos_d_daily = cos_d.resample(time="1D").mean()
    d_daily = (np.rad2deg(np.arctan2(sin_d_daily, cos_d_daily)) % 360.0)
    
    # Convert to numpy
    h_arr = np.asarray(h_daily.values, dtype=float)
    d_arr_deg = np.asarray(d_daily.values, dtype=float)
    
    time_daily = pd.to_datetime(h_daily.time.values, utc=True).normalize()
    n_time_daily = len(time_daily)
    n_pts = len(transects_df)
    
    daily = pd.DataFrame({
        "timestamp": np.tile(time_daily, n_pts),
        "transect_id": np.repeat(transects_df["transect_id"].values, n_time_daily),
        "W_m": h_arr.ravel(order="F"),
        "W_dir_deg": d_arr_deg.ravel(order="F"),
    })
    
    daily["region"] = cfg.region.name
    daily["timestamp"] = ensure_utc(daily["timestamp"])
    return daily[["region", "timestamp", "transect_id", "W_m", "W_dir_deg"]]
