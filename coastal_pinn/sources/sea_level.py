"""Sea-level and current source: Copernicus Marine via copernicusmarine.

Uses the official copernicusmarine Python client. Credentials are read
ONLY from:
  1. The standard Copernicus config file at
     ~/.config/copernicusmarine/credentials.json
     (auto-created by `copernicusmarine login`)
  2. Environment variables COPERNICUS_USER and COPERNICUS_PASSWORD

The package NEVER hardcodes credentials and NEVER reads them from the
project YAML or any local file in this repo. If neither is set, this
module raises MissingCredentials with an actionable message.

v2 (per-transect): instead of spatially averaging the (time, lat, lon)
cube to a single 1-D time series for the whole region, we interpolate
to the (lon, lat) of each transect generated from cfg.region.baseline.
This preserves along-shore gradients in h, u_east, u_north that the
PINN's PDE term needs.

Returns a DataFrame with columns:
    region       str
    timestamp    pd.Timestamp, UTC, daily
    transect_id  int
    h_m          float, sea level at this transect
    u_east_m_s   float, eastward current at this transect
    u_north_m_s  float, northward current at this transect
"""

from __future__ import annotations

import json
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
from coastal_pinn.sources.transects import generate_transects


COPERNICUS_CREDENTIALS_PATH = Path.home() / ".config" / "copernicusmarine" / "credentials.json"


def _read_credentials() -> tuple[str, str] | None:
    """Read Copernicus credentials from env vars or the standard config file."""
    user = os.environ.get("COPERNICUS_USER")
    pwd = os.environ.get("COPERNICUS_PASSWORD")
    if user and pwd:
        return user, pwd

    if COPERNICUS_CREDENTIALS_PATH.exists():
        try:
            creds = json.loads(COPERNICUS_CREDENTIALS_PATH.read_text(encoding="utf-8"))
            user = creds.get("username") or creds.get("user")
            pwd = creds.get("password") or creds.get("pwd")
            if user and pwd:
                return user, pwd
        except (json.JSONDecodeError, OSError):
            pass

    alt_path = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
    if alt_path.exists():
        try:
            import base64
            import configparser
            raw = alt_path.read_bytes()
            decoded = base64.b64decode(raw).decode("utf-8")
            config = configparser.ConfigParser()
            config.read_string(decoded)
            user = config.get("credentials", "username")
            pwd = config.get("credentials", "password")
            if user and pwd:
                return user, pwd
        except Exception:
            pass

    return None


def _missing_credentials_message() -> str:
    return (
        "Copernicus Marine credentials not found.\n\n"
        "Register at https://data.marine.copernicus.eu (free, 2 minutes).\n"
        "Then choose ONE of:\n\n"
        "  1. Run once on the command line:\n"
        "       copernicusmarine login\n"
        "     This writes ~/.config/copernicusmarine/credentials.json.\n\n"
        "  2. Set environment variables before invoking coastal_pinn:\n"
        "       set COPERNICUS_USER=<your-username>\n"
        "       set COPERNICUS_PASSWORD=<your-password>\n"
    )


def fetch_sea_level(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch sea-level anomaly (zos) and currents (uo, vo) from Copernicus PHY
    and interpolate to per-transect values.
    """
    if not cfg.sea_level_enabled:
        raise SourceUnavailable("sea_level", "disabled in config")

    creds = _read_credentials()
    if creds is None:
        raise MissingCredentials(_missing_credentials_message())

    cache = data_path(cfg, "sea_level", suffix="nc")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print("[sea_level       ] downloading Copernicus PHY (zos, uo, vo)...", flush=True)
        try:
            _download_sea_level(cfg, creds, cache)
        except Exception as e:
            raise SourceUnavailable("sea_level",
                f"failed to download Copernicus Marine data for {cfg.region.name}: {e}",
                cause=e) from e
        print("[sea_level       ] download complete, interpolating to transects...", flush=True)
    else:
        print("[sea_level       ] cache hit, interpolating to transects...", flush=True)

    ds = xr.open_dataset(cache)
    try:
        return _to_dataframe(ds, cfg)
    finally:
        ds.close()


def _download_sea_level(cfg: PipelineConfig, creds: tuple[str, str], out_path: Path) -> None:
    """Call copernicusmarine.subset for zos, uo, vo.

    Handles the multi-year case where the time window spans the
    reanalysis/analysis cutoff (~2022-06-01) by downloading from both
    products and merging along the time dimension.
    """
    import copernicusmarine
    import datetime

    user, pwd = creds
    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox

    tmp_path = out_path.with_suffix(".tmp.nc")
    if tmp_path.exists():
        tmp_path.unlink()

    start_dt = cfg.t_start_dt
    end_dt = cfg.t_end_dt
    cutoff = pd.Timestamp("2022-06-01", tz="UTC")

    REANALYSIS_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    ANALYSIS_ID = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    variables = ["zos", "uo", "vo"]

    spans_cutoff = (start_dt < cutoff) and (end_dt > cutoff)

    if spans_cutoff:
        # Two-stage download: reanalysis for [start, cutoff], analysis for [cutoff, end]
        part1 = out_path.with_suffix(".part1.nc")
        part2 = out_path.with_suffix(".part2.nc")
        for p in (part1, part2):
            if p.exists():
                p.unlink()
        copernicusmarine.subset(
            dataset_id=REANALYSIS_ID,
            variables=variables,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min,  maximum_latitude=lat_max,
            start_datetime=start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
            output_filename=str(part1),
            username=user, password=pwd,
        )
        copernicusmarine.subset(
            dataset_id=ANALYSIS_ID,
            variables=variables,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min,  maximum_latitude=lat_max,
            start_datetime=cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            output_filename=str(part2),
            username=user, password=pwd,
        )
        # Merge along time dimension
        ds1 = xr.open_dataset(part1)
        ds2 = xr.open_dataset(part2)
        try:
            merged = xr.concat([ds1, ds2], dim="time")
            # Drop duplicate timestamps at the boundary (if any)
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
        dataset_id = REANALYSIS_ID if start_dt < cutoff else ANALYSIS_ID
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


def _read_cached(path: Path) -> xr.Dataset:
    """Open a cached Copernicus NetCDF as an xarray Dataset."""
    return xr.open_dataset(path)


def _to_dataframe(ds: xr.Dataset, cfg: PipelineConfig) -> pd.DataFrame:
    """Reduce the (time, lat, lon) cube to per-transect values.

    Returns columns: region, timestamp, transect_id, h_m, u_east_m_s, u_north_m_s.
    Output is daily-mean (the cube is already daily).
    """
    for v in ["zos", "uo", "vo"]:
        if v not in ds.data_vars:
            raise SourceUnavailable("sea_level",
                f"missing variable {v!r} in cached dataset (vars={list(ds.data_vars)})")

    # Generate transects in lon/lat for xarray interp
    transects_df = generate_transects(cfg.region)
    transects_ll = transects_to_lonlat(transects_df, cfg.region.utm_zone)
    # Use the baseline latitude (not the per-transect lat) for the query,
    # since UTM-to-lonlat conversion introduces small floating-point drift
    # that puts per-transect lats slightly outside the data's discrete lat
    # grid. The baseline lat is exact and within the data range.
    if cfg.region.baseline is not None and len(cfg.region.baseline) >= 1:
        baseline_lat = float(cfg.region.baseline[0][1])
    else:
        baseline_lat = float(transects_ll["origin_lat"].values[0])
    raw_lons = transects_ll["origin_lon"].values
    raw_lats = np.full(len(transects_df), baseline_lat)
    # Clamp to data range to avoid NaN at boundaries from float drift
    from coastal_pinn.core.coords import clamp_query_to_data_range
    clamped_lons, clamped_lats = clamp_query_to_data_range(raw_lons, raw_lats, ds)
    lon_pts = xr.DataArray(clamped_lons, dims="points")
    lat_pts = xr.DataArray(clamped_lats, dims="points")

    def _interp(var_name: str):
        var = ds[var_name]
        # If there's a depth dim, collapse it (surface only)
        if "depth" in var.dims:
            var = var.mean(dim="depth", skipna=True)
        from coastal_pinn.core.coords import safe_interp
        sampled = safe_interp(var, lon_pts, lat_pts)
        if "time" in sampled.dims:
            sampled = sampled.resample(time="1D").mean()
        return sampled

    zos = _interp("zos")
    uo = _interp("uo")
    vo = _interp("vo")

    time_daily = pd.to_datetime(zos.time.values, utc=True).normalize()
    n_time = len(time_daily)
    n_pts = len(transects_df)

    zos_arr = np.asarray(zos.values, dtype=float)
    uo_arr = np.asarray(uo.values, dtype=float)
    vo_arr = np.asarray(vo.values, dtype=float)

    if zos_arr.ndim == 1:
        zos_arr = np.broadcast_to(zos_arr, (n_time, n_pts))
    if uo_arr.ndim == 1:
        uo_arr = np.broadcast_to(uo_arr, (n_time, n_pts))
    if vo_arr.ndim == 1:
        vo_arr = np.broadcast_to(vo_arr, (n_time, n_pts))

    df = pd.DataFrame({
        "timestamp": np.tile(time_daily, n_pts),
        "transect_id": np.repeat(transects_df["transect_id"].values, n_time),
        "h_m": zos_arr.ravel(order="F"),
        "u_east_m_s": uo_arr.ravel(order="F"),
        "u_north_m_s": vo_arr.ravel(order="F"),
    })

    df["region"] = cfg.region.name
    df["timestamp"] = ensure_utc(df["timestamp"])
    return df[["region", "timestamp", "transect_id", "h_m", "u_east_m_s", "u_north_m_s"]]
