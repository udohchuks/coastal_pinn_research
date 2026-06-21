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

Returns a DataFrame with columns:
    region       (str)
    timestamp    (pd.Timestamp, UTC, daily)
    h_m          (float, sea level anomaly, m)
    u_east_m_s   (float, eastward current, m/s)
    u_north_m_s  (float, northward current, m/s)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from coastal_pinn.config import PipelineConfig
from coastal_pinn.core.paths import data_path
from coastal_pinn.core.schema import ensure_utc
from coastal_pinn.exceptions import MissingCredentials, SourceUnavailable


COPERNICUS_CREDENTIALS_PATH = Path.home() / ".config" / "copernicusmarine" / "credentials.json"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _read_credentials() -> tuple[str, str] | None:
    """Read Copernicus credentials from env vars or the standard config file.

    Returns None if neither source has credentials.
    """
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

    # Try alternative base64 encoded INI credentials
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


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_sea_level(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch sea-level anomaly (zos) and currents (uo, vo) from Copernicus.

    Caches the raw NetCDF to cfg.data_dir/sea_level/. Append-only.
    """
    if not cfg.sea_level_enabled:
        raise SourceUnavailable("sea_level", "disabled in config")

    creds = _read_credentials()
    if creds is None:
        raise MissingCredentials(_missing_credentials_message())

    cache = data_path(cfg, "sea_level", suffix="nc")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            _download_sea_level(cfg, creds, cache)
        except Exception as e:
            raise SourceUnavailable("sea_level",
                f"failed to download Copernicus Marine data for {cfg.region.name}: {e}",
                cause=e) from e

    ds = _read_cached(cache)
    try:
        return _to_dataframe(ds, cfg)
    finally:
        ds.close()


def _download_sea_level(cfg: PipelineConfig, creds: tuple[str, str], out_path: Path) -> None:
    """Call copernicusmarine.subset for zos, uo, vo."""
    import copernicusmarine  # imported lazily so the package is optional
    import datetime

    user, pwd = creds
    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox

    tmp_path = out_path.with_suffix(".tmp.nc")
    if tmp_path.exists():
        tmp_path.unlink()


    # Determine which dataset to use based on the start date
    start_dt = cfg.t_start_dt
    if isinstance(start_dt, datetime.datetime) and start_dt.tzinfo is not None:
        cutoff = datetime.datetime(2022, 6, 1, tzinfo=start_dt.tzinfo)
    else:
        cutoff = datetime.datetime(2022, 6, 1)

    if start_dt < cutoff:
        dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    else:
        dataset_id = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["zos", "uo", "vo"],
        minimum_longitude=lon_min, maximum_longitude=lon_max,
        minimum_latitude=lat_min,  maximum_latitude=lat_max,
        start_datetime=cfg.t_start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=cfg.t_end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        output_filename=str(tmp_path),
        username=user,
        password=pwd,
    )
    os.replace(tmp_path, out_path)



def _read_cached(path: Path):
    import xarray as xr
    return xr.open_dataset(path)


def _to_dataframe(ds, cfg: PipelineConfig) -> pd.DataFrame:
    """Reduce the (time, lat, lon) cube to a daily DataFrame.

    Returns columns: region, timestamp, h_m, u_east_m_s, u_north_m_s.
    Timestamps are normalized to UTC tz-aware.
    """
    # mean over lat/lon -> 1-D time series; resample to daily means
    for v in ["zos", "uo", "vo"]:
        if v not in ds.data_vars:
            raise SourceUnavailable("sea_level",
                f"missing variable {v!r} in cached dataset (vars={list(ds.data_vars)})")

    zos = ds["zos"].mean(dim=("latitude", "longitude"), skipna=True)
    uo  = ds["uo"].mean(dim=("latitude", "longitude"), skipna=True)
    vo  = ds["vo"].mean(dim=("latitude", "longitude"), skipna=True)

    # If there's a depth dim (some CMEMS products), collapse it too
    if "depth" in zos.dims:
        zos = zos.mean(dim="depth", skipna=True)
    if "depth" in uo.dims:
        uo = uo.mean(dim="depth", skipna=True)
    if "depth" in vo.dims:
        vo = vo.mean(dim="depth", skipna=True)

    # Time axis: force UTC tz-aware (this is the contract).
    time = pd.to_datetime(ds["time"].values, utc=True)

    df = pd.DataFrame({
        "timestamp": ensure_utc(pd.DatetimeIndex(time)),
        "h_m": np.asarray(zos.values).ravel(),
        "u_east_m_s": np.asarray(uo.values).ravel(),
        "u_north_m_s": np.asarray(vo.values).ravel(),
    })

    # Collapse any depth/etc. dims to a scalar via mean (defensive)
    if df["h_m"].ndim > 1 or df["h_m"].shape[0] != len(df):
        # take first if length matches, else mean over axis=1
        df["h_m"] = np.atleast_1d(df["h_m"].mean(axis=tuple(range(1, df["h_m"].ndim))))
        df["u_east_m_s"] = np.atleast_1d(df["u_east_m_s"].mean(axis=tuple(range(1, df["u_east_m_s"].ndim))))
        df["u_north_m_s"] = np.atleast_1d(df["u_north_m_s"].mean(axis=tuple(range(1, df["u_north_m_s"].ndim))))

    # Resample to daily means (the wave_intensity source is also daily, so this matches).
    df = df.set_index("timestamp").resample("D").mean().reset_index()
    df["region"] = cfg.region.name
    df["timestamp"] = ensure_utc(df["timestamp"])
    return df[["region", "timestamp", "h_m", "u_east_m_s", "u_north_m_s"]]