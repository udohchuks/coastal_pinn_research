"""Pipeline orchestration.

run(cfg)        runs all enabled sources for one region and returns the wide table.
run([cfg, ...]) runs multiple regions and concatenates the per-region tables.
reconcile(...)  joins per-source DataFrames into the wide table (the contract).

All timestamps flowing into reconcile() are guaranteed UTC tz-aware by the
fetch_* functions. merge_asof will throw a hard error if any are naive;
this is the contract that every fetcher upholds.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from coastal_pinn.config import PipelineConfig, Region
from coastal_pinn.core.coords import depth_at_shore
from coastal_pinn.core.io import write_csv_atomic, write_pickle_atomic
from coastal_pinn.core.paths import pinn_wide_path
from coastal_pinn.core.schema import (
    PINN_COLUMNS,
    PINN_REQUIRED_COLUMNS,
    validate_schema,
)
from coastal_pinn.exceptions import SchemaError, SourceUnavailable
from coastal_pinn.sources.bathymetry import fetch_bathymetry
from coastal_pinn.sources.sea_level import fetch_sea_level
from coastal_pinn.sources.wave_intensity import fetch_wave_intensity
from coastal_pinn.sources.shoreline import fetch_shorelines
from coastal_pinn.sources.sediment_recovery import compute_sediment_recovery


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_region(cfg: PipelineConfig, *, skip_sources: Iterable[str] = (),
               offline: bool = False) -> pd.DataFrame:
    """Run all enabled sources for one region and return the wide table.

    skip_sources: a list of source names to skip even if enabled.
        Useful for partial smoke tests ('give me everything except shoreline').
    offline: if True, fail fast on cache miss instead of fetching from network.
    """
    bathy_df = _run_one(fetch_bathymetry, cfg, "bathymetry", skip_sources, offline)
    sea_level_df = _run_one(fetch_sea_level, cfg, "sea_level", skip_sources, offline)
    waves_df = _run_one(fetch_wave_intensity, cfg, "wave_intensity", skip_sources, offline)
    shore_df = _run_one(fetch_shorelines, cfg, "shoreline", skip_sources, offline)

    return reconcile(cfg, bathy_df, sea_level_df, waves_df, shore_df)


def run(configs: list[PipelineConfig], *, offline: bool = False,
        continue_on_error: bool = False) -> pd.DataFrame:
    """Run for multiple regions and concatenate the per-region wide tables.

    Returns a single DataFrame with all regions' rows stacked. The 'region'
    column distinguishes them.
    """
    tables: list[pd.DataFrame] = []
    errors: list[tuple[str, Exception]] = []
    for cfg in configs:
        try:
            tables.append(run_region(cfg, offline=offline))
        except Exception as e:
            if continue_on_error:
                errors.append((cfg.region.name, e))
                print(f"[warn] region {cfg.region.name} failed: {e}")
            else:
                raise

    if not tables:
        if errors:
            raise RuntimeError(
                f"all {len(errors)} region(s) failed; first error: {errors[0][1]}"
            )
        raise RuntimeError("no regions supplied")

    wide = pd.concat(tables, ignore_index=True)
    return wide


def build_from_cache(configs: list[PipelineConfig], out_csv: Path | None = None
                     ) -> pd.DataFrame:
    """Build the wide table from cache only (no network).

    If out_csv is given, also writes the concatenated CSV there.
    """
    wide = run(configs, offline=True)
    if out_csv is not None:
        write_csv_atomic(wide, out_csv)
    return wide


def reconcile(cfg: PipelineConfig,
              bathy_df: pd.DataFrame,
              sea_level_df: pd.DataFrame,
              waves_df: pd.DataFrame,
              shoreline_df: pd.DataFrame) -> pd.DataFrame:
    """Join per-source DataFrames into the PINN wide table.

    Steps:
      1. Aggregate shoreline per (region, timestamp) -> scalar (mean Easting, mean Northing)
      2. Resample sea_level / waves to daily means (already daily from fetchers)
      3. asof-join shoreline <- sea_level, shoreline <- waves, with cfg.join_tolerance
      4. Attach scalar depth_at_shore_m from bathymetry (broadcast)
      5. Leave R_sediment_m_yr as NaN (sediment_recovery not implemented)
      6. Drop rows with any NaN in PINN_REQUIRED_COLUMNS
      7. Validate schema, write CSV + pkl to cfg.output_dir
    """
    from coastal_pinn.core.schema import ensure_utc
    if shoreline_df.empty:
        raise SourceUnavailable("shoreline",
            f"no shoreline observations for {cfg.region.name} in [{cfg.t_start}, {cfg.t_end}]")

    # Normalize all timestamps to UTC tz-aware with consistent resolution
    # (datetime64[ns, UTC]). merge_asof requires matching dtypes.
    shoreline_df = shoreline_df.copy()
    shoreline_df["timestamp"] = ensure_utc(shoreline_df["timestamp"])
    sea_level_df = sea_level_df.copy()
    sea_level_df["timestamp"] = ensure_utc(sea_level_df["timestamp"])
    waves_df = waves_df.copy()
    waves_df["timestamp"] = ensure_utc(waves_df["timestamp"])

    # 1. shoreline -> scalar per (region, timestamp)
    s_scalar = (shoreline_df
                .groupby(["region", "timestamp"], as_index=False)
                .agg(easting_m=("easting_m", "mean"),
                     northing_m=("northing_m", "mean")))

    # 2. Sea level and waves: indexed by (region, timestamp), already daily
    sl = sea_level_df.set_index(["region", "timestamp"]).sort_index()
    wv = waves_df.set_index(["region", "timestamp"]).sort_index()

    # 3. asof-join shoreline -> sea_level and shoreline -> waves
    tolerance = pd.Timedelta(cfg.join_tolerance)

    s_scalar = s_scalar.sort_values("timestamp")
    sl_reset = sea_level_df.sort_values("timestamp")
    wv_reset = waves_df.sort_values("timestamp")

    merged = pd.merge_asof(
        s_scalar, sl_reset,
        on="timestamp", by="region",
        direction="nearest", tolerance=tolerance,
    )
    # Compute current speed magnitude from east/north components
    if "u_east_m_s" in merged.columns and "u_north_m_s" in merged.columns:
        merged["u_mag_m_s"] = np.sqrt(
            merged["u_east_m_s"].astype(float) ** 2
            + merged["u_north_m_s"].astype(float) ** 2
        )
    else:
        raise SchemaError(
            f"sea_level table missing u_east_m_s / u_north_m_s; got {list(merged.columns)}"
        )

    merged = pd.merge_asof(
        merged, wv_reset,
        on="timestamp", by="region",
        direction="nearest", tolerance=tolerance,
    )

    # 4. depth_at_shore_m scalar from bathy
    merged["depth_at_shore_m"] = depth_at_shore(bathy_df)

    # 5. R_sediment_m_yr is currently NaN (not implemented). Try anyway for
    # users who later wire in a real source; if it raises NotImplementedError,
    # leave NaN.
    try:
        dates_index = pd.DatetimeIndex(merged["timestamp"].unique())
        r_series = compute_sediment_recovery(cfg, dates_index)
        merged["R_sediment_m_yr"] = merged["timestamp"].map(
            lambda ts: r_series.get(ts, np.nan)
        )
    except NotImplementedError:
        merged["R_sediment_m_yr"] = np.nan

    # 6. Drop rows with missing required inputs. No fabrication.
    n_total = len(merged)
    merged = merged.dropna(subset=PINN_REQUIRED_COLUMNS).reset_index(drop=True)
    n_dropped = n_total - len(merged)
    if n_dropped > 0:
        warnings.warn(
            f"[{cfg.region.name}] dropped {n_dropped} of {n_total} rows due to "
            f"missing inputs after asof-join (tolerance={cfg.join_tolerance})",
            stacklevel=2,
        )

    # Final column order
    merged = merged[PINN_COLUMNS]

    # 7. Validate schema
    validate_schema(merged)

    # Write to disk
    csv_path = pinn_wide_path(cfg, suffix="csv")
    pkl_path = pinn_wide_path(cfg, suffix="pkl")
    write_csv_atomic(merged, csv_path)
    write_pickle_atomic(merged, pkl_path)
    print(f"[{cfg.region.name}] wrote {len(merged)} rows -> {csv_path}")

    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_one(fn, cfg: PipelineConfig, name: str,
             skip_sources: Iterable[str], offline: bool):
    if name in skip_sources:
        return _empty_for(name)
    if offline and not _cache_exists(cfg, name):
        raise SourceUnavailable(name,
            f"offline mode and no cache file for {name} in {cfg.data_dir}")
    return fn(cfg)


def _empty_for(name: str) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical columns for `name`."""
    if name == "bathymetry":
        return pd.DataFrame(columns=["region", "lon", "lat", "depth_m", "zone"])
    if name == "sea_level":
        return pd.DataFrame(columns=["region", "timestamp", "h_m",
                                     "u_east_m_s", "u_north_m_s"])
    if name == "wave_intensity":
        return pd.DataFrame(columns=["region", "timestamp", "W_m", "W_dir_deg"])
    if name == "shoreline":
        return pd.DataFrame(columns=["region", "timestamp", "sat", "pt_idx",
                                     "easting_m", "northing_m"])
    raise ValueError(f"unknown source: {name}")


def _cache_exists(cfg: PipelineConfig, name: str) -> bool:
    from coastal_pinn.core.paths import data_path
    suffix = "pkl" if name == "shoreline" else "nc"
    return data_path(cfg, name, suffix=suffix).exists()