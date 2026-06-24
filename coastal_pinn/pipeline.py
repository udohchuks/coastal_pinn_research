"""Pipeline orchestration (v2, per-transect One-Line Model).

run(cfg)        runs all enabled sources for one region and returns the wide table.
run([cfg, ...]) runs multiple regions and concatenates the per-region tables.
reconcile(...)  joins per-source DataFrames into the wide table (the contract).

Schema: per-(transect, date) long format. ~150,000 rows for Keta 2018-2025.

All timestamps flowing into reconcile() are guaranteed UTC tz-aware by the
fetch_* functions. merge_asof will throw a hard error if any are naive.

All four sources are fetched in parallel via ThreadPoolExecutor. Progress
is printed to stdout so the user can see which source is downloading,
processing, or done.
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from coastal_pinn.config import PipelineConfig, Region
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


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _log(source: str, msg: str, *, end: str = "\n") -> None:
    """Print a progress line tagged with the source name."""
    print(f"[{source:16s}] {msg}", end=end, flush=True)


def _run_one_logged(fn, cfg: PipelineConfig, name: str,
                    skip_sources: Iterable[str], offline: bool) -> pd.DataFrame:
    """Run a single fetcher with progress logging and timing."""
    if name in skip_sources:
        _log(name, "skipped (in skip_sources)")
        return _empty_for(name)
    if offline and not _cache_exists(cfg, name):
        raise SourceUnavailable(name,
            f"offline mode and no cache file for {name} in {cfg.data_dir}")

    cache_file = _cache_path(cfg, name)
    from_cache = cache_file.exists() if cache_file else False

    _log(name, f"starting ({'from cache' if from_cache else 'downloading'})...")
    t0 = time.time()
    try:
        df = fn(cfg)
        elapsed = time.time() - t0
        _log(name, f"done: {len(df)} rows in {elapsed:.1f}s")
        return df
    except Exception as e:
        elapsed = time.time() - t0
        _log(name, f"FAILED after {elapsed:.1f}s: {e}")
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_region(cfg: PipelineConfig, *, skip_sources: Iterable[str] = (),
               offline: bool = False) -> pd.DataFrame:
    """Run all enabled sources for one region and return the wide table.

    All four sources are fetched in parallel. The forcing sources
    (bathymetry, sea_level, wave_intensity) typically finish in seconds;
    the shoreline source (CoastSat + GEE) takes much longer.

    skip_sources: a list of source names to skip even if enabled.
    offline: if True, fail fast on cache miss instead of fetching from network.
    """
    sources = {
        "bathymetry": fetch_bathymetry,
        "sea_level": fetch_sea_level,
        "wave_intensity": fetch_wave_intensity,
        "shoreline": fetch_shorelines,
    }

    skip_set = set(skip_sources)
    results: dict[str, pd.DataFrame] = {}

    _log("pipeline", f"running {len(sources)} sources for {cfg.region.name} "
         f"[{cfg.t_start} to {cfg.t_end}] in parallel...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_name = {
            executor.submit(
                _run_one_logged, fn, cfg, name, skip_set, offline
            ): name
            for name, fn in sources.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                # Cancel remaining futures
                for f in future_to_name:
                    f.cancel()
                raise SourceUnavailable(name, f"source {name} failed: {e}", cause=e) from e

    _log("pipeline", "all sources fetched, reconciling...")

    return reconcile(
        cfg,
        results["bathymetry"],
        results["sea_level"],
        results["wave_intensity"],
        results["shoreline"],
    )


def run(configs: list[PipelineConfig], *, offline: bool = False,
        continue_on_error: bool = False) -> pd.DataFrame:
    """Run for multiple regions and concatenate the per-region wide tables."""
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
    """Build the wide table from cache only (no network)."""
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
      1. Normalize all timestamps to UTC tz-aware.
      2. For each (transect_id, shoreline_timestamp), find nearest sea level
         and waves within cfg.join_tolerance (per-transect asof join).
      3. Compute u_mag from u_east and u_north.
      4. Compute E_wave = W_m**2 / 16 (Yates 2009 wave energy).
      5. Attach per-transect depth_m from bathymetry.
      6. Drop rows with any NaN in PINN_REQUIRED_COLUMNS.
      7. Validate schema, write CSV + pkl to cfg.output_dir.
    """
    from coastal_pinn.core.schema import ensure_utc
    if shoreline_df.empty:
        raise SourceUnavailable("shoreline",
            f"no shoreline observations for {cfg.region.name} in [{cfg.t_start}, {cfg.t_end}]")

    # Normalize timestamps
    shoreline_df = shoreline_df.copy()
    shoreline_df["timestamp"] = ensure_utc(shoreline_df["timestamp"])
    sea_level_df = sea_level_df.copy()
    sea_level_df["timestamp"] = ensure_utc(sea_level_df["timestamp"])
    waves_df = waves_df.copy()
    waves_df["timestamp"] = ensure_utc(waves_df["timestamp"])

    # Sort all by timestamp (required for merge_asof with by=)
    shoreline_df = shoreline_df.sort_values("timestamp")
    sea_level_df = sea_level_df.sort_values("timestamp")
    waves_df = waves_df.sort_values("timestamp")

    tolerance = pd.Timedelta(cfg.join_tolerance)

    # Vectorized as-of join using pandas native by= parameter.
    # Both dataframes are sorted by timestamp (required by merge_asof);
    # the by="transect_id" groups the join per-transect in one call.
    merged = _per_transect_asof(
        shoreline_df, sea_level_df,
        on="timestamp", by="transect_id",
        tolerance=tolerance,
    )
    # After merge, drop duplicate transect_id column (if added) and
    # ensure single canonical name
    if "transect_id_x" in merged.columns:
        merged = merged.drop(columns=["transect_id_x"])
    if "transect_id_y" in merged.columns:
        merged = merged.rename(columns={"transect_id_y": "transect_id"})

    # Compute current speed magnitude
    if "u_east_m_s" in merged.columns and "u_north_m_s" in merged.columns:
        merged["u_mag_m_s"] = np.sqrt(
            merged["u_east_m_s"].astype(float) ** 2
            + merged["u_north_m_s"].astype(float) ** 2
        )
    else:
        raise SchemaError(
            f"sea_level table missing u_east_m_s / u_north_m_s; got {list(merged.columns)}"
        )

    merged = _per_transect_asof(
        merged, waves_df,
        on="timestamp", by="transect_id",
        tolerance=tolerance,
    )
    # After second merge, drop duplicate transect_id column again
    if "transect_id_x" in merged.columns:
        merged = merged.drop(columns=["transect_id_x"])
    if "transect_id_y" in merged.columns:
        merged = merged.rename(columns={"transect_id_y": "transect_id"})

    # Per-transect depth (static)
    if "depth_m" in bathy_df.columns:
        depth_by_transect = bathy_df.set_index("transect_id")["depth_m"]
        merged["depth_m"] = merged["transect_id"].map(depth_by_transect)
    else:
        raise SchemaError(
            f"bathymetry table missing depth_m; got {list(bathy_df.columns)}"
        )

    # E_wave: the Yates et al. (2009) wave energy, E proportional to the
    # significant wave height squared. Yates scales it as E = H_s**2 / 16
    # (a wave energy of ~0.05 m^2 corresponds to H_s = 0.9 m). This is the
    # only model-specific quantity the data provides; the coefficients
    # C_pm, E_eq, and the Vitousek (2017) trend term v are all learned by
    # the PINN. See data.md §7.
    merged["E_wave"] = merged["W_m"].astype(float) ** 2 / 16.0

    # Drop rows with missing required inputs
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

    # Validate schema
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
        return pd.DataFrame(columns=["region", "transect_id", "depth_m"])
    if name == "sea_level":
        return pd.DataFrame(columns=["region", "timestamp", "transect_id",
                                     "h_m", "u_east_m_s", "u_north_m_s"])
    if name == "wave_intensity":
        return pd.DataFrame(columns=["region", "timestamp", "transect_id",
                                     "W_m", "W_dir_deg"])
    if name == "shoreline":
        return pd.DataFrame(columns=["region", "timestamp", "transect_id",
                                     "along_shore_x_m", "cross_shore_S_m", "sat"])
    raise ValueError(f"unknown source: {name}")


def _cache_path(cfg: PipelineConfig, name: str) -> Path | None:
    """Return the cache file path for a source, or None if not applicable."""
    from coastal_pinn.core.paths import data_path
    suffix = "pkl" if name == "shoreline" else "nc"
    return data_path(cfg, name, suffix=suffix)


def _cache_exists(cfg: PipelineConfig, name: str) -> bool:
    p = _cache_path(cfg, name)
    return p.exists() if p else False


def _per_transect_asof(left: pd.DataFrame, right: pd.DataFrame,
                       *, on: str, by: str,
                       tolerance: pd.Timedelta) -> pd.DataFrame:
    """Per-transect merge_asof using pandas native by= parameter."""
    return pd.merge_asof(
        left, right,
        on=on, by=by,
        direction="nearest", tolerance=tolerance,
    )
