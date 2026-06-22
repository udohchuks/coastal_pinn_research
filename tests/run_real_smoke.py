"""Run the smoke test with a full-year sea_level/waves dataset.

Produces the complete Keta 2018 PINN wide CSV under the real cache root
using real-shape fixtures covering the full year (so the asof-join does
not drop rows due to time-range mismatch).

Run:  python tests/run_real_smoke.py
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coastal_pinn import PipelineConfig, REGIONS
from coastal_pinn.pipeline import reconcile
from coastal_pinn.sources.bathymetry import _extract_per_transect
from coastal_pinn.sources.sea_level import _to_dataframe as sl_to_df
from coastal_pinn.sources.shoreline import _intersect_with_transects as shore_to_df
from coastal_pinn.sources.wave_intensity import _to_dataframe as wv_to_df


def _make_bathy(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    lats = np.linspace(5.85, 6.10, 16)
    lons = np.linspace(0.80, 1.40, 37)
    LON, LAT = np.meshgrid(lons, lats)
    z = -1500 * (LON - 0.80) / (1.40 - 0.80) + 60 * np.sin(8 * np.pi * LAT)
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), z.astype("float32"))},
        coords={"latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path); ds.close()
    ds = xr.open_dataset(path)
    try:
        return _extract_per_transect(ds, cfg)
    finally:
        ds.close()


def _make_sea_level(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    """Full year of hourly sea level + currents (8760 hours)."""
    times = pd.date_range(cfg.t_start, cfg.t_end, freq="h").values
    lats = np.linspace(5.0, 6.5, 6)
    lons = np.linspace(0.5, 1.5, 8)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "zos": (("time", "latitude", "longitude"),
                    (0.05 * rng.standard_normal(times.size * lats.size * lons.size)
                     .reshape(times.size, lats.size, lons.size)).astype("float32")),
            "uo":  (("time", "latitude", "longitude"),
                    (0.15 + 0.05 * rng.standard_normal(times.size * lats.size * lons.size)
                     .reshape(times.size, lats.size, lons.size)).astype("float32")),
            "vo":  (("time", "latitude", "longitude"),
                    (-0.05 + 0.05 * rng.standard_normal(times.size * lats.size * lons.size)
                     .reshape(times.size, lats.size, lons.size)).astype("float32")),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path); ds.close()
    ds = xr.open_dataset(path)
    try:
        return sl_to_df(ds, cfg)
    finally:
        ds.close()


def _make_waves(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    """Full year of 3-hourly wave intensity (2920 timesteps)."""
    times = pd.date_range(cfg.t_start, cfg.t_end, freq="3h").values
    lats = np.linspace(5.0, 6.5, 6)
    lons = np.linspace(0.5, 1.5, 8)
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "VHM0": (("time", "latitude", "longitude"),
                     (1.0 + 0.3 * rng.standard_normal(times.size * lats.size * lons.size)
                      .reshape(times.size, lats.size, lons.size)).astype("float32")),
            "VMDR": (("time", "latitude", "longitude"),
                     (200 + 20 * rng.standard_normal(times.size * lats.size * lons.size)
                      .reshape(times.size, lats.size, lons.size)).astype("float32")),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path); ds.close()
    ds = xr.open_dataset(path)
    try:
        return wv_to_df(ds, cfg)
    finally:
        ds.close()


def _make_shoreline(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    """41 cloud-free dates across the full year."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(cfg.t_start, cfg.t_end, freq="9D", tz="UTC")
    sh = {"dates": list(dates), "shorelines": [], "satname": ["S2"] * len(dates)}
    from coastal_pinn.core.coords import lonlat_to_utm
    # Synthetic shoreline ~200 m seaward (south) of a stretch of the onshore
    # baseline so the south-pointing transects intersect it; UTM since
    # CoastSat returns polylines in output_epsg = region.epsg.
    base = np.array(cfg.region.baseline)[15:45]
    lat_shift = 200.0 / 111_000.0
    for _ in dates:
        jitter = rng.normal(0, 0.0002, base.shape)
        lons = base[:, 0] + jitter[:, 0]
        lats = base[:, 1] - lat_shift + jitter[:, 1]
        x, y = lonlat_to_utm(lons, lats, cfg.region.utm_zone)
        sh["shorelines"].append(np.column_stack([x, y]))
    import pickle
    with open(path, "wb") as f:
        pickle.dump(sh, f)
    return shore_to_df(pd.read_pickle(path), cfg)


if __name__ == "__main__":
    # Use a dedicated throwaway cache root so this smoke run can never wipe
    # the real download cache (coastal_research_ashesi/), which may hold
    # hours of real CoastSat/Copernicus downloads.
    cache_root = ROOT / "coastal_research_ashesi_smoke"
    if cache_root.exists():
        shutil.rmtree(cache_root)

    cfg = PipelineConfig(
        region=REGIONS["keta"],
        t_start="2018-01-01",
        t_end="2018-12-31",
        cache_root=cache_root,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    print("[smoke] building real-shape fixtures (no network)...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bathy = _make_bathy(cfg, td / "bathy.nc")
        gc.collect()
        sl    = _make_sea_level(cfg, td / "sl.nc")
        gc.collect()
        wv    = _make_waves(cfg, td / "wv.nc")
        gc.collect()
        shore = _make_shoreline(cfg, td / "shore.pkl")
        gc.collect()

    print(f"[smoke]   bathy:        {len(bathy)} points")
    print(f"[smoke]   sea_level:    {len(sl)} daily rows")
    print(f"[smoke]   wave:         {len(wv)} daily rows")
    print(f"[smoke]   shoreline:    {len(shore)} polyline vertices across "
          f"{shore['timestamp'].nunique()} unique dates")
    print(f"[smoke] running reconcile() -> {cfg.cache_root}")
    wide = reconcile(cfg, bathy, sl, wv, shore)
    print()
    print("=" * 70)
    print(f"[smoke] WIDE TABLE: {len(wide)} rows x {len(wide.columns)} cols")
    print("=" * 70)
    print(wide.head().to_string())
    print()
    print("describe():")
    print(wide.describe().to_string())
    print()
    print(f"[smoke] SUCCESS - produced {len(wide)} PINN-ready rows for {cfg.region.name}")
    print(f"[smoke] CSV: {next(cfg.output_dir.glob('*.csv'))}")
    print(f"[smoke] PKL: {next(cfg.output_dir.glob('*.pkl'))}")