"""Offline smoke test: prove the pipeline produces a valid PINN wide CSV.

This test uses the real-shape NetCDF fixtures and the real-shape CoastSat
pickle fixture, plus a tmp cache root. It does NOT touch the network,
GEE auth, or Copernicus credentials. It exists to confirm that the
reconcile() join logic, schema validation, and CSV/pkl writers all work
on real-shape data end-to-end.

Run directly:
    python tests/smoke_offline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Make the package importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coastal_pinn import PipelineConfig, REGIONS
from coastal_pinn.pipeline import reconcile
from coastal_pinn.sources.bathymetry import _extract_points
from coastal_pinn.sources.sea_level import _to_dataframe as sl_to_df
from coastal_pinn.sources.shoreline import _to_dataframe as shore_to_df
from coastal_pinn.sources.wave_intensity import _to_dataframe as wv_to_df
from coastal_pinn.core.io import read_csv
from coastal_pinn.core.schema import PINN_COLUMNS, PINN_REQUIRED_COLUMNS, validate_schema


def _make_bathy(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    lats = np.linspace(5.85, 6.10, 16)
    lons = np.linspace(0.80, 1.40, 37)
    LON, LAT = np.meshgrid(lons, lats)
    z = -1500 * (LON - 0.80) / (1.40 - 0.80) + 60 * np.sin(8 * np.pi * LAT)
    ds = xr.Dataset(
        {"z": (("latitude", "longitude"), z.astype("float32"))},
        coords={"latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)
    ds.close()
    return _extract_points(xr.open_dataset(path), cfg)


def _make_sea_level(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    times = pd.date_range("2018-01-01", periods=24 * 90, freq="h").values  # 90 days
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
    ds.to_netcdf(path)
    ds.close()
    return sl_to_df(xr.open_dataset(path), cfg)


def _make_waves(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    times = pd.date_range("2018-01-01", periods=8 * 90, freq="3h").values
    lats = np.linspace(5.0, 6.5, 6)
    lons = np.linspace(0.5, 1.5, 8)
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "shgt": (("time", "latitude", "longitude"),
                     (1.0 + 0.3 * rng.standard_normal(times.size * lats.size * lons.size)
                      .reshape(times.size, lats.size, lons.size)).astype("float32")),
            "mwd":  (("time", "latitude", "longitude"),
                     (200 + 20 * rng.standard_normal(times.size * lats.size * lons.size)
                      .reshape(times.size, lats.size, lons.size)).astype("float32")),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)
    ds.close()
    return wv_to_df(xr.open_dataset(path), cfg)


def _make_shoreline(cfg: PipelineConfig, path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    # 41 cloud-free dates across 2018 (matches the ~40/yr figure in the PDF)
    dates = pd.date_range("2018-01-01", "2018-12-31", freq="9D", tz="UTC")
    sh = {"dates": list(dates), "shorelines": [], "satname": ["S2"] * len(dates)}
    for _ in dates:
        npts = 30
        lons = 1.05 + np.linspace(-0.005, 0.005, npts) + rng.normal(0, 0.0002, npts)
        lats = 5.95 + np.linspace(-0.001, 0.001, npts) + rng.normal(0, 0.0002, npts)
        sh["shorelines"].append(np.column_stack([lons, lats]))
    import pickle
    with open(path, "wb") as f:
        pickle.dump(sh, f)
    return shore_to_df(pd.read_pickle(path), cfg)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = PipelineConfig(
            region=REGIONS["keta"],
            t_start="2018-01-01",
            t_end="2018-12-31",
            cache_root=tmp / "coastal_research_ashesi",
        )
        print(f"[smoke] cache root: {cfg.cache_root}")
        print(f"[smoke] region:     {cfg.region.name}, bbox={cfg.region.bbox}")
        print(f"[smoke] time:       {cfg.t_start} -> {cfg.t_end}")

        bathy = _make_bathy(cfg, tmp / "etopo.nc")
        print(f"[smoke] bathy:        {len(bathy)} points")
        sl    = _make_sea_level(cfg, tmp / "cmems.nc")
        print(f"[smoke] sea_level:    {len(sl)} daily rows")
        wv    = _make_waves(cfg, tmp / "ww3.nc")
        print(f"[smoke] wave:         {len(wv)} daily rows")
        shore = _make_shoreline(cfg, tmp / "coast.pkl")
        print(f"[smoke] shoreline:    {len(shore)} polyline vertices across "
              f"{shore['timestamp'].nunique()} unique dates")

        wide = reconcile(cfg, bathy, sl, wv, shore)
        print()
        print("=" * 70)
        print(f"[smoke] WIDE TABLE: {len(wide)} rows x {len(wide.columns)} cols")
        print("=" * 70)
        print(wide.head())
        print()
        print("describe():")
        print(wide.describe())
        print()

        # Validate schema
        validate_schema(wide)
        print(f"[smoke] schema OK; required columns all non-null: "
              f"{wide[PINN_REQUIRED_COLUMNS].notna().all().all()}")

        # Reload from disk to prove round-trip
        csv_path = next(cfg.output_dir.glob("*.csv"))
        reloaded = read_csv(csv_path)
        reloaded["timestamp"] = pd.to_datetime(reloaded["timestamp"], utc=True)
        assert (reloaded[PINN_COLUMNS].notna().sum().sort_index()
                == wide[PINN_COLUMNS].notna().sum().sort_index()).all(), \
            "round-trip mismatched"
        print(f"[smoke] disk round-trip OK ({csv_path})")
        print()
        print(f"[smoke] SUCCESS - produced {len(wide)} PINN-ready rows for {cfg.region.name}")