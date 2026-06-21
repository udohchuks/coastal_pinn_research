"""Run the offline smoke test against the *real* cache root so subsequent
CLI commands can reuse the cache via --offline.

Usage:
    python tests/run_offline_smoke_real_cache.py
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coastal_pinn import PipelineConfig, REGIONS
from coastal_pinn.pipeline import reconcile

# These helper functions from smoke_offline are also reused.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_offline  # noqa: E402


def _build_fixtures_in_tmp(cfg: PipelineConfig) -> tuple:
    """Build the four source DataFrames in a fresh tmp dir (no Windows
    file-lock issues), then return them. We do NOT touch the real cache
    root here.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        b = smoke_offline._make_bathy(cfg, td / "bathy.nc")
        gc.collect()
        s = smoke_offline._make_sea_level(cfg, td / "sl.nc")
        gc.collect()
        w = smoke_offline._make_waves(cfg, td / "wv.nc")
        gc.collect()
        sh = smoke_offline._make_shoreline(cfg, td / "shore.pkl")
        gc.collect()
        return b, s, w, sh


if __name__ == "__main__":
    cache_root = ROOT / "coastal_research_ashesi"
    if cache_root.exists():
        shutil.rmtree(cache_root)

    cfg = PipelineConfig(
        region=REGIONS["keta"],
        t_start="2018-01-01",
        t_end="2018-12-31",
        cache_root=cache_root,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] building real-shape fixtures (no network)...")
    bathy, sl, wv, shore = _build_fixtures_in_tmp(cfg)
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
    print(f"[smoke] CSV written: {next(cfg.output_dir.glob('*.csv'))}")
    print(f"[smoke] PKL written: {next(cfg.output_dir.glob('*.pkl'))}")