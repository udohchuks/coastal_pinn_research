"""Append-only artifact directory helpers.

The cache layout under <cache_root>/<region>/ is fixed:

    <cache_root>/<region>/data/<source>/<key>.nc|pkl
    <cache_root>/<region>/download/<source>/<key>.csv

A cache key encodes source + time window so the same fetch never collides
with another fetch of a different window. The directory is append-only:
files are never overwritten. To refresh, delete the file by hand (or use
the CLI's clean-cache command).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from coastal_pinn.config import PipelineConfig


def _bbox_tag(bbox: tuple[float, float, float, float]) -> str:
    """Short stable hash of a bbox, used to scope cache files to a domain."""
    s = ",".join(f"{float(v):.4f}" for v in bbox)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def cache_key(*, source: str, t_start: str, t_end: str, suffix: str,
              bbox: tuple[float, float, float, float] | None = None) -> str:
    """Build a stable, filesystem-safe cache key.

    When ``bbox`` is given, a short hash of it is folded into the key so that
    changing the study domain (without changing the time window) does NOT
    silently reuse a cache file built for the old area.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", source)
    tag = f"_{_bbox_tag(bbox)}" if bbox is not None else ""
    return f"{safe}_{t_start}_to_{t_end}{tag}.{suffix}"


def data_path(cfg: PipelineConfig, source: str, *, t_start: str | None = None,
              t_end: str | None = None, suffix: str = "nc") -> Path:
    """Return the append-only data path for a (source, time-window) pair.

    Defaults to the PipelineConfig's t_start / t_end.
    """
    ts = t_start if t_start is not None else cfg.t_start
    te = t_end if t_end is not None else cfg.t_end
    d = cfg.data_dir / source
    d.mkdir(parents=True, exist_ok=True)
    return d / cache_key(source=source, t_start=ts, t_end=te, suffix=suffix,
                         bbox=cfg.region.bbox)


def download_path(cfg: PipelineConfig, source: str, *, t_start: str | None = None,
                  t_end: str | None = None, suffix: str = "csv") -> Path:
    ts = t_start if t_start is not None else cfg.t_start
    te = t_end if t_end is not None else cfg.t_end
    d = cfg.output_dir / source
    d.mkdir(parents=True, exist_ok=True)
    return d / cache_key(source=source, t_start=ts, t_end=te, suffix=suffix,
                         bbox=cfg.region.bbox)


def pinn_wide_path(cfg: PipelineConfig, *, t_start: str | None = None,
                   t_end: str | None = None, suffix: str = "csv") -> Path:
    """Path of the final wide-format table for one region."""
    ts = t_start if t_start is not None else cfg.t_start
    te = t_end if t_end is not None else cfg.t_end
    key = cache_key(source="pinn_wide", t_start=ts, t_end=te, suffix=suffix,
                    bbox=cfg.region.bbox)
    return cfg.output_dir / key