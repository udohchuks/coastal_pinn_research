"""Configuration model: Region, PipelineConfig, load_config, init_config_yaml.

A Region defines the geographic extent and UTM zone for one coastline.
A PipelineConfig binds a Region to a time window and a set of enabled sources.

Configuration is loaded from a YAML file and optionally overridden via CLI
arguments. See README.md for the YAML schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from coastal_pinn.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """A named coastal region (standard One-Line Model geometry).

    bbox: (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees.
    utm_zone: e.g. '31N', '30N'. Used by shoreline/coord conversions.
    baseline: inland polyline of (lon, lat) points, drawn parallel to the
        general coastline trend. USED AS the reference for transect casting
        (DSAS convention). The PINN measures cross-shore distance S from
        this baseline seaward.
    transect_spacing_m: along-shore spacing between transects, in meters
        (DSAS default 100 m).
    transect_length_m: cross-shore length of each perpendicular transect,
        in meters. Must be long enough to span from the inland baseline
        seaward past the observed shoreline.
    """
    name: str
    bbox: tuple[float, float, float, float]
    utm_zone: str
    baseline: tuple[tuple[float, float], ...] | None = None
    transect_spacing_m: float = 100.0
    transect_length_m: float = 500.0
    # Cardinal direction of the open ocean relative to the coast
    # ('north'|'south'|'east'|'west'). Used to orient transects seaward.
    # Robust for diagonal coastlines where the "toward bbox-center"
    # heuristic fails. If None, the legacy bbox-center heuristic is used.
    ocean_side: str | None = None

    def lon_min(self) -> float: return self.bbox[0]
    def lat_min(self) -> float: return self.bbox[1]
    def lon_max(self) -> float: return self.bbox[2]
    def lat_max(self) -> float: return self.bbox[3]

    @property
    def epsg(self) -> int:
        """Convert UTM zone string to EPSG code (e.g. '31N' -> 32631)."""
        return _utm_zone_to_epsg(self.utm_zone)


def _utm_zone_to_epsg(zone: str) -> int:
    """Convert UTM zone like '31N' to EPSG code (32631 for northern hemisphere).

    EPSG = 32600 + zone_number for northern hemisphere,
           32700 + zone_number for southern hemisphere.
    """
    import re
    m = re.fullmatch(r"\s*(\d+)\s*([NnSs])\s*", zone)
    if not m:
        raise ConfigError(f"invalid UTM zone string: {zone!r} (expected e.g. '31N')")
    num = int(m.group(1))
    hemi = m.group(2).upper()
    base = 32600 if hemi == "N" else 32700
    return base + num


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """One region's worth of pipeline configuration.

    data_dir / output_dir default to <cache_root>/<region_name>/{data,download}.
    """
    region: Region
    t_start: str
    t_end: str
    cache_root: Path = field(default_factory=lambda: Path("coastal_research_ashesi"))

    # per-source enable flags and parameters
    bathymetry_enabled: bool = True
    bathymetry_product: str = "gebco_2026"

    sea_level_enabled: bool = True
    sea_level_product: str = "copernicus_phy_anfc"

    wave_intensity_enabled: bool = True
    wave_intensity_product: str = "copernicus_wam"

    shoreline_enabled: bool = True
    shoreline_gee_project: str = "igem2026"
    shoreline_sitename: str = "Keta"
    # When False (default), CoastSat skips writing per-image QA JPEGs and
    # per-detection figures. These are visual-QA only and do not affect the
    # extracted shoreline data, but they dominate the per-image processing
    # time. Set True when you need to eyeball the detections.
    shoreline_save_qc: bool = False
    # Server-side cloud pre-filter: drop scenes whose scene-level cloud cover
    # (S2 CLOUDY_PIXEL_PERCENTAGE) exceeds this percentage BEFORE download.
    # CoastSat's built-in default is a very loose 95 (only near-total cloud).
    # Lower = fewer downloads/faster, but the percentage is over the whole S2
    # tile, so too aggressive a value can drop scenes that are cloudy tile-wide
    # yet clear over this small AOI. 80-90 is a safe range for cloudy coasts
    # like the Gulf of Guinea.
    shoreline_cloud_cover_max: int = 85
    # Number of tiles downloaded + extracted in parallel. Each in-flight tile
    # uses a few GB of disk (deleted right after extraction). GEE throttles
    # beyond ~10 concurrent requests, so 6-10 is the practical sweet spot.
    shoreline_download_workers: int = 8

    # reconciliation
    join_tolerance: str = "36h"   # pandas Timedelta-compatible string

    @property
    def data_dir(self) -> Path:
        return self.cache_root / self.region.name / "data"

    @property
    def output_dir(self) -> Path:
        return self.cache_root / self.region.name / "download"

    @property
    def t_start_dt(self) -> "pd.Timestamp":  # type: ignore[name-defined]
        import pandas as pd
        return pd.Timestamp(self.t_start, tz="UTC")

    @property
    def t_end_dt(self) -> "pd.Timestamp":  # type: ignore[name-defined]
        import pandas as pd
        return pd.Timestamp(self.t_end, tz="UTC")


# ---------------------------------------------------------------------------
# Region registry
# ---------------------------------------------------------------------------

# Default regions. Users can extend by importing and adding to REGIONS
# from their own config, or by editing this dict directly.

def _load_keta_baseline() -> tuple[tuple[float, float], ...] | None:
    """Load the Keta onshore baseline polyline from the bundled data file.

    Single source of truth shared with config/keta.yaml. Regenerate both via
    scripts/derive_keta_baseline.py. Returns None if the file is missing.
    """
    import json
    p = Path(__file__).resolve().parent.parent / "data" / "keta_baseline.json"
    if not p.exists():
        return None
    pts = json.loads(p.read_text(encoding="utf-8"))
    return tuple((float(a), float(b)) for a, b in pts)


REGIONS: dict[str, Region] = {
    # Keta, eastern Ghana. Geometry follows the OSM Atlantic coastline (the
    # coast runs NE from the Volta estuary toward Aflao) — NOT the inland Keta
    # Lagoon. The baseline sits ~150 m onshore; transects point seaward (south)
    # toward the open Atlantic. See data.md and scripts/derive_keta_baseline.py.
    "keta": Region(
        name="keta",
        bbox=(0.85, 5.74, 1.24, 6.15),
        utm_zone="31N",
        baseline=_load_keta_baseline(),
        transect_spacing_m=50.0,
        transect_length_m=750.0,
        ocean_side="south",
    ),
}


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = ("region", "time")
REQUIRED_REGION_KEYS = ("name", "bbox", "utm_zone")


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file into a PipelineConfig.

    Raises ConfigError on missing required keys or invalid types.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    return _raw_to_config(raw, source=str(p))


def _raw_to_config(raw: dict[str, Any], source: str = "<dict>") -> PipelineConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top-level must be a mapping, got {type(raw).__name__}")

    for k in REQUIRED_TOP_KEYS:
        if k not in raw:
            raise ConfigError(f"{source}: missing required key '{k}'")

    region_raw = raw["region"]
    if not isinstance(region_raw, dict):
        raise ConfigError(f"{source}: 'region' must be a mapping")
    for k in REQUIRED_REGION_KEYS:
        if k not in region_raw:
            raise ConfigError(f"{source}: 'region' missing required key '{k}'")

    bbox_raw = region_raw["bbox"]
    if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        raise ConfigError(f"{source}: 'region.bbox' must be a 4-element list")
    bbox = tuple(float(x) for x in bbox_raw)

    baseline = None
    if "baseline" in region_raw and region_raw["baseline"] is not None:
        bl = region_raw["baseline"]
        if not isinstance(bl, (list, tuple)) or len(bl) < 2:
            raise ConfigError(
                f"{source}: 'region.baseline' must be a list of >= 2 (lon, lat) points"
            )
        baseline = tuple((float(p[0]), float(p[1])) for p in bl)

    transect_spacing_m = float(region_raw.get("transect_spacing_m", 100.0))
    transect_length_m = float(region_raw.get("transect_length_m", 500.0))
    ocean_side = region_raw.get("ocean_side")
    ocean_side = str(ocean_side) if ocean_side is not None else None

    region = Region(
        name=str(region_raw["name"]),
        bbox=bbox,
        utm_zone=str(region_raw["utm_zone"]),
        baseline=baseline,
        transect_spacing_m=transect_spacing_m,
        transect_length_m=transect_length_m,
        ocean_side=ocean_side,
    )

    time_raw = raw["time"]
    if not isinstance(time_raw, dict) or "start" not in time_raw or "end" not in time_raw:
        raise ConfigError(f"{source}: 'time' must have 'start' and 'end'")
    t_start = str(time_raw["start"])
    t_end = str(time_raw["end"])

    cache_root = Path(raw.get("cache", {}).get("root", "coastal_research_ashesi"))

    sources = raw.get("sources") or {}

    def _src_bool(key: str, default: bool) -> bool:
        sub = sources.get(key) or {}
        return bool(sub.get("enabled", default))

    def _src_str(key: str, field_name: str, default: str) -> str:
        sub = sources.get(key) or {}
        return str(sub.get(field_name, default))

    join_tolerance = str(raw.get("reconcile", {}).get("join_tolerance", "36h"))

    shoreline_src = sources.get("shoreline") or {}
    shoreline_save_qc = bool(shoreline_src.get("save_qc", False))
    shoreline_cloud_cover_max = int(shoreline_src.get("cloud_cover_max", 85))
    shoreline_download_workers = int(shoreline_src.get("download_workers", 8))

    return PipelineConfig(
        region=region,
        t_start=t_start,
        t_end=t_end,
        cache_root=cache_root,
        bathymetry_enabled=_src_bool("bathymetry", True),
        bathymetry_product=_src_str("bathymetry", "product", "gebco_2026"),
        sea_level_enabled=_src_bool("sea_level", True),
        sea_level_product=_src_str("sea_level", "product", "copernicus_phy_anfc"),
        wave_intensity_enabled=_src_bool("wave_intensity", True),
        wave_intensity_product=_src_str("wave_intensity", "product", "copernicus_wam"),
        shoreline_enabled=_src_bool("shoreline", True),
        shoreline_gee_project=_src_str("shoreline", "gee_project", "igem2026"),
        shoreline_sitename=_src_str("shoreline", "sitename", "Keta"),
        shoreline_save_qc=shoreline_save_qc,
        shoreline_cloud_cover_max=shoreline_cloud_cover_max,
        shoreline_download_workers=shoreline_download_workers,
        join_tolerance=join_tolerance,
    )


def init_config_yaml(region: Region, out_path: str | Path) -> None:
    """Write a default YAML config for the given region to out_path."""
    body = {
        "region": {
            "name": region.name,
            "bbox": list(region.bbox),
            "utm_zone": region.utm_zone,
            "transect_spacing_m": region.transect_spacing_m,
            "transect_length_m": region.transect_length_m,
            **({"ocean_side": region.ocean_side} if region.ocean_side else {}),
        },
        "time": {
            "start": "2018-01-01",
            "end": "2018-12-31",
        },
        "cache": {"root": "coastal_research_ashesi", "strategy": "append_only"},
        "reconcile": {"join_tolerance": "36h"},
        "sources": {
            "bathymetry": {"enabled": True, "product": "gebco_2026"},
            "sea_level": {"enabled": True, "product": "copernicus_phy_anfc"},
            "wave_intensity": {"enabled": True, "product": "copernicus_wam"},
            "shoreline": {"enabled": True, "gee_project": "igem2026", "sitename": region.name.title(), "save_qc": False, "cloud_cover_max": 85, "download_workers": 8},
        },
    }
    if region.baseline is not None:
        body["region"]["baseline"] = [list(p) for p in region.baseline]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def merge_overrides(cfg: PipelineConfig, **kwargs: Any) -> PipelineConfig:
    """Return a new PipelineConfig with any non-None kwargs applied.

    Recognised override keys:
        t_start, t_end, join_tolerance, cache_root,
        bathymetry_enabled, sea_level_enabled, wave_intensity_enabled, shoreline_enabled
    """
    changes = {k: v for k, v in kwargs.items() if v is not None}
    if not changes:
        return cfg
    try:
        return replace(cfg, **changes)
    except TypeError as e:
        raise ConfigError(f"unknown config override: {list(changes)}") from e