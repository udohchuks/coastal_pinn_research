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
    """A named coastal region.

    bbox: (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees.
    utm_zone: e.g. '31N', '30N'. Used by shoreline/coord conversions.
    transect: optional (lon, lat) endpoints defining the cross-shore
        reference transect. PINN input uses the UTM coords directly, but
        this is exposed for cross-shore distance computation if needed.
    """
    name: str
    bbox: tuple[float, float, float, float]
    utm_zone: str
    transect: tuple[tuple[float, float], tuple[float, float]] | None = None

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
    wave_intensity_product: str = "noaa_ww3"

    shoreline_enabled: bool = True
    shoreline_gee_project: str = "igem2026"
    shoreline_sitename: str = "Keta"

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

REGIONS: dict[str, Region] = {
    "keta": Region(
        name="keta",
        bbox=(0.80, 5.85, 1.40, 6.10),
        utm_zone="31N",
        transect=((1.40, 5.95), (0.80, 5.95)),
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

    transect = None
    if "transect" in region_raw and region_raw["transect"] is not None:
        tr = region_raw["transect"]
        if not isinstance(tr, (list, tuple)) or len(tr) != 2:
            raise ConfigError(f"{source}: 'region.transect' must be 2 points")
        transect = (
            (float(tr[0][0]), float(tr[0][1])),
            (float(tr[1][0]), float(tr[1][1])),
        )

    region = Region(
        name=str(region_raw["name"]),
        bbox=bbox,
        utm_zone=str(region_raw["utm_zone"]),
        transect=transect,
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
        wave_intensity_product=_src_str("wave_intensity", "product", "noaa_ww3"),
        shoreline_enabled=_src_bool("shoreline", True),
        shoreline_gee_project=_src_str("shoreline", "gee_project", "igem2026"),
        shoreline_sitename=_src_str("shoreline", "sitename", "Keta"),
        join_tolerance=join_tolerance,
    )


def init_config_yaml(region: Region, out_path: str | Path) -> None:
    """Write a default YAML config for the given region to out_path."""
    body = {
        "region": {
            "name": region.name,
            "bbox": list(region.bbox),
            "utm_zone": region.utm_zone,
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
            "wave_intensity": {"enabled": True, "product": "noaa_ww3"},
            "shoreline": {"enabled": True, "gee_project": "igem2026", "sitename": region.name.title()},
        },
    }
    if region.transect is not None:
        body["region"]["transect"] = [list(region.transect[0]), list(region.transect[1])]
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