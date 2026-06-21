"""Tests for coastal_pinn.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from coastal_pinn import REGIONS, PipelineConfig, init_config_yaml, load_config
from coastal_pinn.exceptions import ConfigError


def test_keta_region_in_registry():
    assert "keta" in REGIONS
    r = REGIONS["keta"]
    assert r.name == "keta"
    assert r.utm_zone == "31N"
    assert r.epsg == 32631  # 32600 + 31 for northern hemisphere


def test_utm_zone_to_epsg_southern():
    r = REGIONS["keta"]
    assert _zone("30N") == 32630
    assert _zone("30S") == 32730


def _zone(zone: str) -> int:
    from coastal_pinn.config import Region, _utm_zone_to_epsg
    return _utm_zone_to_epsg(zone)


def test_load_keta_config(tmp_path: Path):
    out = tmp_path / "keta.yaml"
    init_config_yaml(REGIONS["keta"], out)
    cfg = load_config(out)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.region.name == "keta"
    assert cfg.t_start == "2018-01-01"
    assert cfg.t_end == "2018-12-31"
    assert cfg.bathymetry_enabled is True
    assert cfg.sea_level_enabled is True
    assert cfg.join_tolerance == "36h"


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_missing_required_key(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("time:\n  start: 2018-01-01\n  end: 2018-12-31\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="region"):
        load_config(bad)


def test_data_output_dirs(tmp_path: Path):
    out = tmp_path / "keta.yaml"
    init_config_yaml(REGIONS["keta"], out)
    cfg = load_config(out)
    cfg.cache_root = tmp_path
    assert cfg.data_dir == tmp_path / "keta" / "data"
    assert cfg.output_dir == tmp_path / "keta" / "download"


def test_timestamps_are_utc_aware(tmp_path: Path):
    out = tmp_path / "keta.yaml"
    init_config_yaml(REGIONS["keta"], out)
    cfg = load_config(out)
    ts = cfg.t_start_dt
    assert ts.tzinfo is not None
    assert str(ts.tzinfo) == "UTC"