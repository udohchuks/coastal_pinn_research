"""Tests for the CLI surface (argparse)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import coastal_pinn.cli as cli


def test_help(capsys):
    rc = cli.main(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "coastal_pinn" in out
    assert "run" in out
    assert "fetch" in out


def test_list_regions(capsys):
    rc = cli.main(["list-regions"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "keta" in out


def test_init_config(tmp_path: Path):
    out = tmp_path / "keta.yaml"
    rc = cli.main(["init-config", "--region", "keta", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "region:" in out.read_text(encoding="utf-8")


def test_init_config_unknown_region(tmp_path: Path, capsys):
    out = tmp_path / "x.yaml"
    rc = cli.main(["init-config", "--region", "atlantis", "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown region" in err


def test_run_missing_config(capsys):
    rc = cli.main(["run", "--config", "nonexistent.yaml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err or "missing" in err or "ERROR" in err