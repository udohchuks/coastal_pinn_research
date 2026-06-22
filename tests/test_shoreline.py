"""Tests for coastal_pinn.sources.shoreline (post-fetch transformation only).

Tests _intersect_with_transects against a real-shape CoastSat output pickle.
"""

from __future__ import annotations

import pickle

import pytest

from coastal_pinn import PipelineConfig
from coastal_pinn.core.io import read_pickle
from coastal_pinn.exceptions import SourceUnavailable
from coastal_pinn.sources.shoreline import _intersect_with_transects


def test_shoreline_intersect_columns(keta_config, real_shape_shoreline_pkl):
    d = read_pickle(real_shape_shoreline_pkl)
    df = _intersect_with_transects(d, keta_config)
    # New per-transect schema
    assert set(df.columns) >= {"region", "timestamp", "transect_id",
                               "along_shore_x_m", "cross_shore_S_m", "sat"}
    assert (df["region"] == "keta").all()


def test_shoreline_timestamps_are_utc(keta_config, real_shape_shoreline_pkl):
    d = read_pickle(real_shape_shoreline_pkl)
    df = _intersect_with_transects(d, keta_config)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_shoreline_cross_shore_in_range(keta_config, real_shape_shoreline_pkl):
    """S values should be in [0, transect_length_m] (i.e. between baseline and seaward end)."""
    d = read_pickle(real_shape_shoreline_pkl)
    df = _intersect_with_transects(d, keta_config)
    if not df.empty:
        assert (df["cross_shore_S_m"] >= 0).all()
        # max should be <= transect_length_m
        assert (df["cross_shore_S_m"] <= keta_config.region.transect_length_m).all()


def test_shoreline_has_per_transect_rows(keta_config, real_shape_shoreline_pkl):
    """Each cloud-free date should have a row per intersected transect."""
    d = read_pickle(real_shape_shoreline_pkl)
    df = _intersect_with_transects(d, keta_config)
    n_transects_with_data = df["transect_id"].nunique()
    # The fixture polylines are localized near the bbox center;
    # at least one transect should have intersected.
    assert n_transects_with_data >= 1


def test_shoreline_empty_raises(keta_config):
    with pytest.raises(SourceUnavailable, match="no polylines"):
        _intersect_with_transects({"dates": [], "shorelines": []}, keta_config)
