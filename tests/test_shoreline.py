"""Tests for coastal_pinn.sources.shoreline (post-fetch transformation only).

Tests _to_dataframe against a real-shape CoastSat output pickle.
"""

from __future__ import annotations

import pickle

import pytest

from coastal_pinn import PipelineConfig
from coastal_pinn.core.io import read_pickle
from coastal_pinn.exceptions import SourceUnavailable
from coastal_pinn.sources.shoreline import _to_dataframe


def test_shoreline_to_dataframe_columns(keta_config, real_shape_shoreline_pkl):
    d = read_pickle(real_shape_shoreline_pkl)
    df = _to_dataframe(d, keta_config)
    assert set(df.columns) >= {"region", "timestamp", "sat", "pt_idx",
                               "easting_m", "northing_m"}
    assert (df["region"] == "keta").all()


def test_shoreline_timestamps_are_utc(keta_config, real_shape_shoreline_pkl):
    d = read_pickle(real_shape_shoreline_pkl)
    df = _to_dataframe(d, keta_config)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_shoreline_coords_in_utm_zone_31n(keta_config, real_shape_shoreline_pkl):
    """Easting/Northing should be in UTM 31N (Ghana)."""
    d = read_pickle(real_shape_shoreline_pkl)
    df = _to_dataframe(d, keta_config)
    # Keta is around lon=1.0, lat=5.95. UTM 31N easting is ~260k-265k, northing ~656k-657k
    assert df["easting_m"].between(200_000, 350_000).all()
    assert df["northing_m"].between(600_000, 700_000).all()


def test_shoreline_pt_idx_monotonic(keta_config, real_shape_shoreline_pkl):
    d = read_pickle(real_shape_shoreline_pkl)
    df = _to_dataframe(d, keta_config)
    # pt_idx within each timestamp should be 0..N-1
    for ts, sub in df.groupby("timestamp"):
        assert list(sub["pt_idx"]) == list(range(len(sub)))


def test_shoreline_empty_raises(keta_config):
    with pytest.raises(SourceUnavailable, match="no polylines"):
        _to_dataframe({"dates": [], "shorelines": []}, keta_config)