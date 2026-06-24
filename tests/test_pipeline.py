"""Tests for coastal_pinn.pipeline.reconcile.

Drives the reconciliation logic with real-shape fixtures (no synthetic
data). Uses a tmp cache root so no real network or GEE auth is touched.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from coastal_pinn import PipelineConfig
from coastal_pinn.pipeline import reconcile
from coastal_pinn.sources.bathymetry import _extract_per_transect
from coastal_pinn.sources.sea_level import _to_dataframe as sl_to_df
from coastal_pinn.sources.shoreline import _intersect_with_transects
from coastal_pinn.sources.wave_intensity import _to_dataframe as wv_to_df
from coastal_pinn.exceptions import SourceUnavailable
import xarray as xr


def _make_bathy_df(keta_config, real_shape_bathy_nc):
    ds = xr.open_dataset(real_shape_bathy_nc)
    return _extract_per_transect(ds, keta_config)


def _make_sea_level_df(keta_config, real_shape_sea_level_nc):
    ds = xr.open_dataset(real_shape_sea_level_nc)
    return sl_to_df(ds, keta_config)


def _make_waves_df(keta_config, real_shape_waves_nc):
    ds = xr.open_dataset(real_shape_waves_nc)
    return wv_to_df(ds, keta_config)


def _make_shoreline_df(keta_config, real_shape_shoreline_pkl):
    with open(real_shape_shoreline_pkl, "rb") as f:
        d = pickle.load(f)
    return _intersect_with_transects(d, keta_config)


def test_reconcile_full_pipeline(keta_config, real_shape_bathy_nc,
                                 real_shape_sea_level_nc,
                                 real_shape_waves_nc,
                                 real_shape_shoreline_pkl,
                                 tmp_path):
    # Redirect cache_root to tmp so the wide-table CSV is written there
    keta_config.cache_root = tmp_path / "coastal_research_ashesi"

    bathy = _make_bathy_df(keta_config, real_shape_bathy_nc)
    sl    = _make_sea_level_df(keta_config, real_shape_sea_level_nc)
    wv    = _make_waves_df(keta_config, real_shape_waves_nc)
    shore = _make_shoreline_df(keta_config, real_shape_shoreline_pkl)

    wide = reconcile(keta_config, bathy, sl, wv, shore)

    from coastal_pinn.core.schema import PINN_COLUMNS, PINN_REQUIRED_COLUMNS
    assert list(wide.columns) == PINN_COLUMNS
    assert wide["timestamp"].dt.tz is not None
    assert str(wide["timestamp"].dt.tz) == "UTC"
    assert (wide["region"] == "keta").all()
    # Per-(transect, date) rows
    assert "transect_id" in wide.columns
    assert "along_shore_x_m" in wide.columns
    assert "cross_shore_S_m" in wide.columns
    # no NaN in required columns (all columns are required)
    assert wide[PINN_REQUIRED_COLUMNS].notna().all().all()
    # E_wave is the derived Yates wave energy, E = W_m**2 / 16
    assert np.allclose(wide["E_wave"], wide["W_m"] ** 2 / 16.0)
    # The csv + pkl files were written
    out_dir = keta_config.output_dir
    csvs = list(out_dir.glob("*.csv"))
    pkls = list(out_dir.glob("*.pkl"))
    assert csvs, "wide CSV was not written"
    assert pkls, "wide pkl was not written"


def test_reconcile_empty_shoreline_raises(keta_config, real_shape_bathy_nc,
                                           real_shape_sea_level_nc,
                                           real_shape_waves_nc,
                                           tmp_path):
    keta_config.cache_root = tmp_path / "coastal_research_ashesi"

    bathy = _make_bathy_df(keta_config, real_shape_bathy_nc)
    sl    = _make_sea_level_df(keta_config, real_shape_sea_level_nc)
    wv    = _make_waves_df(keta_config, real_shape_waves_nc)
    # Empty shoreline in the new per-transect schema
    shore = pd.DataFrame(columns=["region", "timestamp", "transect_id",
                                  "along_shore_x_m", "cross_shore_S_m", "sat"])

    with pytest.raises(SourceUnavailable):
        reconcile(keta_config, bathy, sl, wv, shore)


def test_reconcile_drops_rows_missing_inputs(keta_config, real_shape_bathy_nc,
                                              real_shape_sea_level_nc,
                                              real_shape_waves_nc,
                                              real_shape_shoreline_pkl,
                                              tmp_path):
    """If a shoreline observation falls outside the asof tolerance of any
    sea_level/wave row, that row should be dropped (not fabricated)."""
    keta_config.cache_root = tmp_path / "coastal_research_ashesi"
    # tighten tolerance to force drops
    keta_config.join_tolerance = "1s"

    bathy = _make_bathy_df(keta_config, real_shape_bathy_nc)
    sl    = _make_sea_level_df(keta_config, real_shape_sea_level_nc)
    wv    = _make_waves_df(keta_config, real_shape_waves_nc)
    shore = _make_shoreline_df(keta_config, real_shape_shoreline_pkl)

    # With 1s tolerance, very few if any rows will match — must not crash
    try:
        wide = reconcile(keta_config, bathy, sl, wv, shore)
        # rows that did match must still be complete
        from coastal_pinn.core.schema import PINN_REQUIRED_COLUMNS
        assert wide[PINN_REQUIRED_COLUMNS].notna().all().all()
    except SourceUnavailable:
        # dropping everything is acceptable; the contract is "no fabrication"
        pass
