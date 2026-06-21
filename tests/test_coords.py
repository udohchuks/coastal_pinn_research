"""Tests for coastal_pinn.core.coords."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coastal_pinn.core.coords import (
    depth_at_shore,
    lonlat_to_utm,
    make_transformer,
    utm_to_lonlat,
)


def test_utm_roundtrip_keta():
    """lon/lat -> UTM -> lon/lat should round-trip within 1e-6 deg."""
    lon = np.array([0.85, 1.05, 0.95])
    lat = np.array([5.93, 5.95, 5.97])
    x, y = lonlat_to_utm(lon, lat, "31N")
    assert np.all(x > 0) and np.all(y > 0)
    lon2, lat2 = utm_to_lonlat(x, y, "31N")
    np.testing.assert_allclose(lon2, lon, atol=1e-6)
    np.testing.assert_allclose(lat2, lat, atol=1e-6)


def test_utm_zone_keta_is_epsg_32631():
    fwd, inv = make_transformer("31N")
    # EPSG:32631 = WGS84 / UTM zone 31N
    s = str(fwd.source_crs) + str(fwd.target_crs)
    assert "32631" in s


def test_utm_zone_invalid_raises():
    with pytest.raises(ValueError, match="invalid UTM zone"):
        lonlat_to_utm([0.0], [0.0], "NOT_A_ZONE")


def test_depth_at_shore_intertidal_and_sea():
    df = pd.DataFrame({
        "depth_m": [-100.0, -50.0, 2.0, 10.0, -5.0],
        "zone":    ["sea",   "sea", "intertidal", "land", "sea"],
    })
    expected = float(np.mean([-100.0, -50.0, 2.0, -5.0]))  # -38.25
    got = depth_at_shore(df)
    assert abs(got - expected) < 1e-9


def test_depth_at_shore_no_qualifying_cells():
    df = pd.DataFrame({"depth_m": [10.0, 20.0], "zone": ["land", "land"]})
    assert depth_at_shore(df) == 0.0


def test_depth_at_shore_missing_columns():
    df = pd.DataFrame({"depth_m": [-5.0]})  # no 'zone' column
    assert depth_at_shore(df) == 0.0