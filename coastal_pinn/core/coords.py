"""Coordinate conversions: lon/lat <-> UTM, cross-shore projection."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pyproj


def make_transformer(utm_zone: str) -> tuple[pyproj.Transformer, pyproj.Transformer]:
    """Build a (lonlat_to_utm, utm_to_lonlat) pair for the given UTM zone string.

    Uses EPSG codes derived from the zone; e.g. '31N' -> EPSG:32631.
    """
    import re
    m = re.fullmatch(r"\s*(\d+)\s*([NnSs])\s*", utm_zone)
    if not m:
        raise ValueError(f"invalid UTM zone: {utm_zone!r}")
    num = int(m.group(1))
    hemi = m.group(2).upper()
    base = 32600 if hemi == "N" else 32700
    epsg = base + num
    fwd = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return fwd, inv


def lonlat_to_utm(lon: Iterable[float], lat: Iterable[float], utm_zone: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized lon/lat (WGS84) -> UTM easting/northing (m)."""
    fwd, _ = make_transformer(utm_zone)
    x, y = fwd.transform(list(lon), list(lat))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def utm_to_lonlat(easting: Iterable[float], northing: Iterable[float], utm_zone: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized UTM easting/northing -> lon/lat (WGS84)."""
    _, inv = make_transformer(utm_zone)
    lon, lat = inv.transform(list(easting), list(northing))
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def depth_at_shore(bathy_df: pd.DataFrame, *, depth_column: str = "depth_m",
                   zone_column: str = "zone") -> float:
    """Return the mean depth of intertidal + shallow-sea cells (negative = below MSL).

    Used as the PINN's `depth_at_shore_m` input. Returns 0.0 if no qualifying cells.
    """
    if zone_column not in bathy_df.columns or depth_column not in bathy_df.columns:
        return 0.0
    mask = bathy_df[zone_column].isin(["intertidal", "sea"])
    if not mask.any():
        return 0.0
    return float(bathy_df.loc[mask, depth_column].mean())