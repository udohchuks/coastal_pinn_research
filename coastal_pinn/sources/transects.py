"""Transect generation for the standard One-Line Model.

Given a Region with an inland baseline, this module generates N perpendicular
transects (DSAS convention, 100 m along-shore spacing) that will be used to
intersect the CoastSat shoreline polylines.

Each transect is a line segment:
    - origin: a point on the baseline
    - end: a point transect_length_m seaward of the origin
    - along_shore_x_m: distance from the first baseline endpoint to the origin
        (this becomes the spatial coordinate x in the PINN)
    - shore_normal_deg: direction the transect points (degrees CCW from East),
        i.e. the direction toward the sea at this transect

For a 2-point straight baseline (the common case), the shore_normal is
constant along the entire baseline. For an N-point baseline (curved coast),
the shore_normal varies per transect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from coastal_pinn.config import Region
from coastal_pinn.core.coords import lonlat_to_utm, make_transformer


@dataclass(frozen=True)
class Transect:
    """A single perpendicular transect."""
    transect_id: int
    along_shore_x_m: float      # distance from baseline[0] along the baseline
    origin_x: float             # UTM easting of transect origin
    origin_y: float             # UTM northing of transect origin
    end_x: float                # UTM easting of seaward end
    end_y: float                # UTM northing of seaward end
    shore_normal_deg: float     # direction the transect points (deg CCW from East)


def _bbox_center_lat(region: Region) -> float:
    """Latitude of the bbox center. Used to determine seaward side."""
    return (region.bbox[1] + region.bbox[3]) / 2.0


def _orient_to_ocean(perp: np.ndarray, ocean_side: str) -> np.ndarray:
    """Flip a perpendicular (dx_east, dy_north) so it points toward the ocean.

    ocean_side is a cardinal direction: the side of the coast on which the
    open ocean lies. This is robust for diagonal coastlines, where the
    "perpendicular toward bbox-center latitude" heuristic can point inland.
    """
    side = ocean_side.lower()
    dx, dy = float(perp[0]), float(perp[1])
    if side == "south":
        want = dy < 0          # negative northing
    elif side == "north":
        want = dy > 0
    elif side == "east":
        want = dx > 0
    elif side == "west":
        want = dx < 0
    else:
        raise ValueError(
            f"invalid ocean_side {ocean_side!r}; expected north/south/east/west"
        )
    return perp if want else -perp


def _shore_normal_for_baseline(
    region: Region, baseline_utm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (perpendicular_dx, perpendicular_dy) for a 2-point baseline.

    The perpendicular is the direction from the baseline point toward the
    bbox center latitude. For a straight baseline (2 points), this is a
    single (dx, dy) pair.
    """
    p0 = baseline_utm[0]
    p1 = baseline_utm[1]
    tangent = p1 - p0
    t_norm = np.linalg.norm(tangent)
    if t_norm == 0:
        raise ValueError("baseline has zero length")
    tangent = tangent / t_norm
    # Two perpendiculars: rotate +90 and -90
    perp_a = np.array([-tangent[1], tangent[0]])
    perp_b = np.array([tangent[1], -tangent[0]])
    # If the region declares which side the ocean is on, use it (robust).
    if region.ocean_side is not None:
        oriented = _orient_to_ocean(perp_a, region.ocean_side)
        return oriented[0], oriented[1]
    # Otherwise pick the one pointing toward the bbox center latitude.
    # Convert center lat/lon to UTM for consistent units
    center_lon = (region.bbox[0] + region.bbox[2]) / 2.0
    center_lat = _bbox_center_lat(region)
    fwd, _ = make_transformer(region.utm_zone)
    center_x, center_y = fwd.transform(center_lon, center_lat)
    # Compare perpendicular's northing component with the direction
    # from the baseline midpoint toward the bbox center
    mid_y = (p0[1] + p1[1]) / 2.0
    if perp_a[1] * (center_y - mid_y) > 0:
        return perp_a[0], perp_a[1]
    else:
        return perp_b[0], perp_b[1]


@lru_cache(maxsize=8)
def generate_transects(region: Region) -> pd.DataFrame:
    """Generate N perpendicular transects from the inland baseline.

    Returns a DataFrame with one row per transect:
        transect_id, along_shore_x_m,
        origin_x, origin_y, end_x, end_y, shore_normal_deg

    Raises ValueError if region.baseline is missing.
    """
    if region.baseline is None or len(region.baseline) < 2:
        raise ValueError(
            f"region {region.name!r} has no baseline; "
            "set region.baseline to a list of (lon, lat) points"
        )

    # Convert baseline to UTM
    bl_lons = [p[0] for p in region.baseline]
    bl_lats = [p[1] for p in region.baseline]
    bl_x, bl_y = lonlat_to_utm(bl_lons, bl_lats, region.utm_zone)
    baseline_utm = np.column_stack([bl_x, bl_y])

    # For a 2-point baseline, the perpendicular is constant
    if len(region.baseline) == 2:
        perp_dx, perp_dy = _shore_normal_for_baseline(region, baseline_utm)
        tangent = baseline_utm[1] - baseline_utm[0]
        t_len = float(np.linalg.norm(tangent))
        n_transects = max(2, int(math.floor(t_len / region.transect_spacing_m)) + 1)
        # Sample N points along the baseline at spacing
        s_vals = np.linspace(0.0, t_len, n_transects)
        origins = baseline_utm[0] + np.outer(s_vals, tangent / t_len)
        along_shore_x = s_vals
    else:
        # N-point baseline: arc-length parameterize, sample at spacing,
        # compute local tangent and perpendicular at each sample.
        # This is the more general case (curved coast).
        deltas = np.diff(baseline_utm, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_len = float(cumlen[-1])
        n_transects = max(2, int(math.floor(total_len / region.transect_spacing_m)) + 1)
        s_vals = np.linspace(0.0, total_len, n_transects)
        origins = np.zeros((n_transects, 2))
        per_shore_normal = np.zeros(n_transects)
        center_lat = _bbox_center_lat(region)
        for i, s in enumerate(s_vals):
            # find segment containing s
            seg_idx = int(np.searchsorted(cumlen, s, side="right") - 1)
            seg_idx = min(seg_idx, len(deltas) - 1)
            local_t = deltas[seg_idx] / seg_lengths[seg_idx]
            # Position at arc-length s: start of segment + offset within segment
            offset_in_seg = s - cumlen[seg_idx]
            origins[i] = baseline_utm[seg_idx] + offset_in_seg * local_t
            # Perpendicular at this point
            local_perp = np.array([-local_t[1], local_t[0]])
            if region.ocean_side is not None:
                # Explicit ocean side: robust for diagonal coasts.
                local_perp = _orient_to_ocean(local_perp, region.ocean_side)
            else:
                # Legacy heuristic: orient toward bbox-center latitude.
                mid_lat = (baseline_utm[seg_idx, 1] + baseline_utm[seg_idx + 1, 1]) / 2.0
                if (local_perp[1] > 0 and mid_lat > center_lat) or \
                   (local_perp[1] < 0 and mid_lat < center_lat):
                    pass  # already points toward center (seaward)
                else:
                    local_perp = -local_perp
            per_shore_normal[i] = math.degrees(math.atan2(local_perp[1], local_perp[0])) % 360.0
        along_shore_x = s_vals
        # For N-point baselines, transect ends use per-transect shore normals
        # (handled below via per_shore_normal array)
        # For the 2-point case, perp_dx/perp_dy is used; for N-point, we
        # build ends per-transect.
        ends = np.zeros((n_transects, 2))
        for i in range(n_transects):
            sn_rad = math.radians(per_shore_normal[i])
            ends[i] = origins[i] + region.transect_length_m * np.array(
                [math.cos(sn_rad), math.sin(sn_rad)]
            )
        # Build DataFrame with per-transect shore normals
        df = pd.DataFrame({
            "transect_id": np.arange(n_transects, dtype=np.int64),
            "along_shore_x_m": along_shore_x.astype(float),
            "origin_x": origins[:, 0].astype(float),
            "origin_y": origins[:, 1].astype(float),
            "end_x": ends[:, 0].astype(float),
            "end_y": ends[:, 1].astype(float),
            "shore_normal_deg": per_shore_normal,
        })
        return df

    # Compute seaward end of each transect
    ends = origins + region.transect_length_m * np.array([perp_dx, perp_dy])

    # Shore-normal direction in degrees CCW from East
    shore_normal_rad = math.atan2(perp_dy, perp_dx)
    shore_normal_deg = (math.degrees(shore_normal_rad)) % 360.0

    df = pd.DataFrame({
        "transect_id": np.arange(n_transects, dtype=np.int64),
        "along_shore_x_m": along_shore_x.astype(float),
        "origin_x": origins[:, 0].astype(float),
        "origin_y": origins[:, 1].astype(float),
        "end_x": ends[:, 0].astype(float),
        "end_y": ends[:, 1].astype(float),
        "shore_normal_deg": np.full(n_transects, shore_normal_deg, dtype=float),
    })
    return df


def transect_intersection_distance(
    polyline_xy: np.ndarray,
    origin_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> float | None:
    """Compute the distance from the transect origin to where the polyline
    crosses the transect line, or None if no intersection.

    polyline_xy: (N, 2) array of (easting, northing) along the CoastSat
                 shoreline polyline.
    origin_xy, end_xy: the transect endpoints in UTM.

    Returns the distance from origin to the intersection point along the
    transect direction, in meters. None if no intersection.
    """
    from shapely.geometry import LineString, Point

    if polyline_xy.shape[0] < 2:
        return None

    try:
        shore_line = LineString(polyline_xy)
        transect_line = LineString([origin_xy, end_xy])
        if not shore_line.intersects(transect_line):
            return None
        inter = shore_line.intersection(transect_line)
        if inter.is_empty:
            return None
        if inter.geom_type == "Point":
            pt = inter
        elif inter.geom_type == "MultiPoint":
            # multiple intersections: take the one closest to origin
            distances = [Point(p).distance(Point(origin_xy)) for p in inter.geoms]
            pt = Point(list(inter.geoms)[int(np.argmin(distances))])
        elif inter.geom_type == "GeometryCollection":
            pts = [g for g in inter.geoms if g.geom_type == "Point"]
            if not pts:
                return None
            distances = [p.distance(Point(origin_xy)) for p in pts]
            pt = pts[int(np.argmin(distances))]
        else:
            return None
        # Distance from origin to intersection along the transect direction
        dx = pt.x - origin_xy[0]
        dy = pt.y - origin_xy[1]
        # Project onto transect direction
        tdx = end_xy[0] - origin_xy[0]
        tdy = end_xy[1] - origin_xy[1]
        tlen = math.hypot(tdx, tdy)
        if tlen == 0:
            return None
        # Signed distance along the transect
        return (dx * tdx + dy * tdy) / tlen
    except Exception:
        return None
