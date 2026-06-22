"""Shoreline source: Google Earth Engine + CoastSat with N-transect intersection.

Fetches cloud-free Sentinel-2 / Landsat-8 imagery via Earth Engine, runs
the CoastSat shoreline classifier, intersects the per-date polylines with
N perpendicular transects (DSAS convention, generated from the inland
baseline), and returns the per-(transect, date) cross-shore distances.

Outputs DataFrame with columns:
    region              str
    timestamp           pd.Timestamp, UTC
    transect_id         int, 0..N-1
    along_shore_x_m     float, transect's along-shore position (m)
    cross_shore_S_m     float, observed shoreline cross-shore distance (m)
    sat                 str, 'S2' | 'L8' | 'UNK'

Caches the dict (shorelines + dates) to a pickle under
cfg.data_dir/shoreline/. Append-only.

The GEE project ID (cfg.shoreline_gee_project) and sitename
(cfg.shoreline_sitename) come from the PipelineConfig.

NB: GEE requires `earthengine authenticate` to have been run once per
machine. The pickle of the CoastSat output is what we cache -- if GEE
auth fails on a fresh machine, the user must re-run the auth.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coastal_pinn.config import PipelineConfig
from coastal_pinn.core.coords import lonlat_to_utm
from coastal_pinn.core.io import write_pickle_atomic, read_pickle
from coastal_pinn.core.paths import data_path
from coastal_pinn.core.schema import ensure_utc
from coastal_pinn.exceptions import SourceUnavailable
from coastal_pinn.sources.transects import (
    generate_transects,
    transect_intersection_distance,
)


def _patch_skimage_io():
    """Inject a Pillow-backed ``skimage.io`` shim so CoastSat can import.

    On some Windows setups, ``skimage.io`` crashes with ``0xc06d007f``
    (MSVC MODULE_NOT_FOUND) because its native image-format DLLs conflict
    with conda's shared libraries.  CoastSat only uses ``skimage.io.imsave``
    and ``skimage.io.imread``, so we replace the whole module with thin
    wrappers around Pillow / imageio.

    Also forces the matplotlib ``Agg`` backend so CoastSat's figure-saving
    code doesn't try to open a Tk window (which fails headless).
    """
    if "skimage.io" in sys.modules:
        return

    # Force non-interactive matplotlib backend before CoastSat touches it.
    import matplotlib
    matplotlib.use("Agg")

    # Patch the figure manager so CoastSat's ``mng.window.showMaximized()``
    # calls don't crash with the Agg backend (which has no window).
    import matplotlib.backend_bases as _backend_bases
    _orig_fm_init = _backend_bases.FigureManagerBase.__init__
    def _patched_fm_init(self, *args, **kwargs):
        _orig_fm_init(self, *args, **kwargs)
        if not hasattr(self, "window"):
            class _DummyWindow:
                def showMaximized(self): pass
                def __getattr__(self, name): pass
            self.window = _DummyWindow()
    _backend_bases.FigureManagerBase.__init__ = _patched_fm_init

    import types
    import imageio.v3 as _iio3

    mod = types.ModuleType("skimage.io")
    mod.imsave = lambda fname, image, **kw: _iio3.imwrite(fname, image, **kw)
    mod.imread = lambda fname, **kw: _iio3.imread(fname, **kw)
    mod.imsave.__doc__ = "Pillow-backed imsave shim."
    mod.imread.__doc__ = "imageio-backed imread shim."
    sys.modules["skimage.io"] = mod
    try:
        import skimage
        skimage.io = mod  # type: ignore[attr-defined]
    except ImportError:
        pass


# Where CoastSat expects the cloned repo on disk
COASTSAT_REPO_PATHS = [
    Path("/content/CoastSat"),
    Path(os.environ.get("COASTSAT_REPO", "")),
    Path("CoastSat"),
]


def _resolve_coastsat_repo() -> Path | None:
    for p in COASTSAT_REPO_PATHS:
        if p and p != Path("") and p.exists():
            return p
    return None


def fetch_shorelines(cfg: PipelineConfig) -> pd.DataFrame:
    """Fetch shoreline polylines via GEE + CoastSat and intersect with N transects.

    Returns a long-format DataFrame with one row per (date, transect).
    The polylines are intersected with each of the N transects generated
    from cfg.region.baseline. The intersection distance from each transect
    origin is the cross-shore shoreline position S(x_n, t).
    """
    if not cfg.shoreline_enabled:
        raise SourceUnavailable("shoreline", "disabled in config")

    # Fast path: the per-transect intersected table is already cached. This
    # skips BOTH the (slow) CoastSat download and the per-transect
    # intersection, which otherwise re-runs on every call.
    #
    # NB: this cache is keyed only by region + time window, NOT by transect
    # geometry. If you change region.baseline / transect_spacing_m /
    # transect_length_m, delete the *.transects.pkl file to force a rebuild.
    intersected_cache = data_path(cfg, "shoreline", suffix="transects.pkl")
    if intersected_cache.exists():
        print("[shoreline       ] intersected-table cache hit, loading...", flush=True)
        return read_pickle(intersected_cache)

    cache = data_path(cfg, "shoreline", suffix="pkl")
    if cache.exists():
        print("[shoreline       ] cache hit, intersecting with transects...", flush=True)
        shoreline_dict = read_pickle(cache)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        print("[shoreline       ] downloading satellite images via GEE + CoastSat "
              f"(bbox={cfg.region.bbox}, {cfg.t_start} to {cfg.t_end})...", flush=True)
        print("[shoreline       ] this is the slowest source — expect 2-5 min per "
              "satellite image...", flush=True)
        try:
            shoreline_dict = _download_shorelines(cfg)
        except SourceUnavailable:
            raise
        except Exception as e:
            raise SourceUnavailable("shoreline",
                f"GEE/CoastSat fetch failed for {cfg.region.name}: {e}",
                cause=e) from e
        write_pickle_atomic(shoreline_dict, cache)
        print(f"[shoreline       ] CoastSat done: {len(shoreline_dict.get('dates', []))} "
              "cloud-free dates cached", flush=True)

    print(f"[shoreline       ] intersecting {len(shoreline_dict.get('dates', []))} "
          "dates with transects...", flush=True)
    df = _intersect_with_transects(shoreline_dict, cfg)
    write_pickle_atomic(df, intersected_cache)
    print(f"[shoreline       ] cached intersected table ({len(df)} rows) "
          f"-> {intersected_cache.name}", flush=True)
    return df


def _download_shorelines(cfg: PipelineConfig) -> dict[str, Any]:
    """Run the CoastSat workflow end-to-end.

    Automatically tiles large polygons to stay under GEE's 50 MB export
    limit.  Each tile is downloaded and processed separately; the results
    are merged before shoreline extraction so that the output dict has
    the same ``{dates, shorelines, satname}`` structure that CoastSat
    normally returns.
    """
    repo = _resolve_coastsat_repo()
    if repo is None:
        raise FileNotFoundError(
            "CoastSat repo not found. Clone it with:\n"
            "  git clone https://github.com/kvos/CoastSat.git\n"
            "and either place it at ./CoastSat, /content/CoastSat, "
            "or set COASTSAT_REPO=/path/to/CoastSat."
        )

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    _patch_skimage_io()

    try:
        import ee                                       # noqa: F401
        import coastsat                                # noqa: F401
        from coastsat import SDS_download, SDS_preprocess, SDS_shoreline  # noqa: F401
    except ImportError as e:
        raise SourceUnavailable("shoreline",
            f"CoastSat/GDAL dependencies not installed: {e}. "
            "Install GDAL: conda install -c conda-forge gdal  OR  "
            "pip install GDAL (requires system GDAL library). "
            "Then install CoastSat: pip install CoastSat",
            cause=e) from e

    try:
        ee.Initialize(project=cfg.shoreline_gee_project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=cfg.shoreline_gee_project)

    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox
    polygon = [
        [lon_min, lat_min], [lon_max, lat_min],
        [lon_max, lat_max], [lon_min, lat_max],
    ]

    inputs = {
        "polygon": polygon,
        "dates": [cfg.t_start, cfg.t_end],
        "sat_list": ["S2"],
        "sitename": cfg.shoreline_sitename,
        "filepath": str(cfg.data_dir),
    }
    settings = {
        "inputs": inputs,
        "cloud_thresh": 0.5,
        "dist_clouds": 300,
        # Output in the region's UTM projection so polylines and transects
        # are in the same CRS for intersection.
        "output_epsg": cfg.region.epsg,
        "min_beach_area": 4500,
        "min_length_sl": 500,
        "s2cloudless_prob": 60,
        "sand_color": "default",
        "check_detection": False,
        # QA figures are visual-only and dominate per-image processing time.
        # Off by default; controlled via cfg.shoreline_save_qc.
        "save_figure": cfg.shoreline_save_qc,
        "adjust_detection": False,
        "pan_off": False,
        "cloud_mask_issue": False,
        "buffer_size": 150,
    }

    MAX_TILE_DEG = 0.10
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    if lon_span <= MAX_TILE_DEG and lat_span <= MAX_TILE_DEG:
        print("[shoreline       ] downloading satellite images from GEE...", flush=True)
        metadata = SDS_download.retrieve_images(inputs)
        if cfg.shoreline_save_qc:
            print("[shoreline       ] preprocessing images (cloud masking, jpg)...", flush=True)
            SDS_preprocess.save_jpg(metadata, settings)
        print("[shoreline       ] extracting shorelines (pixel classifier + sub-pixel)...", flush=True)
        result = SDS_shoreline.extract_shorelines(metadata, settings)
        print(f"[shoreline       ] extracted {len(result.get('dates', []))} shorelines", flush=True)
        return result

    import concurrent.futures

    tiles = _tile_polygon(polygon, MAX_TILE_DEG)
    print(f"[shoreline       ] bbox too large, tiling into {len(tiles)} tiles. Downloading with 18 threads...", flush=True)
    merged_result: dict[str, list] = {}

    def _process_tile(tile_idx, tile):
        tile_inputs = dict(inputs)
        tile_inputs["polygon"] = tile
        tile_inputs["sitename"] = f"{inputs['sitename']}_tile{tile_idx}"

        tile_settings = dict(settings)
        tile_settings["inputs"] = tile_inputs

        print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: downloading...", flush=True)
        try:
            metadata = SDS_download.retrieve_images(tile_inputs)
            if cfg.shoreline_save_qc:
                print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: preprocessing...", flush=True)
                SDS_preprocess.save_jpg(metadata, tile_settings)
            print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: extracting shorelines...", flush=True)

            tile_result = SDS_shoreline.extract_shorelines(metadata, tile_settings)
            return tile_result
        except Exception as e:
            print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)} failed: {e}. Skipping.", flush=True)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_tile, i, tile): i for i, tile in enumerate(tiles)}
        for future in concurrent.futures.as_completed(futures):
            tile_result = future.result()
            if tile_result is not None:
                for key, value in tile_result.items():
                    merged_result.setdefault(key, []).extend(value)

    return merged_result


def _tile_polygon(polygon: list[list[float]], max_deg: float = 0.25) -> list[list[list[float]]]:
    """Split a rectangular polygon into smaller tiles for GEE export."""
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    tiles: list[list[list[float]]] = []
    lon = lon_min
    while lon < lon_max:
        lat = lat_min
        while lat < lat_max:
            tiles.append([
                [lon, lat],
                [min(lon + max_deg, lon_max), lat],
                [min(lon + max_deg, lon_max), min(lat + max_deg, lat_max)],
                [lon, min(lat + max_deg, lat_max)],
            ])
            lat += max_deg
        lon += max_deg
    return tiles


def _intersect_with_transects(shoreline_dict: dict[str, Any],
                              cfg: PipelineConfig) -> pd.DataFrame:
    """Intersect CoastSat polylines with N transects generated from baseline.

    Returns a long-format DataFrame with columns:
        region, timestamp, transect_id, along_shore_x_m, cross_shore_S_m, sat

    Uses CoastSat's SDS_transects.compute_intersection_QC if available
    (recommended per the user's choice); falls back to a shapely-based
    per-transect intersection otherwise.
    """
    transects_df = generate_transects(cfg.region)
    n_transects = len(transects_df)

    # Normalize the CoastSat output to a flat list of (timestamp, polyline, sat)
    polylines: list[tuple[Any, np.ndarray, str]] = []
    if "dates" in shoreline_dict and "shorelines" in shoreline_dict:
        sat_list = shoreline_dict.get("satname")
        for idx in range(len(shoreline_dict["dates"])):
            ts = shoreline_dict["dates"][idx]
            sl = shoreline_dict["shorelines"][idx]
            sat = sat_list[idx] if sat_list is not None and idx < len(sat_list) else "UNK"
            if sl is None:
                continue
            arr = np.asarray(sl)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.size == 0 or arr.ndim < 2:
                continue
            polylines.append((ts, arr, sat))
    else:
        for sat, data in shoreline_dict.items():
            if not isinstance(data, dict) or "dates" not in data:
                continue
            for ts, sl in zip(data["dates"], data["shorelines"]):
                if sl is None:
                    continue
                arr = np.asarray(sl)
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.size == 0 or arr.ndim < 2:
                    continue
                polylines.append((ts, arr, str(sat)))

    if not polylines:
        raise SourceUnavailable("shoreline",
            "CoastSat returned no polylines for the given time window and ROI")

    # Convert polylines to UTM if they're in lon/lat (only needed if
    # output_epsg was not honored or CoastSat's SDS_transects isn't used).
    # Our settings set output_epsg = region.epsg so they should already be UTM.
    use_coastsat_intersect = _try_use_coastsat_intersect(
        shoreline_dict, transects_df, cfg
    )
    if use_coastsat_intersect is not None:
        _warn_if_low_intersection(use_coastsat_intersect, n_transects)
        return use_coastsat_intersect

    # Shapely fallback: per-transect intersection for each (date, polyline)
    rows: list[dict[str, Any]] = []
    for ts, polyline_xy, sat in polylines:
        ts_utc = ensure_utc(pd.Timestamp(ts))
        for _, tr in transects_df.iterrows():
            d = transect_intersection_distance(
                polyline_xy,
                (float(tr["origin_x"]), float(tr["origin_y"])),
                (float(tr["end_x"]), float(tr["end_y"])),
            )
            if d is None or not np.isfinite(d):
                continue
            # Clamp negative distances to 0 (shoreline cannot be inland of
            # baseline by definition)
            d_clamped = max(0.0, float(d))
            rows.append({
                "timestamp": ts_utc,
                "transect_id": int(tr["transect_id"]),
                "along_shore_x_m": float(tr["along_shore_x_m"]),
                "cross_shore_S_m": d_clamped,
                "sat": str(sat),
            })

    if not rows:
        raise SourceUnavailable("shoreline",
            "No intersections between polylines and transects for this region/window")

    df = pd.DataFrame(rows)
    df["region"] = cfg.region.name
    _warn_if_low_intersection(df, n_transects)
    return df[["region", "timestamp", "transect_id", "along_shore_x_m",
               "cross_shore_S_m", "sat"]]


def _warn_if_low_intersection(df: pd.DataFrame, n_transects: int) -> None:
    """Warn if few transects intersect the shoreline (geometry mismatch).

    A low hit rate means the baseline is mis-placed relative to the detected
    shoreline, the transects point the wrong way, or they are too short — i.e.
    the kind of error that yields an empty/degenerate training label.
    """
    import warnings
    if df.empty or n_transects == 0:
        return
    hit_frac = df["transect_id"].nunique() / n_transects
    if hit_frac < 0.50:
        warnings.warn(
            f"[shoreline] only {hit_frac:.0%} of {n_transects} transects "
            f"intersect the shoreline. Likely a geometry mismatch: check "
            f"region.baseline placement, region.ocean_side, and "
            f"transect_length_m.",
            stacklevel=2,
        )


def _try_use_coastsat_intersect(shoreline_dict, transects_df, cfg):
    """Try to use CoastSat's SDS_transects.compute_intersection_QC.

    Returns a DataFrame in our schema, or None if CoastSat's transect
    module isn't available / the polylines aren't in the expected format.
    """
    repo = _resolve_coastsat_repo()
    if repo is None:
        return None
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from coastsat import SDS_transects  # type: ignore
    except Exception:
        return None

    # Build CoastSat transects dict: {name: np.array of [[x0,y0],[x1,y1]]}
    cs_transects: dict[str, np.ndarray] = {}
    for _, tr in transects_df.iterrows():
        name = f"t{int(tr['transect_id']):04d}"
        cs_transects[name] = np.array([
            [float(tr["origin_x"]), float(tr["origin_y"])],
            [float(tr["end_x"]), float(tr["end_y"])],
        ])

    settings_qc = {
        "along_dist": 25,
        "min_points": 3,
        "max_std": 15,
        "max_range": 30,
        "min_chainage": -100,
        "multiple_inter": "auto",
        "auto_prc": 0.1,
    }
    try:
        cross_distance = SDS_transects.compute_intersection_QC(
            shoreline_dict, cs_transects, settings_qc
        )
    except Exception:
        return None

    # cross_distance is {transect_name: np.array of per-date distances}
    rows: list[dict[str, Any]] = []
    # Need to recover timestamps and sat per index
    if "dates" in shoreline_dict and "shorelines" in shoreline_dict:
        dates = list(shoreline_dict["dates"])
        sats = list(shoreline_dict.get("satname") or ["UNK"] * len(dates))
    else:
        dates, sats = [], []
        for sat, data in shoreline_dict.items():
            if isinstance(data, dict) and "dates" in data:
                for ts in data["dates"]:
                    dates.append(ts)
                    sats.append(str(sat))
    if len(sats) < len(dates):
        sats = sats + ["UNK"] * (len(dates) - len(sats))

    transect_by_name = {f"t{int(r['transect_id']):04d}": r
                        for _, r in transects_df.iterrows()}
    for name, distances in cross_distance.items():
        tr = transect_by_name.get(name)
        if tr is None:
            continue
        for i, d in enumerate(distances):
            if i >= len(dates):
                break
            if d is None or not np.isfinite(d):
                continue
            d_clamped = max(0.0, float(d))
            rows.append({
                "timestamp": ensure_utc(pd.Timestamp(dates[i])),
                "transect_id": int(tr["transect_id"]),
                "along_shore_x_m": float(tr["along_shore_x_m"]),
                "cross_shore_S_m": d_clamped,
                "sat": sats[i] if i < len(sats) else "UNK",
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["region"] = cfg.region.name
    return df[["region", "timestamp", "transect_id", "along_shore_x_m",
               "cross_shore_S_m", "sat"]]
