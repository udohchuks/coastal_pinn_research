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
import shutil
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

def _patch_os_rename():
    """Inject a retry loop into os.rename to bypass Windows file locks (WinError 32).
    
    CoastSat's SDS_download script downloads TIFFs and immediately renames them. On Windows, 
    background processes like Defender or OneDrive often lock new files briefly, causing 
    os.rename to crash. This adds a retry loop to handle those ephemeral locks.
    """
    import os
    import time
    _orig_rename = os.rename
    def _patched_rename(src, dst):
        for attempt in range(15):
            try:
                return _orig_rename(src, dst)
            except OSError as e:
                # WinError 32: The process cannot access the file because it is being used by another process.
                if getattr(e, "winerror", None) == 32:
                    time.sleep(1)
                else:
                    raise
        return _orig_rename(src, dst)
    os.rename = _patched_rename


def _patch_process_shoreline():
    """Inject cKDTree into SDS_shoreline.process_shoreline to vectorize cloud filtering."""
    import sys
    import numpy as np
    from scipy.spatial import cKDTree

    if "coastsat.SDS_shoreline" not in sys.modules:
        return
    SDS_shoreline = sys.modules["coastsat.SDS_shoreline"]
    
    if getattr(SDS_shoreline, "_patched_process_shoreline", False):
        return
        
    def _fast_process_shoreline(contours, cloud_mask, im_nodata, georef, image_epsg, settings):
        from coastsat import SDS_tools
        from shapely.geometry import LineString
        
        contours_world = SDS_tools.convert_pix2world(contours, georef)
        contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
        
        contours_long = []
        for wl in contours_epsg:
            coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
            a = LineString(coords)
            if a.length >= settings['min_length_sl']:
                contours_long.append(wl)
                
        if len(contours_long) == 0:
            return np.zeros((0, 2))
            
        x_points = np.concatenate([c[:,0] for c in contours_long])
        y_points = np.concatenate([c[:,1] for c in contours_long])
        shoreline = np.column_stack((x_points, y_points))
        
        if np.sum(cloud_mask) > 0:
            idx_cloud = np.where(cloud_mask)
            idx_cloud = np.column_stack((idx_cloud[0], idx_cloud[1]))
            coords_cloud = SDS_tools.convert_epsg(SDS_tools.convert_pix2world(idx_cloud, georef),
                                                   image_epsg, settings['output_epsg'])
            tree = cKDTree(coords_cloud)
            dists, _ = tree.query(shoreline, k=1, distance_upper_bound=settings['dist_clouds'])
            idx_keep = dists >= settings['dist_clouds']
            shoreline = shoreline[idx_keep]
            
        if len(shoreline) > 0 and np.sum(im_nodata) > 0:
            idx_cloud = np.where(im_nodata)
            idx_cloud = np.column_stack((idx_cloud[0], idx_cloud[1]))
            coords_cloud = SDS_tools.convert_epsg(SDS_tools.convert_pix2world(idx_cloud, georef),
                                                   image_epsg, settings['output_epsg'])
            tree = cKDTree(coords_cloud)
            dists, _ = tree.query(shoreline, k=1, distance_upper_bound=30)
            idx_keep = dists >= 30
            shoreline = shoreline[idx_keep]
            
        return shoreline

    SDS_shoreline.process_shoreline = _fast_process_shoreline
    SDS_shoreline._patched_process_shoreline = True


def _patch_extract_shorelines_print():
    """Inject a wrapper around print to set flush=True."""
    import builtins
    import sys
    if "coastsat.SDS_shoreline" not in sys.modules:
        return
    SDS_shoreline = sys.modules["coastsat.SDS_shoreline"]
    
    if getattr(SDS_shoreline, "_patched_print", False):
        return

    _orig_print = builtins.print
    def _flushed_print(*args, **kwargs):
        kwargs["flush"] = True
        return _orig_print(*args, **kwargs)
        
    SDS_shoreline.print = _flushed_print
    SDS_shoreline._patched_print = True


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

    _patch_process_shoreline()
    _patch_extract_shorelines_print()

    # #3 Server-side cloud pre-filter. CoastSat hardcodes a loose 95% scene
    # cloud-cover cutoff inside get_image_info() (it calls remove_cloudy_images
    # with the default prc_cloud_cover). get_image_info looks the function up in
    # the module namespace at call time, so reassigning it here lowers the
    # threshold and we never download near-hopeless scenes. Idempotent: the
    # original is stashed so repeated calls don't re-wrap.
    import functools
    _orig_remove_cloudy = getattr(SDS_download, "_orig_remove_cloudy_images", None)
    if _orig_remove_cloudy is None:
        _orig_remove_cloudy = SDS_download.remove_cloudy_images
        SDS_download._orig_remove_cloudy_images = _orig_remove_cloudy
    SDS_download.remove_cloudy_images = functools.partial(
        _orig_remove_cloudy, prc_cloud_cover=cfg.shoreline_cloud_cover_max
    )
    print(f"[shoreline       ] cloud pre-filter: dropping scenes >"
          f"{cfg.shoreline_cloud_cover_max}% cloud before download", flush=True)

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
        "sitename": f"{cfg.shoreline_sitename}_{cfg.t_start}_to_{cfg.t_end}",
        "filepath": str(cfg.data_dir),
    }
    settings = {
        "inputs": inputs,
        "cloud_thresh": cfg.shoreline_cloud_cover_max / 100.0,
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

    if cfg.region.baseline:
        try:
            from coastal_pinn.sources.transects import generate_transects
            transects_df = generate_transects(cfg.region)
            mid_x = (transects_df["origin_x"] + transects_df["end_x"]) / 2.0
            mid_y = (transects_df["origin_y"] + transects_df["end_y"]) / 2.0
            settings["reference_shoreline"] = np.column_stack((mid_x, mid_y))
            settings["max_dist_ref"] = 150
            print("[shoreline       ] applied reference shoreline buffer (150m)", flush=True)
        except Exception as e:
            print(f"[shoreline       ] Could not generate reference shoreline: {e}", flush=True)

    MAX_TILE_DEG = 0.10
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    if lon_span <= MAX_TILE_DEG and lat_span <= MAX_TILE_DEG:
        site_dir = Path(inputs["filepath"]) / inputs["sitename"]
        print("[shoreline       ] downloading satellite images from GEE...", flush=True)
        metadata = SDS_download.retrieve_images(inputs)
        if cfg.shoreline_save_qc:
            print("[shoreline       ] preprocessing images (cloud masking, jpg)...", flush=True)
            SDS_preprocess.save_jpg(metadata, settings)
        print("[shoreline       ] extracting shorelines (pixel classifier + sub-pixel)...", flush=True)
        result = SDS_shoreline.extract_shorelines(metadata, settings)
        print(f"[shoreline       ] extracted {len(result.get('dates', []))} shorelines", flush=True)
        # Reclaim the downloaded TIFFs once shorelines are extracted.
        shutil.rmtree(site_dir, ignore_errors=True)
        return result

    import concurrent.futures

    tiles = _tile_polygon(polygon, MAX_TILE_DEG)

    # #2 Prune tiles that don't touch the coastline. The bbox is a rectangle
    # but the coast is a thin line through it, so many tiles are pure ocean or
    # inland and would download imagery with no shoreline to extract. Keep a
    # tile if any baseline vertex falls within it (expanded by a buffer that
    # covers the seaward transect extent, since the shoreline lies offshore of
    # the baseline). Tile indices stay tied to the full grid so the per-tile
    # cache keys (sitenames) are stable regardless of pruning.
    baseline = cfg.region.baseline
    if baseline:
        buffer_deg = max(0.01, (cfg.region.transect_length_m / 111_000.0) * 1.5)
        kept = [(i, t) for i, t in enumerate(tiles)
                if _tile_has_coast(t, baseline, buffer_deg)]
    else:
        kept = list(enumerate(tiles))
    n_skipped = len(tiles) - len(kept)

    # #1 Concurrency. Each in-flight tile's TIFFs are deleted right after
    # extraction (see _process_tile), so peak disk ~= DOWNLOAD_WORKERS tiles.
    DOWNLOAD_WORKERS = cfg.shoreline_download_workers
    print(f"[shoreline       ] tiling into {len(tiles)} tiles; {len(kept)} intersect "
          f"the coast, skipping {n_skipped} ocean/land tiles. Downloading with "
          f"{DOWNLOAD_WORKERS} threads (imagery deleted per tile after extraction)...",
          flush=True)
    merged_result: dict[str, list] = {}

    def _process_tile(tile_idx, tile):
        tile_inputs = dict(inputs)
        tile_inputs["polygon"] = tile
        tile_inputs["sitename"] = f"{inputs['sitename']}_tile{tile_idx}"

        tile_settings = dict(settings)
        tile_settings["inputs"] = tile_inputs

        # CoastSat writes this tile's downloaded imagery (the multi-year S2
        # stack, several GB) into <filepath>/<sitename>/. The shorelines are
        # extracted into tile_result (in memory), after which the raw TIFFs
        # are no longer needed. We delete the tile folder as soon as the tile
        # is done so peak disk stays ~one tile's worth instead of all tiles
        # at once — essential on a single-drive machine.
        tile_dir = Path(inputs["filepath"]) / tile_inputs["sitename"]
        tile_cache = Path(inputs["filepath"]) / "shoreline" / f"{tile_inputs['sitename']}.pkl"

        if tile_cache.exists():
            print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: loaded from tile cache ({tile_inputs['sitename']})", flush=True)
            return read_pickle(tile_cache)

        print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: downloading...", flush=True)
        try:
            metadata = SDS_download.retrieve_images(tile_inputs)
            if cfg.shoreline_save_qc:
                print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: preprocessing...", flush=True)
                SDS_preprocess.save_jpg(metadata, tile_settings)
            print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: extracting shorelines...", flush=True)

            tile_result = SDS_shoreline.extract_shorelines(metadata, tile_settings)
            
            # Save the cache incrementally!
            tile_cache.parent.mkdir(parents=True, exist_ok=True)
            write_pickle_atomic(tile_result, tile_cache)
            
            return tile_result
        except Exception as e:
            print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)} failed: {e}. Skipping.", flush=True)
            return None
        finally:
            # Reclaim the tile's imagery only if extraction succeeded and was cached.
            # This allows interrupted pipelines to reuse already-downloaded imagery.
            if tile_cache.exists():
                shutil.rmtree(tile_dir, ignore_errors=True)
                print(f"[shoreline       ] tile {tile_idx+1}/{len(tiles)}: "
                      f"cleaned up imagery ({tile_dir.name})", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_process_tile, i, tile): i for i, tile in kept}
        for future in concurrent.futures.as_completed(futures):
            tile_result = future.result()
            if tile_result is not None:
                for key, value in tile_result.items():
                    merged_result.setdefault(key, []).extend(value)

    return merged_result


def _tile_has_coast(tile: list[list[float]],
                    baseline: "tuple[tuple[float, float], ...]",
                    buffer_deg: float) -> bool:
    """True if any baseline (lon, lat) vertex falls within `tile`, expanded by
    `buffer_deg`.

    The baseline vertices are far denser than the tile size, so every tile the
    coast passes through contains at least one vertex; `buffer_deg` extends the
    test to catch the shoreline where it lies just seaward of the baseline (or
    just across a tile boundary). Tiles with no coast are pure ocean/inland and
    are skipped to avoid downloading imagery with no shoreline to extract.
    """
    lons = [p[0] for p in tile]
    lats = [p[1] for p in tile]
    lon0, lon1 = min(lons) - buffer_deg, max(lons) + buffer_deg
    lat0, lat1 = min(lats) - buffer_deg, max(lats) + buffer_deg
    for lon, lat in baseline:
        if lon0 <= lon <= lon1 and lat0 <= lat <= lat1:
            return True
    return False


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

    # Shapely fallback: per-transect intersection for each (date, polyline).
    # Pre-convert transect rows to lists for fast iteration (avoids pandas
    # per-row overhead inside the hot loop).
    transect_rows = [
        (
            int(tr["transect_id"]),
            float(tr["along_shore_x_m"]),
            (float(tr["origin_x"]), float(tr["origin_y"])),
            (float(tr["end_x"]),   float(tr["end_y"])),
        )
        for _, tr in transects_df.iterrows()
    ]
    n_transects = len(transect_rows)
    n_dates     = len(polylines)
    print(f"[shoreline       ] Shapely intersection: {n_dates} dates × "
          f"{n_transects} transects...", flush=True)
    rows: list[dict[str, Any]] = []
    for idx, (ts, polyline_xy, sat) in enumerate(polylines):
        if idx % 100 == 0:
            print(f"[shoreline       ]   date {idx}/{n_dates} ...", flush=True)
        ts_utc = ensure_utc(pd.Timestamp(ts))
        for tid, along_x, origin, end in transect_rows:
            d = transect_intersection_distance(polyline_xy, origin, end)
            if d is None or not np.isfinite(d):
                continue
            d_clamped = max(0.0, float(d))
            rows.append({
                "timestamp": ts_utc,
                "transect_id": tid,
                "along_shore_x_m": along_x,
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
    """Intentionally disabled: CoastSat's compute_intersection_QC contains an
    O(N_transects × N_dates × N_shoreline_points) pure-Python inner loop that
    hangs for large inputs (e.g. 1,173 transects × 2,395 dates × ~10k points
    per polyline). We always fall back to our own vectorized Shapely-based
    intersection, which is correct and fast.

    Returns None unconditionally so _intersect_with_transects uses the Shapely
    fallback path.
    """
    return None

