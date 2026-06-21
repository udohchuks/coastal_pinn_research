"""Shoreline source: Google Earth Engine + CoastSat.

Fetches cloud-free Sentinel-2 / Landsat-8 imagery via Earth Engine, runs
the CoastSat shoreline classifier, and returns the per-date polylines as
UTM (easting, northing) coordinates -- matching the per-paper
convention (PDF Figure 1, the −7.68 m/yr rate is computed in UTM
Easting).

Outputs DataFrame with columns:
    region       (str)
    timestamp    (pd.Timestamp, UTC, per cloud-free satellite date)
    sat          (str, 'S2' | 'L8')
    pt_idx       (int, vertex index along the polyline)
    easting_m    (float, UTM Easting, m)
    northing_m   (float, UTM Northing, m)

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
    # Also register as a submodule of skimage so ``from skimage.io import …``
    # resolves to our shim.
    try:
        import skimage
        skimage.io = mod  # type: ignore[attr-defined]
    except ImportError:
        pass


# Where CoastSat expects the cloned repo on disk, and where it puts
# classifier models. In Colab these are /content/CoastSat and /content/classification.
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
    """Fetch shoreline polylines via GEE + CoastSat for cfg.region.

    Returns a long DataFrame with one row per (date, vertex).

    The polylines are stored in lon/lat and converted to UTM at the
    fetch boundary (per the paper's UTM-based rate calculation).
    """
    if not cfg.shoreline_enabled:
        raise SourceUnavailable("shoreline", "disabled in config")

    cache = data_path(cfg, "shoreline", suffix="pkl")
    if cache.exists():
        shoreline_dict = read_pickle(cache)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            shoreline_dict = _download_shorelines(cfg)
        except SourceUnavailable:
            raise
        except Exception as e:
            raise SourceUnavailable("shoreline",
                f"GEE/CoastSat fetch failed for {cfg.region.name}: {e}",
                cause=e) from e
        write_pickle_atomic(shoreline_dict, cache)

    return _to_dataframe(shoreline_dict, cfg)


def _tile_polygon(polygon: list[list[float]], max_deg: float = 0.25) -> list[list[list[float]]]:
    """Split a rectangular polygon into smaller tiles.

    GEE's ``getDownloadURL`` has a 50 MB limit. Large polygons at 10 m
    resolution can exceed this. Tiling keeps each export under the cap.
    """
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

    # Patch skimage.io before CoastSat imports it to avoid a Windows DLL
    # crash (0xc06d007f) caused by conflicting native image libraries.
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
        # First-time auth: opens a browser. Token is then cached.
        ee.Authenticate()
        ee.Initialize(project=cfg.shoreline_gee_project)

    # CoastSat expects a polygon as list of [lon, lat] vertices
    lon_min, lat_min, lon_max, lat_max = cfg.region.bbox
    polygon = [
        [lon_min, lat_min], [lon_max, lat_min],
        [lon_max, lat_max], [lon_min, lat_max],
    ]

    inputs = {
        "polygon": polygon,
        "dates": [cfg.t_start, cfg.t_end],
        "sat_list": ["S2", "L8"],
        "sitename": cfg.shoreline_sitename,
        "filepath": str(cfg.data_dir),
    }
    settings = {
        "inputs": inputs,
        "cloud_thresh": 0.5,
        "dist_clouds": 300,
        "output_epsg": cfg.region.epsg,
        "min_beach_area": 4500,
        "min_length_sl": 500,
        "s2cloudless_prob": 60,
        "sand_color": "default",
        "check_detection": False,
        "save_figure": True,
        "adjust_detection": False,
        "pan_off": False,
        "cloud_mask_issue": False,
        "buffer_size": 150,
    }

    # If the bbox is small enough, a single export stays under 50 MB.
    # At 10 m resolution: 0.10° ≈ 11 km ≈ 1100 px.  With 13 S2 bands
    # at 2 bytes each that is ~31 MB — safely under the 50 MB cap.
    MAX_TILE_DEG = 0.10
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    if lon_span <= MAX_TILE_DEG and lat_span <= MAX_TILE_DEG:
        metadata = SDS_download.retrieve_images(inputs)
        SDS_preprocess.save_jpg(metadata, settings)
        return SDS_shoreline.extract_shorelines(metadata, settings)

    # Large polygon – tile and merge downloads before shoreline extraction
    tiles = _tile_polygon(polygon, MAX_TILE_DEG)
    merged_result: dict[str, list] = {}

    for tile_idx, tile in enumerate(tiles):
        tile_inputs = dict(inputs)
        tile_inputs["polygon"] = tile
        tile_inputs["sitename"] = f"{inputs['sitename']}_tile{tile_idx}"

        tile_settings = dict(settings)
        tile_settings["inputs"] = tile_inputs

        metadata = SDS_download.retrieve_images(tile_inputs)
        SDS_preprocess.save_jpg(metadata, tile_settings)

        # Extract shorelines per tile (different tiles can have different
        # pixel dimensions, so we can't merge metadata across tiles).
        tile_result = SDS_shoreline.extract_shorelines(metadata, tile_settings)
        for key, value in tile_result.items():
            merged_result.setdefault(key, []).extend(value)

    return merged_result


def _to_dataframe(shoreline_dict: dict[str, Any], cfg: PipelineConfig) -> pd.DataFrame:
    """Flatten a CoastSat output dict into a (region, timestamp, sat, pt_idx,
    easting_m, northing_m) DataFrame.

    Handles both flat-merged and per-satellite dict layouts (CoastSat
    2.x and 1.x). Coordinates are converted from lon/lat to UTM at this
    boundary, per the paper's UTM-based convention.
    """
    rows: list[dict[str, Any]] = []
    if not isinstance(shoreline_dict, dict):
        raise SourceUnavailable("shoreline",
            f"unexpected shoreline_dict type: {type(shoreline_dict).__name__}")

    if "dates" in shoreline_dict and "shorelines" in shoreline_dict:
        # Merged format: {dates: [...], shorelines: [...], satname: [...]}
        for idx in range(len(shoreline_dict["dates"])):
            ts = shoreline_dict["dates"][idx]
            sl = shoreline_dict["shorelines"][idx]
            sat_list = shoreline_dict.get("satname")
            sat = sat_list[idx] if sat_list is not None and idx < len(sat_list) else "UNK"
            _append_polyline(rows, ts, sl, sat, cfg)
    else:
        # Per-satellite format: {sat: {dates, shorelines}}
        for sat, data in shoreline_dict.items():
            if not isinstance(data, dict) or "dates" not in data:
                continue
            for ts, sl in zip(data["dates"], data["shorelines"]):
                _append_polyline(rows, ts, sl, sat, cfg)

    if not rows:
        raise SourceUnavailable("shoreline",
            "CoastSat returned no polylines for the given time window and ROI")

    df = pd.DataFrame(rows)
    df["region"] = cfg.region.name
    df["timestamp"] = ensure_utc(df["timestamp"])
    return df[["region", "timestamp", "sat", "pt_idx", "easting_m", "northing_m"]]


def _append_polyline(rows: list[dict[str, Any]], ts, sl, sat: str,
                     cfg: PipelineConfig) -> None:
    """Append one polyline to the rows list, converting lon/lat to UTM."""
    if sl is None:
        return
    arr = np.asarray(sl)
    if arr.size == 0 or arr.ndim < 2:
        return
    lons = arr[:, 0]
    lats = arr[:, 1]
    easting, northing = lonlat_to_utm(lons, lats, cfg.region.utm_zone)
    ts_utc = ensure_utc(pd.Timestamp(ts))
    for i, (e, n) in enumerate(zip(easting, northing)):
        rows.append({
            "timestamp": ts_utc,
            "sat": str(sat),
            "pt_idx": int(i),
            "easting_m": float(e),
            "northing_m": float(n),
        })