# Data Documentation — Coastal PINN Data Pipeline

This document explains everything about the data: what we use, where it comes from, why we chose it, how we process it, and how it maps to the model equations. It is the single reference for the data layer of the research.

---

## Table of Contents

1. [The Study Site](#1-the-study-site)
2. [The Four Data Sources](#2-the-four-data-sources)
3. [The Spatial Framework — Transects](#3-the-spatial-framework--transects)
4. [How Each Source Is Processed](#4-how-each-source-is-processed)
5. [The Reconciliation Step](#5-the-reconciliation-step)
6. [The Wide Table Schema](#6-the-wide-table-schema)
7. [Mapping Data to the Model Equations](#7-mapping-data-to-the-model-equations)
8. [Gap Handling Policy](#8-gap-handling-policy)
9. [Data Volume and Structure](#9-data-volume-and-structure)
10. [Reproducibility and Credentials](#10-reproducibility-and-credentials)
11. [File Layout](#11-file-layout)

---

## 1. The Study Site

### Keta, Ghana

The study site is the Keta coastline in southeastern Ghana, on the Gulf of Guinea.

| Parameter | Value |
|---|---|
| Bounding box | lon 0.85°E to 1.24°E, lat 5.74°N to 6.15°N |
| UTM zone | 31N (EPSG:32631) |
| Coastline orientation | **NE-trending** (the Atlantic coast runs from the Volta estuary in the SW up toward the Aflao/Togo border in the NE), with land to the NW and the open Atlantic to the SE/south |
| Observed erosion rate | severe; the eastern-Ghana DSAS study reports up to ≈ −10.5 m/yr in the Keta zone (Asare-Bediako et al. 2025, *Sci. Rep.* 15:33032) |
| Coastline length | ~40 km (the along-shore span of the transect array) |

Keta is one of the most rapidly eroding coastlines in West Africa. The Volta River delta lies to the west; the coast is exposed to swell waves from the South Atlantic.

**Important geographic note.** The *ocean* shoreline here is the Atlantic beach (~lat 5.78–6.12°N, rising NE with longitude), **not** the Keta Lagoon. The lagoon is a large brackish water body lying inland (north) of the barrier beach; its water boundary sits near lat 6.07°N. Because a naive ROI/baseline can cause CoastSat to lock onto the lagoon's water/land edge instead of the surf line, the study geometry is anchored to the OpenStreetMap `natural=coastline` (which marks only the sea coast). The authoritative study extent (Asare-Bediako et al. 2025) is lat 5°45′–6°10′N, lon 0°45′–1°10′E.

### Why Keta

- Severe, well-documented erosion making it scientifically significant.
- Located in the Gulf of Guinea — a region with very limited PINN or physics-informed ML coastal research.
- All four open-access data sources cover this region with adequate resolution.
- The coastline is approximately straight at the segment scale, making the 1-D One-Line Model a defensible simplification.

---

## 2. The Four Data Sources

The pipeline uses four open-access data sources. Each provides a specific physical variable that maps to a specific term in the model equations. No proprietary or paid data is used.

### Source 1: Shoreline Position — Google Earth Engine + CoastSat

| Attribute | Value |
|---|---|
| Provider | Google Earth Engine (imagery) + CoastSat (Vos et al. 2019, extraction) |
| Satellites | ESA Sentinel-2 (10 m, since 2015) + NASA Landsat-8 (30 m, since 2013) |
| Output | 2-D polyline of (lon, lat) points per cloud-free date |
| Resolution | Sub-pixel (~5–10 m for S2, ~15–30 m for L8) |
| Auth required | `earthengine authenticate` (one-time, per machine) |
| Cache format | `.pkl` (Python pickle of CoastSat output dict) |

**What it gives us:** The position of the sand-water boundary on each cloud-free day. For Keta 2018–2025, this yields 372 cloud-free observations. Each observation is a polyline — a series of (lon, lat) points tracing where the sand meets the water along the coast.

**How CoastSat works (in plain language):**

1. CoastSat queries the Google Earth Engine archive for cloud-free Sentinel-2 and Landsat-8 images over the Keta bounding box.
2. For each image, it runs a **trained pixel classifier** (a neural network trained on labeled coastlines worldwide) that labels every pixel as one of: water, sand, white-water, or other land.
3. It then traces the **sand/water boundary** at sub-pixel resolution using an Otsu threshold on the classified image.
4. The output is a polyline of (lon, lat) points — the detected shoreline for that date.

**Why CoastSat instead of a simple MNDWI threshold:**

| Method | Accuracy | Handles turbid water | Handles wet sand | Sub-pixel |
|---|---|---|---|---|
| MNDWI + threshold | ~1–2 pixels (30–60 m for Landsat) | Poorly | Poorly | No |
| CoastSat classifier | ~5–10 m for S2 | Yes | Yes | Yes |

The sub-pixel accuracy matters: at 30 m uncertainty (MNDWI on Landsat), the −7.68 m/yr trend is buried in noise. With CoastSat's sub-pixel accuracy, the trend is well-resolved.

**Code location:**
- Upstream CoastSat package: `CoastSat/coastsat/` (cloned repo, used as library)
- Our fetcher: `coastal_pinn/sources/shoreline.py`
- CoastSat's transect intersection: `CoastSat/coastsat/SDS_transects.py` (used by our fetcher)

---

### Source 2: Sea Level and Ocean Currents — Copernicus Marine PHY

| Attribute | Value |
|---|---|
| Provider | EU Copernicus Marine Service |
| Product (reanalysis) | `cmems_mod_glo_phy_my_0.083deg_P1D-m` (multi-year, pre-2022) |
| Product (analysis) | `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` (near-real-time, post-2022) |
| Variables | `zos` (sea surface height anomaly, m), `uo` (eastward current, m/s), `vo` (northward current, m/s) |
| Native resolution | 0.083° (~9 km), daily |
| Auth required | `copernicusmarine login` (one-time, per machine) |
| Cache format | `.nc` (NetCDF) |

**What it gives us:** The water height `h` and current velocity `u` at each transect, for each day. These feed the shallow-water PDE and the closure sub-network.

**How we process it:**
1. Download the (time, lat, lon) cube for the Keta bbox via `copernicusmarine.subset()`.
2. If the time window spans the 2022-06-01 reanalysis/analysis cutoff, download from both products and merge along the time dimension.
3. If the cube has a depth dimension, collapse it via mean (we use surface values).
4. Interpolate the (lat, lon) grid to the exact (lon, lat) of each transect's **seaward sample point** (the transect's offshore end, in open water) using `xarray.interp(method='linear')`. Sampling seaward — not at the onshore baseline origin — keeps the query off land-masked cells and returns real nearshore values, while still giving each transect its own value (no spatial averaging).
5. Compute current speed magnitude: `u_mag = sqrt(uo² + vo²)`.
6. Resample to daily means (defensive; the product is already daily).
7. Clamp query lon/lat to the data's coordinate range to avoid NaN at boundary transects.

**Why Copernicus PHY:**
- The only global product that pairs `zos` (for the PDE) with `uo`/`vo` (for the PDE) at the same daily resolution.
- Validated globally against tide gauges and Argo floats.
- 9 km resolution gives ~6–7 grid cells along the 66 km Keta segment — enough for meaningful along-shore variation.

**Code location:** `coastal_pinn/sources/sea_level.py`

---

### Source 3: Wave Forcing — Copernicus Marine WAM

| Attribute | Value |
|---|---|
| Provider | EU Copernicus Marine Service |
| Product (reanalysis) | `cmems_mod_glo_wav_my_0.2deg_PT3H-i` (multi-year, pre-2022) |
| Product (analysis) | `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i` (near-real-time, post-2022) |
| Variables | `VHM0` (significant wave height, m), `VMDR` (mean wave direction, deg meteorological) |
| Native resolution | 0.2° (~22 km) reanalysis / 0.083° (~9 km) analysis, 3-hourly |
| Auth required | Same Copernicus credentials as PHY (shared) |
| Cache format | `.nc` (NetCDF) |

**What it gives us:** The wave height `W` and wave direction `W_dir` at each transect, for each day. These feed the CERC longshore transport term `W·sin(2θ)` in the shoreline ODE.

**How we process it:**
1. Download the (time, lat, lon) cube for the Keta bbox.
2. If the time window spans the cutoff, download from both products and merge.
3. Interpolate to per-transect (lon, lat) values using `xarray.interp(method='linear')`.
4. Resample 3-hourly to daily means:
   - Wave height: linear mean.
   - Wave direction: **circular mean** via separate averaging of sin and cos components, then `arctan2(sin_mean, cos_mean)`. This is essential because 350° and 10° are 20° apart (circular), not 340° apart (arithmetic).
5. Clamp query lon/lat to data range.

**Why Copernicus WAM (not NOAA WAVEWATCH III):**

| Aspect | WAVEWATCH III | Copernicus WAM |
|---|---|---|
| Resolution | 0.5° (~55 km) | 0.083° (~9 km) |
| Cells along Keta | ~1–2 | ~6–7 |
| Auth | None | Copernicus login |
| Forcing consistency | Different wind forcing than PHY | Same ECMWF wind forcing as PHY |

WAM's higher resolution preserves along-shore wave gradients needed by the CERC longshore term. Shared ECMWF forcing with PHY ensures physical consistency between wave and ocean variables. The auth is shared with the PHY product — no additional credential setup.

**Code location:** `coastal_pinn/sources/wam.py` (with backward-compat re-export from `coastal_pinn/sources/wave_intensity.py`)

---

### Source 4: Bathymetry — GEBCO 2026

| Attribute | Value |
|---|---|
| Provider | GEBCO Compilation Group (via NOAA ERDDAP) |
| Product | ETOPO1 / GEBCO blended grid |
| Variable | `z` (elevation relative to MSL, m; positive = above, negative = below) |
| Native resolution | 1 arc-minute (~1.8 km via ERDDAP; 15 arc-sec ~450 m from GEBCO directly) |
| Auth required | None (open-access ERDDAP) |
| Cache format | `.nc` (NetCDF) |

**What it gives us:** The seafloor depth at each transect. This is a static input (bathymetry does not change over the 2018–2025 window). It provides geometric context to the closure sub-network: steep sections of the beach erode differently than flat sections.

**How we process it:**
1. Download the (lat, lon) depth grid for the Keta bbox from NOAA ERDDAP (no auth, multiple mirrors tried sequentially).
2. Interpolate to per-transect (lon, lat) values using `xarray.interp(method='linear')`.
3. Fill any NaN with the regional mean (defensive — should not happen if the bbox is within the data range).
4. Clamp query lon/lat to data range.

**Why GEBCO:**
- The standard global reference for coastal and shelf bathymetry.
- Free, no auth, well-validated against ship soundings.
- 1.8 km resolution via ERDDAP is sufficient for the Keta nearshore (which is uniformly shallow, mostly 0–5 m depth).

**Code location:** `coastal_pinn/sources/bathymetry.py`

---

## 3. The Spatial Framework — Transects

### The One-Line Model geometry

The pipeline uses the standard USGS Digital Shoreline Analysis System (DSAS) convention:

```
              Sea
              ↑
              |  ← S(x_n, t) = cross-shore distance from
              |            inland baseline to shoreline
              |
   ──────────●━━━━━━━━━━━━●━━━━━━━━━━━━●──── Coastline (varies in time)
              \            \            \
               \            \            \  ← transect n-1, n, n+1
                \            \            \
                 \            \            \
                  ●━━━━━━━━━━━━●━━━━━━━━━━━━●  ← Inland baseline (x = 0)
                  x_1          x_n          x_N
                  ↑            ↑            ↑
                  |←—100 m—→|  |←—100 m—→|
```

### The baseline

- **Position:** Just *onshore* (≈150 m inland) of the Atlantic shoreline, parallel to the NE-trending coast.
- **Keta baseline:** A 60-point polyline tracing the OSM Atlantic coastline offset ~150 m onshore, running from ~(0.87°E, 5.78°N) in the SW to ~(1.22°E, 6.12°N) in the NE. It is **derived from data, not hand-placed** — see `scripts/derive_keta_baseline.py`, which writes `data/keta_baseline.json` (the single source of truth shared by `config/keta.yaml` and `REGIONS`).
- **Convention:** x = 0 is at the onshore baseline. Transects extend seaward (toward the open Atlantic, ~south). The cross-shore distance S increases as the shoreline moves seaward.
- **Defined in:** `config/keta.yaml` under `region.baseline`; ocean side via `region.ocean_side: south`.

### The transects

- **Count:** ~1,173 transects for the Keta segment (~40 km of coast at 50 m spacing).
- **Spacing:** 50 m along-shore (matching the DSAS setup of Asare-Bediako et al. 2025).
- **Length:** 750 m cross-shore (the onshore baseline sits ~150 m from the shore; 750 m gives margin for shoreline excursion — far shorter than a mis-placed baseline would require).
- **Orientation:** Perpendicular to the baseline, pointing seaward (toward the open Atlantic, which lies *south* of this coast). Orientation is set explicitly by `region.ocean_side` rather than a bbox-center heuristic, because the NE-trending coast crosses the bbox-center latitude (where the heuristic fails).
- **Shore-normal direction:** Computed per transect from the local baseline tangent. For this curved (N-point) baseline each transect gets its own shore-normal.
- **Generated by:** `coastal_pinn/sources/transects.py::generate_transects()`

### What each transect provides

Each transect is a line segment in UTM coordinates:

| Field | Description |
|---|---|
| `transect_id` | Integer index, 0 to ~1172 |
| `along_shore_x_m` | Distance from the first baseline endpoint along the baseline (m). This is the spatial coordinate `x` in the PINN. |
| `origin_x, origin_y` | UTM easting/northing of the transect origin (on the baseline) |
| `end_x, end_y` | UTM easting/northing of the seaward end of the transect |
| `shore_normal_deg` | Direction the transect points, in degrees CCW from East. Used to compute the wave angle of incidence θ. |

### Why this convention

The One-Line Model (Pelnard-Considère 1956) — which our shoreline evolution equation is based on — is parameterized this way in the published literature. USGS DSAS, the 2025 eastern-Ghana shoreline study, and most global shoreline-change papers use this exact setup: inland baseline, perpendicular transects at fixed along-shore spacing, cross-shore distance measured from the baseline.

The along-shore coordinate `x` (which transect you're on) is the spatial axis of the model. The cross-shore distance `S(x_n, t)` is the target variable — the shoreline position at that transect at that time.

---

## 4. How Each Source Is Processed

### Source 1: Shoreline (CoastSat → transect intersection)

```
Sentinel-2 / Landsat-8 raw images on Google Earth Engine
    ↓  GEE + SDS_download.retrieve_images
Cropped, cloud-masked images on disk
    ↓  SDS_classify → trained pixel classifier
Pixel-level labels: water, sand, white-water, other
    ↓  SDS_shoreline.extract_shorelines → sub-pixel border
2-D polyline: Nx2 array of (lon, lat) per cloud-free date
    ↓  Our fetcher: output_epsg = cfg.region.epsg (UTM 31N)
2-D polyline in UTM: Nx2 array of (easting, northing)
    ↓  SDS_transects.compute_intersection_QC (per transect)
Cross-shore distance S(x_n, t) for each (transect, date)
    ↓  Flatten to long-format DataFrame
Shoreline DataFrame: region, timestamp, transect_id, along_shore_x_m, cross_shore_S_m, sat
```

**Key design choices:**
- CoastSat's `output_epsg` is set to the region's UTM EPSG so polylines come back in UTM, matching the transect coordinates.
- We use CoastSat's built-in `SDS_transects.compute_intersection_QC` for the intersection (with a shapely-based fallback in `transects.py` if CoastSat's module isn't importable).
- The QC function includes `along_dist=25` median smoothing and multiple-intersection handling.
- Negative intersection distances (shoreline inland of baseline) are clamped to 0.

### Source 2: Sea Level (Copernicus PHY → per-transect interpolation)

```
Copernicus Marine server
    ↓  copernicusmarine.subset(bbox, dates, zos/uo/vo)
.nc file: data(time, lat, lon) for zos, uo, vo
    ↓  Two-stage download if spanning 2022-06-01 cutoff
Single merged .nc along time dimension
    ↓  xarray.interp(longitude=transect_lons, latitude=baseline_lat)
Per-transect values: h(x_n, t), u_east(x_n, t), u_north(x_n, t)
    ↓  Compute u_mag = sqrt(uo² + vo²)
    ↓  Resample to daily means (groupby transect_id + resample D)
Sea level DataFrame: region, timestamp, transect_id, h_m, u_east_m_s, u_north_m_s
```

**Key design choices:**
- **No spatial averaging.** The (lat, lon) cube is interpolated to each transect's exact (lon, lat). This preserves the along-shore gradient `∂h/∂x` that the PDE term needs.
- **Seaward sample point** is used for every source (`core/coords.py::transect_sample_points`): ocean fields and seafloor depth are read at each transect's offshore end, in open water, not at the onshore baseline origin (which lands on/behind the beach, where ocean products are land-masked and GEBCO returns land elevation).
- **Query clamping:** `clamp_query_to_data_range()` clips query lon/lat to the data's coordinate range, preventing NaN at boundary transects.

### Source 3: Waves (Copernicus WAM → per-transect interpolation)

```
Copernicus Marine server
    ↓  copernicusmarine.subset(bbox, dates, VHM0/VMDR)
.nc file: data(time, lat, lon) for VHM0, VMDR (3-hourly)
    ↓  Two-stage download if spanning 2022-06-01 cutoff
    ↓  xarray.interp to per-transect (lon, lat)
Per-transect values: W(x_n, t), W_dir(x_n, t)
    ↓  Daily resample: linear mean for W, circular mean for W_dir
Wave DataFrame: region, timestamp, transect_id, W_m, W_dir_deg
```

**Key design choices:**
- **Circular mean for direction.** Averaging sin and cos separately, then `arctan2`, correctly handles the wrap-around at 0°/360°.
- **3-hourly → daily.** The cube is 3-hourly (8 substeps per day). We resample to daily means using pandas `groupby + agg` for robustness against uneven substeps and partial days.

### Source 4: Bathymetry (GEBCO → per-transect interpolation)

```
NOAA ERDDAP server
    ↓  HTTP GET with subset URL (no auth)
.nc file: z(lat, lon) grid
    ↓  xarray.interp to per-transect (lon, lat)
Per-transect depth: depth(x_n)
    ↓  Fill NaN with regional mean (defensive)
Bathymetry DataFrame: region, transect_id, depth_m
```

**Key design choices:**
- **Per-transect depth, not a scalar.** The v1 pipeline used a single `depth_at_shore_m` scalar for the whole region. The v2 pipeline interpolates to each transect, giving a 1-D depth profile `depth(x)` along the coast. This lets the closure sub-network learn that steep sections erode differently than flat sections.
- **Static.** Bathymetry has no time dimension — the same depth value is attached to every date for a given transect.

---

## 5. The Reconciliation Step

The reconcile step (`coastal_pinn/pipeline.py::reconcile()`) is the single contract that turns four source-specific DataFrames into one clean wide table.

### What it does, step by step

1. **Normalize timestamps to UTC tz-aware.** Every source's `timestamp` column is forced through `ensure_utc()`. This is the hard contract — `merge_asof` will fail if any are naive or mixed-resolution.

2. **Per-transect as-of joins.** For each transect, find the nearest sea-level and wave reading within the join tolerance (default 36 hours). This is done via `_per_transect_asof()` which groups by `transect_id` and runs `merge_asof` within each group — more robust than a global `merge_asof` with `by=` when the shoreline is sparse (only some transects have observations on a given date) and the forcing is dense (all transects have daily data).

3. **Compute current speed magnitude.** `u_mag_m_s = sqrt(u_east_m_s² + u_north_m_s²)`.

4. **Compute W_longshore (the CERC longshore factor).**
   - For each merged row, look up the transect's `shore_normal_deg` from the transects table.
   - Compute the wave angle of incidence: `θ = (W_dir_deg + 180°) − shore_normal_deg`. The `+180°` converts meteorological convention (direction waves come FROM) to oceanographic convention (direction waves are GOING TO).
   - Compute: `W_longshore = W_m × sin(2θ)`.
   - This is the CERC (USACE 1984) longshore transport factor. `sin(2θ)` is zero when waves hit perpendicular (θ=0° or 180°) and maximum at θ=45°.

5. **Attach per-transect depth.** Map `depth_m` from the bathymetry DataFrame to each row by `transect_id`.

6. **Leave R_sediment_m_yr as NaN.** Sediment recovery is not fetched — it is a model concern (the PINN learns it as a neural closure). The column is reserved so the schema doesn't change when the closure is implemented.

7. **Drop rows with any NaN in required columns.** No fabrication, no interpolation. If a shoreline observation has no matching forcing within 36 hours, the row is dropped with a warning.

8. **Validate schema.** Check that all required columns are present, timestamps are UTC, numeric columns are numeric, and `transect_id` is integer.

9. **Write to disk.** Atomic write of CSV and pickle to `cfg.output_dir`.

### The as-of join explained

The shoreline is sparse: 372 cloud-free dates over 8 years, irregularly spaced. The forcing is dense: daily for sea level and waves. The as-of join finds, for each shoreline observation at each transect, the nearest forcing value within 36 hours.

```
Shoreline date: 2018-01-04 10:29 UTC
    ↓  as-of join with 36h tolerance
Sea level:  2018-01-04 00:00 UTC  (within 36h ✓)
Waves:      2018-01-04 00:00 UTC  (within 36h ✓)
    ↓
Joined row: (transect_id, 2018-01-04, S, h, u, W, W_dir, depth)
```

If no forcing falls within 36 hours of a shoreline date, that row is dropped. The 36-hour tolerance is generous enough to handle the daily cadence of the forcing with margin for model data delays.

---

## 6. The Wide Table Schema

The output of the pipeline is a single pandas DataFrame (written to CSV and pickle) with the following columns:

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `region` | str | Region name (e.g., `keta`) |
| 2 | `timestamp` | datetime, UTC | Observation time |
| 3 | `transect_id` | int | Transect index (0 to ~1172) |
| 4 | `along_shore_x_m` | float | Transect's along-shore position from baseline origin (m) |
| 5 | `cross_shore_S_m` | float | **Target:** observed shoreline cross-shore distance from inland baseline (m) |
| 6 | `h_m` | float | Sea level at this transect and time (m) |
| 7 | `u_mag_m_s` | float | Current speed magnitude at this transect and time (m/s) |
| 8 | `W_m` | float | Significant wave height at this transect and time (m) |
| 9 | `W_dir_deg` | float | Mean wave direction (meteorological, 0–360°) |
| 10 | `W_longshore` | float | **Derived:** `W_m × sin(2θ)` — CERC longshore transport factor |
| 11 | `depth_m` | float | GEBCO depth at this transect (m; static, same for all dates) |
| 12 | `R_sediment_m_yr` | float | **Placeholder:** NaN at fetch time; learned by the PINN closure |

### Required vs. optional columns

- **Required (must be non-NaN):** columns 1–11. Any row with NaN in these is dropped.
- **Optional (allowed to be NaN):** column 12 (`R_sediment_m_yr`). This is the sediment recovery placeholder — the network learns it; the data does not provide it.

### Schema validation

The schema is enforced by `coastal_pinn/core/schema.py::validate_schema()`, which checks:
- All required columns are present.
- `timestamp` is tz-aware UTC.
- `region` is string-like.
- Numeric columns are numeric.
- `transect_id` is integer.

The schema is the **single source of truth** — defined in `PINN_COLUMNS` in `schema.py` and referenced by the pipeline, the CLI, and the tests.

---

## 7. Mapping Data to the Model Equations

The model has two equations and one learned closure. Here is the exact mapping from data columns to equation terms.

### Equation 1: The shallow-water PDE (physics constraint)

$$\frac{\partial h}{\partial t} + u \frac{\partial h}{\partial x} = 0$$

| Symbol | Physical meaning | Data column | Source |
|---|---|---|---|
| `h` | Water height (m) | `h_m` | Copernicus PHY (`zos`) |
| `u` | Current velocity (m/s) | `u_mag_m_s` | Copernicus PHY (`sqrt(uo²+vo²)`) |
| `x` | Along-shore position (m) | `along_shore_x_m` | Derived from transect geometry |
| `t` | Time (days) | `timestamp` | Observation timestamp |

**Role in the model:** This is a **physics constraint** in the PINN loss. The network's predicted `ĥ` and `û` should obey this equation at every collocation point. The spatial gradient `∂h/∂x` is computed by autograd through the network, but it is only meaningful if the data has along-shore variation in `h` — which is why we interpolate to per-transect values instead of spatially averaging.

### Equation 2: The shoreline evolution ODE (data-fitted)

$$\frac{\partial S}{\partial t} = \alpha W \sin(2\theta) - \beta R - \gamma \frac{\partial^2 S}{\partial x^2}$$

| Symbol | Physical meaning | Data column | Source |
|---|---|---|---|
| `S` | Shoreline position (m) | `cross_shore_S_m` | CoastSat (transect intersection) |
| `W` | Wave height (m) | `W_m` | Copernicus WAM (`VHM0`) |
| `θ` | Wave angle relative to shore-normal | Computed from `W_dir_deg` and `shore_normal_deg` | Copernicus WAM (`VMDR`) + transect geometry |
| `W·sin(2θ)` | CERC longshore transport factor | `W_longshore` | Derived in reconcile step |
| `R` | Sediment recovery (m/yr) | `R_sediment_m_yr` (NaN) | **Learned by PINN closure** |
| `α` | Erosion coefficient | — | Learnable parameter (sigmoid-bounded [0, 1]) |
| `β` | Recovery coefficient | — | Learnable parameter (sigmoid-bounded [0, 0.5]) |
| `γ` | Alongshore diffusion coefficient | — | Learnable parameter (bounded [10², 10⁴]) |
| `x` | Along-shore position (m) | `along_shore_x_m` | Transect geometry |
| `t` | Time (days) | `timestamp` | Observation timestamp |

**Role in the model:** This is the **data-fitted** equation. The observed `S(t)` supervises the model. The `∂²S/∂x²` diffusion term is a finite difference across adjacent transects — this is the standard One-Line Model (Pelnard-Considère 1956) alongshore sediment transport smoothing.

### The neural closure for R

$$R_\theta(h, u, W, \text{depth}, t) \geq 0$$

| Closure input | Data column | Source |
|---|---|---|
| `h` | `h_m` | Copernicus PHY |
| `u` | `u_mag_m_s` | Copernicus PHY |
| `W` | `W_m` | Copernicus WAM |
| `depth` | `depth_m` | GEBCO |
| `t` | `timestamp` | Observation timestamp |

**Role in the model:** Sediment recovery `R` is a real physical process (longshore transport, riverine input, beach-face recovery between storms) but is not directly observable in our data. Rather than fabricate it from a proxy, the PINN learns a function from forcing variables to recovery rate. The non-negativity constraint (`R ≥ 0`) is enforced by a `softplus` activation in the closure sub-network.

### The PINN loss function

$$\mathcal{L} = \lambda_{\text{PDE}} r_{\text{PDE}}^2 + \lambda_{\text{data}} r_{\text{data}}^2 + \lambda_{\text{BC}} r_{\text{BC}}^2 + \lambda_{\text{ODE}} r_{\text{ODE}}^2 + \lambda_{\text{nonneg}} \max(0, -R)^2$$

| Loss term | What it penalizes | Data it uses |
|---|---|---|
| `r_PDE` | Violation of `∂h/∂t + u·∂h/∂x = 0` | `h_m`, `u_mag_m_s`, `along_shore_x_m`, `timestamp` |
| `r_data` | `Ŝ(x_n, t_i) − S_obs(x_n, t_i)` | `cross_shore_S_m`, `along_shore_x_m`, `timestamp` |
| `r_BC` | Boundary condition (zero-gradient at spatial ends) | `along_shore_x_m` at transect 0 and ~1172 |
| `r_ODE` | Violation of `∂S/∂t − αW·sin(2θ) + βR + γ·∂²S/∂x² = 0` | `cross_shore_S_m`, `W_longshore`, `R_sediment_m_yr` (learned), `along_shore_x_m`, `timestamp` |
| `max(0, −R)` | Negative sediment recovery (physically forbidden) | `R_sediment_m_yr` (learned) |

---

## 8. Gap Handling Policy

### The four kinds of NaN

| Kind | Physical meaning | How we handle it |
|---|---|---|
| **Time-of-observation** | A shoreline date has no matching forcing within 36 h | **Drop the row.** No fabrication. |
| **Structural** (`R_sediment_m_yr`) | Sediment recovery is not observed | **Leave as NaN.** The PINN learns R as a closure. |
| **Source failure** | A whole source is unavailable (auth expired, ERDDAP timeout) | **Fail loud** at the fetch step with a clear error message. |
| **Out-of-bounds** | Transect lon/lat is just outside the data grid (floating-point drift) | **Clamp** query lon/lat to the data's coordinate range. |

### Why we drop rather than interpolate

For a preprint that claims "sparse-data framework" and "no fabrication" as part of its contribution, dropping rows with missing forcing is the honest choice. The pipeline reports exactly how many rows survived the gap policy and how many were dropped, giving a clear audit trail.

### The no-fabrication principle

The pipeline never:
- Interpolates forcing across gaps.
- Fabricates shoreline positions for cloud-blocked dates.
- Fills missing values with climatological means.
- Imputes sediment recovery from a proxy.

Every value in the wide table is either directly observed, directly interpolated from a gridded product to a transect location, or derived from observed values (like `u_mag` or `W_longshore`). The only exception is `R_sediment_m_yr`, which is intentionally NaN — the model's job is to learn it.

---

## 9. Data Volume and Structure

### Per-source data volumes (Keta, 2018–2025)

| Source | Native cadence | Per-transect rows | Total rows (before join) |
|---|---|---|---|
| Shoreline (CoastSat) | ~372 cloud-free dates (irregular) | Up to ~1,173 per date | intersection attempts ∝ dates × transects |
| Sea level (PHY) | Daily, 2922 days | ~1,173 per day | ~3.4 million |
| Waves (WAM) | Daily (from 3-hourly), 2922 days | ~1,173 per day | ~3.4 million |
| Bathymetry (GEBCO) | Static | ~1,173 (one per transect) | ~1,173 |

### Wide table volume

After the as-of join with 36 h tolerance and dropping NaN rows:

| Metric | Value |
|---|---|
| Transects | ~1,173 (50 m spacing over ~40 km) |
| Cloud-free dates | ~372 |
| Maximum possible rows | ~1,173 × 372 ≈ 436,000 |
| Expected valid rows (after drops) | scales with the intersection rate (see the low-intersection guard) |
| Columns | 12 |
| Typical CSV size | ~20–30 MB |

### The train/val/test split

Time-based split (no leakage):

| Split | Time range | Approx. rows |
|---|---|---|
| Train | 2018-01-01 to 2023-12-31 | ~130,000 (6 years × ~280 obs/yr × ~1,173 transects) |
| Validation | 2024-01-01 to 2024-12-31 | ~30,000 (1 year × ~50 obs × ~1,173 transects) |
| Test | 2025-01-01 to 2025-12-31 | ~30,000 (1 year × ~50 obs × ~1,173 transects) |

Normalization statistics (mean, std) are computed on the **training set only** and applied to val and test — this is the only correct way to prevent temporal leakage.

---

## 10. Reproducibility and Credentials

### One-time setup per machine

```powershell
# For Copernicus Marine (sea level + currents + waves):
copernicusmarine login

# For Google Earth Engine (shoreline imagery):
earthengine authenticate
```

No auth needed for GEBCO (NOAA ERDDAP is open-access).

### What anyone needs to reproduce the wide table

1. Clone this repository.
2. Install the package: `pip install -e ".[dev]"`.
3. Clone CoastSat: `git clone https://github.com/kvos/CoastSat.git` (place at `./CoastSat/`).
4. Run `copernicusmarine login` and `earthengine authenticate`.
5. Run: `python -m coastal_pinn run --config config/keta.yaml`.

The append-only cache means re-running with the same time window reuses cached files — no re-downloads. Anyone with the credentials and the code gets the exact same wide table.

### The cache is append-only

- Each fetch writes a new file named with the time window.
- Re-running with the same window reuses the file.
- There is no `--force-refresh` flag by design — you never lose data and never re-download.
- The cache directory is `coastal_research_ashesi/<region>/{data,download}/`.

### Multi-year download (reanalysis + analysis)

Copernicus Marine reanalysis products cover up to ~2022-06-01; analysis/forecast products cover from ~2022 onward. For time windows that span this cutoff (e.g., 2018–2025), the pipeline automatically:
1. Downloads from the reanalysis product for `[start, 2022-06-01]`.
2. Downloads from the analysis product for `[2022-06-01, end]`.
3. Merges both along the time dimension, deduplicating any overlapping timestamps.

No user action is needed — the two-stage download is automatic.

---

## 11. File Layout

### Source modules

```
coastal_pinn/sources/
├── transects.py          Generate N perpendicular transects from baseline
├── shoreline.py          GEE + CoastSat → per-transect intersection
├── sea_level.py          Copernicus PHY → per-transect h, u_east, u_north
├── wam.py                Copernicus WAM → per-transect W, W_dir
├── wave_intensity.py     Re-export from wam.py (backward compat)
├── bathymetry.py         GEBCO → per-transect depth
└── sediment_recovery.py  NotImplementedError placeholder (R is learned)
```

### Core modules

```
coastal_pinn/core/
├── schema.py             PINN_COLUMNS, validate_schema(), ensure_utc()
├── coords.py             UTM conversions, transect utilities, field interpolation,
│                         clamp_query_to_data_range(), compute_wave_longshore()
├── io.py                 Atomic CSV/pickle writes
└── paths.py              Append-only cache directory structure
```

### Pipeline and config

```
coastal_pinn/
├── config.py             Region (baseline, spacing, length), PipelineConfig, load_config
├── pipeline.py           run_region(), reconcile(), _per_transect_asof()
└── cli.py                CLI: run, fetch, build, concat, validate, init-config
```

### Cache structure

```
coastal_research_ashesi/
└── keta/
    ├── data/
    │   ├── bathymetry/gebco_2026_<window>.nc
    │   ├── sea_level/cmems_phy_<window>.nc
    │   ├── wave_intensity/cmems_wam_<window>.nc
    │   ├── shoreline/gee_<window>.pkl
    │   └── pinn_wide/pinn_wide_<window>.csv
    └── download/         # figures, summaries, all CSVs
```

### Tests

```
tests/
├── conftest.py           Real-shape NetCDF/pickle fixtures (generated on the fly)
├── test_bathymetry.py    Per-transect depth extraction
├── test_sea_level.py     Per-transect h, u interpolation + depth collapse
├── test_wave_intensity.py  Per-transect W, W_dir + circular mean
├── test_shoreline.py     Per-transect intersection
├── test_pipeline.py      Full reconcile pipeline
├── test_schema.py        Schema validation
├── test_config.py        Config loading
├── test_coords.py        UTM conversions, depth_at_shore (deprecated)
└── test_cli.py           CLI surface
```

---

## Summary — One Paragraph

We ingest four open-access data sources — GEBCO bathymetry (NOAA ERDDAP, no auth), Copernicus Marine PHY sea level and currents (`zos`, `uo`, `vo`, daily, 9 km), Copernicus Marine WAM wave forcing (`VHM0`, `VMDR`, 3-hourly, 9 km), and Google Earth Engine + CoastSat shorelines (Sentinel-2 + Landsat-8, sub-pixel, 372 cloud-free dates over 2018–2025) — cache them append-only, and reconcile them into a single per-(transect, date) wide table with a strict UTC-tz-aware 12-column schema. The spatial framework is a standard USGS DSAS One-Line Model setup, anchored to the OSM Atlantic coastline (not the inland Keta Lagoon): an onshore baseline (~150 m inland) following the NE-trending coast, ~1,173 perpendicular transects at 50 m along-shore spacing, each 750 m long, pointing seaward toward the open Atlantic. All spatial sources are interpolated to each transect's seaward sample point (in open water) — no spatial averaging — preserving along-shore variation while keeping queries off land-masked cells. The wave angle of incidence θ is computed from the wave direction and the local shore-normal, yielding the CERC longshore transport factor `W·sin(2θ)` as a derived column. Sparse temporal gaps are handled by dropping rows where forcing is missing within a 36-hour tolerance — no fabrication, no interpolation. The result is a ~150,000-row wide table covering 2018–2025 at Keta, ready to be transformed into model-ready arrays by the dataset loader and consumed by the PINN.
