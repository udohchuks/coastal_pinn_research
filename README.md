# coastal_pinn

Multi-region coastal-erosion data pipeline that fetches open-access satellite,
oceanographic, and bathymetric datasets and reconciles them into a single
per-transect wide-format table for downstream PINN (Physics-Informed Neural
Network) training.

Built for the Ghana coastline (Keta region) using a cross-shore equilibrium
shoreline model (Yates 2009 + Vitousek 2017) with USGS DSAS conventions
(inland baseline, perpendicular transects at 100 m along-shore spacing).
Supports the full Gulf of Guinea by adding new regions to
`coastal_pinn/config.py::REGIONS`.

## Data sources

| Source             | Provider                | Variables                               |
|--------------------|-------------------------|-----------------------------------------|
| Bathymetry         | GEBCO 2026 (NOAA ERDDAP)| depth per transect (interpolated)       |
| Sea level + currents | Copernicus Marine PHY  | `zos` (sea level), `uo`/`vo` (currents) |
| Wave forcing       | Copernicus Marine WAM   | `VHM0` (wave height), `VMDR` (direction)|
| Shorelines         | Google Earth Engine + CoastSat | per-date (lon, lat) polylines   |

All spatial sources (Copernicus PHY, WAM, GEBCO) are interpolated to the
exact (lon, lat) of each transect origin — no spatial averaging. This
preserves along-shore gradients needed by the shallow-water PDE term.

The shoreline model is the Yates et al. (2009) cross-shore equilibrium ODE
with the Vitousek et al. (2017, CoSMoS-COAST) long-term trend term:

```
dS/dt = C± · √E · (E − E_eq) + v        E = H_s² / 16
```

`E` is the wave energy, derived from the significant wave height `H_s`
(`W_m`) as `E = W_m²/16` and stored in the `E_wave` column. The
coefficients `C±` (accretion/erosion rate), `E_eq` (equilibrium wave
energy), and `v` (long-term linear trend — at Keta this absorbs the slow,
non-wave drift from the Volta/Akosombo sediment cutoff) are **not** fetched:
they are all learned by the PINN. The data provides only `E_wave`.

## Wide table schema

Each row is one observation at one (transect, date) pair:

```
region            (str)              'keta', 'abidjan', ...
timestamp         (Timestamp, UTC)   observation time
transect_id       (int)              0, 1, 2, ..., N-1
along_shore_x_m   (float)            transect's along-shore position (m)
cross_shore_S_m   (float)            shoreline cross-shore distance from baseline (m)
h_m               (float)            sea level at this transect (m)
u_mag_m_s         (float)            current speed magnitude at this transect (m/s)
W_m               (float)            significant wave height at this transect (m)
W_dir_deg         (float)            mean wave direction (deg, 0-360)
E_wave            (float)            DERIVED: wave energy E = W_m**2/16 (Yates 2009)
depth_m           (float)            GEBCO depth at this transect (m)
```

All timestamps are **UTC-localized (tz-aware)**. This is enforced at every
fetch boundary to avoid `merge_asof` errors when joining sources.

## Install

Requires Python 3.11+. We recommend using a dedicated `conda` environment.

```powershell
# Create a new conda environment
conda create -n coastal_pinn python=3.11 -y

# Activate the environment
conda activate coastal_pinn

# Install the package and dependencies in editable mode
pip install -e ".[dev]"

# Clone the CoastSat repository into the project directory
git clone https://github.com/kvos/CoastSat.git
```

This installs `coastal_pinn` and exposes a `coastal_pinn` CLI on your PATH.

## One-time credentials setup

The pipeline needs Copernicus Marine credentials for sea level, currents,
and waves (PHY + WAM products).

**Option A (recommended)** — write to the standard Copernicus config file:

```powershell
copernicusmarine login
# When prompted: enter your Copernicus Marine username and password.
# This creates ~/.config/copernicusmarine/credentials.json
```

**Option B** — environment variables (per-session, never persisted):

```powershell
$env:COPERNICUS_USER = "<your-username>"
$env:COPERNICUS_PASSWORD = "<your-password>"
```

For Google Earth Engine (used by the shoreline source):

```powershell
earthengine authenticate
# Opens a browser; sign in and paste the token back.
```

The pipeline never reads credentials from project files. If neither env vars
nor the Copernicus config file is set, the pipeline exits with a clear error.

## Run

```powershell
# Single region, default config
python -m coastal_pinn run --config config/keta.yaml

# Override time window from CLI
python -m coastal_pinn run --config config/keta.yaml --time-end 2025-12-31

# Multi-region (concatenated output)
python -m coastal_pinn run --config config/keta.yaml --config config/abidjan.yaml

# Single source for debugging
python -m coastal_pinn fetch bathymetry    --config config/keta.yaml
python -m coastal_pinn fetch sea-level     --config config/keta.yaml
python -m coastal_pinn fetch wave-intensity --config config/keta.yaml
python -m coastal_pinn fetch shoreline     --config config/keta.yaml

# Build the wide table from cache only (no network)
python -m coastal_pinn build --config config/keta.yaml

# Validate a wide table against the schema
python -m coastal_pinn validate --input coastal_research_ashesi/keta/download/pinn_wide_*.csv

# Generate a default config for a region
python -m coastal_pinn init-config --region keta --out config/keta.yaml
```

## Cache layout

Append-only cache under `coastal_research_ashesi/`:

```
coastal_research_ashesi/
└── keta/
    ├── data/
    │   ├── bathymetry/gebco_2026_<window>.nc
    │   ├── sea_level/cmems_phy_<window>.nc
    │   ├── wave_intensity/cmems_wam_<window>.nc
    │   ├── shoreline/gee_<window>.pkl
    │   └── pinn_wide/pinn_wide_<window>.csv
    └── download/                                # figures, summaries, all CSVs
```

Each fetch writes a new file named with the time window. Re-running with the
same window reuses the file. There is no `--force-refresh` flag by design —
the cache is append-only so you never lose data and never re-download.

## Multi-year download (reanalysis + analysis)

Copernicus Marine reanalysis products cover up to ~2022-06-01; analysis/
forecast products cover from ~2022 onward. For time windows that span this
cutoff (e.g., 2018–2025), the pipeline automatically downloads from both
products and merges along the time dimension. No user action needed.

## Tests

```powershell
pytest
```

Tests use real-shape fixtures generated in `tests/conftest.py` (~kilobytes
each). No synthetic data anywhere in the package.

## Offline smoke test

The smoke test in `tests/run_real_smoke.py` builds real-shape NetCDFs and a
CoastSat-style shoreline pickle and runs the full reconciliation pipeline
end-to-end without touching the network, GEE, or Copernicus. Use it to
verify the pipeline on a machine with no internet access:

```powershell
python tests/run_real_smoke.py
```

## Project layout

```
coastal_pinn/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
├── coastal_pinn/
│   ├── __init__.py
│   ├── config.py            Region, PipelineConfig, load_config
│   ├── exceptions.py        SourceUnavailable, ConfigError, MissingCredentials, SchemaError
│   ├── core/
│   │   ├── paths.py         append-only artifact dirs
│   │   ├── coords.py        UTM / lon-lat / transect utilities / field interpolation
│   │   ├── io.py            atomic CSV/pickle, schema check
│   │   └── schema.py        PINN_COLUMNS, validate(), require_columns()
│   ├── sources/
│   │   ├── transects.py     generate N perpendicular transects from baseline
│   │   ├── bathymetry.py    GEBCO 2026 via NOAA ERDDAP (per-transect interp)
│   │   ├── sea_level.py     Copernicus Marine PHY (zos, uo, vo) per-transect
│   │   ├── wam.py           Copernicus Marine WAM (VHM0, VMDR) per-transect
│   │   ├── wave_intensity.py  re-export from wam.py (backward compat)
│   │   └── shoreline.py     GEE + CoastSat (N-transect intersection)
│   ├── pipeline.py          run_region, run, reconcile
│   └── cli.py               argparse + YAML loader
├── scripts/
│   └── run_pipeline.py
├── config/
│   └── keta.yaml            example: Keta 2018
├── CoastSat/                cloned upstream CoastSat repo (used as library)
└── tests/
    ├── conftest.py
    ├── test_bathymetry.py
    ├── test_sea_level.py
    ├── test_wave_intensity.py
    ├── test_shoreline.py
    ├── test_pipeline.py
    ├── test_schema.py
    ├── test_config.py
    ├── test_coords.py
    └── test_cli.py
```
