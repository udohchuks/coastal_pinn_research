# coastal_pinn

Multi-region coastal-erosion data pipeline that fetches open-access satellite,
oceanographic, and bathymetric datasets and reconciles them into a single
wide-format table for downstream PINN (Physics-Informed Neural Network)
training.

Built for the Ghana coastline (Keta region) per the research framework in
*Coastal_Erosion_project.pdf*; supports the full Gulf of Guinea by adding
new regions to `coastal_pinn/config.py::REGIONS`.

## Data sources

| Source             | Provider                | Variables                               |
|--------------------|-------------------------|-----------------------------------------|
| Bathymetry         | GEBCO 2026 (NOAA ERDDAP)| depth, zone (sea/intertidal/land)       |
| Sea level + currents | Copernicus Marine      | `zos` (sea level), `uo`/`vo` (currents) |
| Wave intensity     | NOAA WAVEWATCH III      | significant wave height, mean direction |
| Shorelines         | Google Earth Engine + CoastSat | per-date (lon, lat) polylines   |

Sediment recovery `R` (paper Eq. 2: `dS/dt = αW − βR`) is intentionally
**not** fetched in v1 — it is a model concern (PINN learns `β`). The wide
table reserves an `R_sediment_m_yr` column for future implementation.

## Wide table schema

Each row is one observation timestep for one region:

```
region            (str)              'keta', 'abidjan', ...
timestamp         (Timestamp, UTC)   observation time
easting_m         (float)            UTM Easting (m)
northing_m        (float)            UTM Northing (m)
h_m               (float)            sea level anomaly (m)
u_mag_m_s         (float)            current speed magnitude (m/s)
W_m               (float)            significant wave height (m)
W_dir_deg         (float)            mean wave direction (deg, 0-360)
depth_at_shore_m  (float)            local bathymetric depth (m)
R_sediment_m_yr   (float)            placeholder, NaN until implemented
```

All timestamps are **UTC-localized (tz-aware)**. This is enforced at every
fetch boundary to avoid `merge_asof` errors when joining sources.

## Install

Requires Python 3.11+.

```powershell
# Install the package in editable mode (in the conda base environment)
pip install -e ".[dev]"
```

This installs `coastal_pinn` and exposes a `coastal_pinn` CLI on your PATH.

## One-time credentials setup

The pipeline needs Copernicus Marine credentials for `h` and `u`.

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

For Google Earth Engine (used by `coastline.py`):

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
python -m coastal_pinn run --config config/keta.yaml --time-end 2018-06-30

# Multi-region (concatenated output)
python -m coastal_pinn run --config config/keta.yaml --config config/abidjan.yaml

# Single source for debugging
python -m coastal_pinn fetch bathymetry    --config config/keta.yaml
python -m coastal_pinn fetch sea-level     --config config/keta.yaml
python -m coastal_pinn fetch wave-intensity --config config/keta.yaml
python -m coastal_pinn fetch shoreline     --config config/keta.yaml

# Build the wide table from cache only (no network)
python -m coastal_pinn build --config config/keta.yaml

# Concatenate cached per-region wide tables
python -m coastal_pinn concat --config config/keta.yaml --config config/abidjan.yaml `
                              --out coastal_research_ashesi/multiregion.csv

# Validate a wide table against the schema
python -m coastal_pinn validate --input coastal_research_ashesi/keta/download/07_pinn_input_wide.csv

# Generate a default config for a region
python -m coastal_pinn init-config --region keta --out config/keta.yaml
```

## Cache layout

Append-only cache under `coastal_research_ashesi/`:

```
coastal_research_ashesi/
└── keta/
    ├── data/
    │   ├── bathymetry/gebco_2026_2018-01-01_to_2018-12-31.nc
    │   ├── sea_level/cmems_2018-01-01_to_2018-12-31.nc
    │   ├── wave_intensity/noaa_ww3_2018-01-01_to_2018-12-31.nc
    │   ├── shoreline/gee_2018-01-01_to_2018-12-31.pkl
    │   └── pinn_wide/07_pinn_input_wide_2018-01-01_to_2018-12-31.csv
    └── download/                                # figures, summaries, all CSVs
```

Each fetch writes a new file named with the time window. Re-running with the
same window reuses the file. There is no `--force-refresh` flag by design —
the cache is append-only so you never lose data and never re-download.

## Tests

```powershell
pytest
```

Tests use real-data fixtures shipped in `tests/fixtures/` (~kilobytes each).
No synthetic data anywhere in the package.

## Offline smoke test

The smoke test in `tests/run_real_smoke.py` builds real-shape NetCDFs and a
CoastSat-style shoreline pickle and runs the full reconciliation pipeline
end-to-end without touching the network, GEE, or Copernicus. Use it to
verify the pipeline on a machine with no internet access:

```powershell
python tests/run_real_smoke.py
```

This writes the complete Keta 2018 PINN wide table to
`coastal_research_ashesi/keta/download/pinn_wide_2018-01-01_to_2018-12-31.csv`
and validates it via `coastal_pinn validate`.

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
│   │   ├── coords.py        UTM / lon-lat / cross-shore
│   │   ├── io.py            atomic CSV/pickle, schema check
│   │   └── schema.py        PINN_COLUMNS, validate(), require_columns()
│   ├── sources/
│   │   ├── bathymetry.py    GEBCO 2026 via NOAA ERDDAP
│   │   ├── sea_level.py     Copernicus Marine (zos, uo, vo)
│   │   ├── wave_intensity.py  NOAA WAVEWATCH III via ERDDAP
│   │   ├── shoreline.py     GEE + CoastSat (UTM output preserved)
│   │   └── sediment_recovery.py  NotImplementedError placeholder
│   ├── pipeline.py          run_region, run, reconcile
│   └── cli.py               argparse + YAML loader
├── scripts/
│   └── run_pipeline.py
├── config/
│   └── keta.yaml            example: Keta 2018
└── tests/
    ├── conftest.py
    ├── fixtures/            tiny real-data slices (~kB each)
    ├── test_bathymetry.py
    ├── test_sea_level.py
    ├── test_wave_intensity.py
    ├── test_shoreline.py
    ├── test_pipeline.py
    └── test_cli.py
```