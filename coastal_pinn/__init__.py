"""coastal_pinn — multi-region coastal-erosion data pipeline.

Fetches open-access satellite, oceanographic, and bathymetric datasets and
reconciles them into a single wide-format table for downstream PINN training.

Modules:
    config      PipelineConfig, Region, load_config
    exceptions  SourceUnavailable, ConfigError, MissingCredentials, SchemaError
    core        paths, coords, io, schema
    sources     one module per data source (bathymetry, sea_level, wave_intensity,
                shoreline)
    pipeline    run_region, run, reconcile
    cli        argparse entry point
"""

from coastal_pinn.config import (
    REGIONS,
    Region,
    PipelineConfig,
    init_config_yaml,
    load_config,
    merge_overrides,
)
from coastal_pinn.exceptions import (
    CoastalPINNError,
    ConfigError,
    MissingCredentials,
    SchemaError,
    SourceUnavailable,
)
from coastal_pinn.pipeline import (
    build_from_cache,
    reconcile,
    run,
    run_region,
)
from coastal_pinn.sources.bathymetry import fetch_bathymetry
from coastal_pinn.sources.sea_level import fetch_sea_level
from coastal_pinn.sources.shoreline import fetch_shorelines
from coastal_pinn.sources.wave_intensity import fetch_wave_intensity

__version__ = "0.1.0"
__all__ = [
    "REGIONS",
    "Region",
    "PipelineConfig",
    "load_config",
    "init_config_yaml",
    "merge_overrides",
    "run",
    "run_region",
    "build_from_cache",
    "reconcile",
    "fetch_bathymetry",
    "fetch_sea_level",
    "fetch_wave_intensity",
    "fetch_shorelines",
    "SourceUnavailable",
    "ConfigError",
    "MissingCredentials",
    "SchemaError",
    "CoastalPINNError",
]