"""CLI entry point for coastal_pinn.

Subcommands:
    run           full pipeline (one or more regions)
    fetch         fetch a single source
    build         build the wide table from cache only (no network)
    concat        concatenate cached per-region wide tables
    validate      validate a wide table against the schema
    init-config   generate a default config YAML for a region
    list-regions  list registered regions

Configuration precedence (highest first):
    CLI args  >  YAML config  >  package defaults

Credentials for Copernicus Marine are read ONLY from:
    ~/.config/copernicusmarine/credentials.json
or:
    env vars COPERNICUS_USER, COPERNICUS_PASSWORD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from coastal_pinn.config import (
    REGIONS,
    PipelineConfig,
    init_config_yaml,
    load_config,
    merge_overrides,
)
from coastal_pinn.exceptions import CoastalPINNError


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coastal_pinn",
        description="Multi-region coastal-erosion data pipeline for PINN training.",
    )
    sub = p.add_subparsers(dest="cmd")

    # --- run ---
    p_run = sub.add_parser("run", help="run the full pipeline (one or more regions)")
    p_run.add_argument("--config", action="append", required=True,
                       help="path to YAML config (repeatable for multi-region)")
    p_run.add_argument("--time-start", help="override config time.start")
    p_run.add_argument("--time-end",   help="override config time.end")
    p_run.add_argument("--offline", action="store_true",
                       help="fail fast on cache miss instead of fetching from network")
    p_run.add_argument("--continue-on-error", action="store_true",
                       help="continue running other regions when one fails")

    # --- fetch ---
    p_fetch = sub.add_parser("fetch", help="fetch a single source")
    p_fetch.add_argument("source", choices=["bathymetry", "sea_level",
                                            "wave_intensity", "shoreline"])
    p_fetch.add_argument("--config", required=True, help="path to YAML config")
    p_fetch.add_argument("--time-start")
    p_fetch.add_argument("--time-end")

    # --- build ---
    p_build = sub.add_parser("build", help="build the wide table from cache only")
    p_build.add_argument("--config", action="append", required=True,
                         help="path to YAML config (repeatable for multi-region)")
    p_build.add_argument("--time-start")
    p_build.add_argument("--time-end")
    p_build.add_argument("--out",
                         help="path for the concatenated CSV (multi-region)")

    # --- concat ---
    p_concat = sub.add_parser("concat",
        help="concatenate per-region wide tables into one CSV")
    p_concat.add_argument("--config", action="append", required=True)
    p_concat.add_argument("--out", required=True)

    # --- validate ---
    p_val = sub.add_parser("validate",
        help="validate a wide table against the schema")
    p_val.add_argument("--input", required=True, help="path to CSV")

    # --- init-config ---
    p_init = sub.add_parser("init-config",
        help="generate a default config YAML for a region")
    p_init.add_argument("--region", required=True)
    p_init.add_argument("--out", required=True)

    # --- list-regions ---
    sub.add_parser("list-regions", help="list registered regions")

    return p


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from coastal_pinn.pipeline import run
    configs = _load_configs(args.config,
                            time_start=args.time_start, time_end=args.time_end)
    wide = run(configs, offline=args.offline,
               continue_on_error=args.continue_on_error)
    print(f"[OK] concatenated wide table: {len(wide)} rows, "
          f"{wide['region'].nunique()} region(s)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg, = _load_configs([args.config],
                          time_start=args.time_start, time_end=args.time_end)
    name = args.source
    if name == "bathymetry":
        from coastal_pinn.sources.bathymetry import fetch_bathymetry
        df = fetch_bathymetry(cfg)
    elif name == "sea_level":
        from coastal_pinn.sources.sea_level import fetch_sea_level
        df = fetch_sea_level(cfg)
    elif name == "wave_intensity":
        from coastal_pinn.sources.wave_intensity import fetch_wave_intensity
        df = fetch_wave_intensity(cfg)
    elif name == "shoreline":
        from coastal_pinn.sources.shoreline import fetch_shorelines
        df = fetch_shorelines(cfg)
    else:
        print(f"unknown source: {name}", file=sys.stderr)
        return 2
    print(f"[OK] fetched {name} for {cfg.region.name}: {len(df)} rows")
    print(df.head())
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from coastal_pinn.pipeline import build_from_cache
    configs = _load_configs(args.config,
                            time_start=args.time_start, time_end=args.time_end)
    out_csv = Path(args.out) if args.out else None
    wide = build_from_cache(configs, out_csv=out_csv)
    print(f"[OK] built wide table from cache: {len(wide)} rows")
    return 0


def cmd_concat(args: argparse.Namespace) -> int:
    from coastal_pinn.pipeline import build_from_cache
    configs = _load_configs(args.config)
    wide = build_from_cache(configs, out_csv=Path(args.out))
    print(f"[OK] concatenated: {len(wide)} rows -> {args.out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from coastal_pinn.core.io import read_csv
    from coastal_pinn.core.schema import validate_schema, PINN_COLUMNS
    df = read_csv(args.input)
    # parse the timestamp column with tz-awareness
    import pandas as pd
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    validate_schema(df)
    print(f"[OK] {args.input}: {len(df)} rows, columns valid")
    print(f"columns: {list(df.columns)}")
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    if args.region not in REGIONS:
        print(f"unknown region: {args.region}. known: {list(REGIONS)}",
              file=sys.stderr)
        return 2
    init_config_yaml(REGIONS[args.region], args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_list_regions(_: argparse.Namespace) -> int:
    for name, region in REGIONS.items():
        print(f"{name}\t{region.bbox}\t{region.utm_zone}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_configs(paths: list[str], *, time_start: str | None = None,
                  time_end: str | None = None) -> list[PipelineConfig]:
    """Load and CLI-override one or more configs."""
    out: list[PipelineConfig] = []
    for p in paths:
        cfg = load_config(p)
        cfg = merge_overrides(cfg, t_start=time_start, t_end=time_end)
        out.append(cfg)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Pre-check for --help to avoid argparse's SystemExit in our error handler
    if argv is not None and "--help" in argv and not any(
        a for a in argv if a in {"run", "fetch", "build", "concat",
                                   "validate", "init-config", "list-regions"}
    ):
        parser.print_help()
        return 0
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0

    if args.cmd is None:
        parser.print_help()
        return 0

    handler = {
        "run":          cmd_run,
        "fetch":        cmd_fetch,
        "build":        cmd_build,
        "concat":       cmd_concat,
        "validate":     cmd_validate,
        "init-config":  cmd_init_config,
        "list-regions": cmd_list_regions,
    }.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 2

    try:
        return handler(args) or 0
    except CoastalPINNError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"[ERROR] file not found: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())