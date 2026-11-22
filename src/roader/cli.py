"""Roader CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click

from roader import __version__
from roader.legacy.config import load_pilot_config
from roader.legacy.pipeline import run_pilot, scene_summary


@click.group()
@click.version_option(__version__, prog_name="roader")
def main() -> None:
    """Road object reconstruction from LiDAR and NAIP (R&D)."""


@main.command("pilot")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to pilot TOML config",
)
def pilot_cmd(config_path: Path) -> None:
    """Legacy pre-spec pilot (roader.legacy). Replaced in milestone 1."""
    cfg = load_pilot_config(config_path)
    click.echo(f"Roader pilot {cfg.name}")
    click.echo(f"  lidar:  {cfg.lidar_path}")
    click.echo(f"  output: {cfg.output_gpkg}")
    scene = run_pilot(cfg)
    summary = scene_summary(scene)
    click.echo(json.dumps(summary, indent=2))
    click.echo(f"Wrote {cfg.output_gpkg}")


@main.command("doctor")
def doctor_cmd() -> None:
    """Print environment / import checks."""
    mods = [
        "numpy",
        "scipy",
        "shapely",
        "pyproj",
        "geopandas",
        "rasterio",
        "laspy",
        "osmnx",
        "networkx",
        "click",
    ]
    for name in mods:
        try:
            __import__(name)
            click.echo(f"OK  {name}")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"MISS {name}: {exc}")
    try:
        import pdal  # noqa: F401

        click.echo("OK  pdal (optional)")
    except Exception:
        click.echo("SKIP pdal (optional extra)")


if __name__ == "__main__":
    main()
