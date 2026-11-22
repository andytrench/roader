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


def _echo_gates(report) -> None:
    for result in sorted(report.results, key=lambda r: r.gate):
        state = (
            "PASS" if result.passed else ("SKIP" if result.skipped_reason else "FAIL")
        )
        note = result.skipped_reason or (
            f"{len(result.violations)} violations" if result.violations else ""
        )
        click.echo(
            f"{state:4} {result.gate}{' (hard)' if result.hard else ''} {note}".rstrip()
        )


@main.command("build")
@click.option(
    "--unit",
    "unit_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Unit config (configs/units/<site>.toml)",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output root; the unit goes to published/ or held/",
)
@click.option(
    "--crossings",
    "crossings_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Site crossing list; default tests/sites/<site>/crossings.yaml",
)
@click.option(
    "--no-points",
    is_flag=True,
    help="Skip the lidar review overlay (points_sample.glb)",
)
def build_cmd(
    unit_path: Path, out_dir: Path, crossings_path: Path | None, no_points: bool
) -> None:
    """Lidar and vectors to a gated, published unit (stages 1 to 12)."""
    from roader.build import build_unit
    from roader.io import load_unit_config
    from roader.validate.val3dity import find_val3dity

    if crossings_path is None:
        site = load_unit_config(unit_path).site_id
        crossings_path = Path("tests") / "sites" / site / "crossings.yaml"
    _model, report, paths = build_unit(
        unit_path,
        out_dir,
        crossings_path=crossings_path,
        val3dity_exe=find_val3dity(),
        points_overlay=not no_points,
    )
    _echo_gates(report)
    click.echo(
        f"{'Published' if report.publishable else 'Held'}: {paths['model'].parent}"
    )


@main.command("export")
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Native model JSON (model.json)",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output root; the unit goes to published/ or held/",
)
@click.option(
    "--crossings",
    "crossings_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="AC-H2 hand-labelled crossing list (YAML)",
)
@click.option(
    "--chord-tol",
    "chord_tol_m",
    type=float,
    default=0.01,
    show_default=True,
    help="Chord tolerance in metres (FR-SG-2)",
)
def export_cmd(
    model_path: Path, out_dir: Path, crossings_path: Path | None, chord_tol_m: float
) -> None:
    """Render, gate and publish one unit from its native model (FR-EX-1, stage 12)."""
    from roader.export.run import export_model, load_crossing_reference
    from roader.model.serialize import loads
    from roader.validate.val3dity import find_val3dity

    model = loads(model_path.read_text())
    reference = load_crossing_reference(crossings_path) if crossings_path else None
    report, paths = export_model(
        model,
        out_dir,
        crossings_ref=reference,
        val3dity_exe=find_val3dity(),
        chord_tol_m=chord_tol_m,
    )
    _echo_gates(report)
    click.echo(
        f"{'Published' if report.publishable else 'Held'}: {paths['model'].parent}"
    )


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
    except Exception:  # noqa: BLE001
        click.echo("SKIP pdal (optional extra)")


if __name__ == "__main__":
    main()
