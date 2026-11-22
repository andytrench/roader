"""Render, gate and publish one unit from its parametric model (stage 12).

``export_model`` renders the meshes from the model (FR-SG), writes the exports twice
into scratch directories from two independent renderings to measure AC-H5, runs every
section 9 gate, and routes the unit through the publish gate (FR-EX-1).
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from roader.export.publish import PUBLISHED, write_unit
from roader.model.schema import RoadModel
from roader.model.serialize import dumps, loads
from roader.solids import generate
from roader.validate.gates import run_gates
from roader.validate.report import GateReport


def load_crossing_reference(path: str | Path) -> list[dict[str, Any]]:
    """AC-H2 reference entries from a crossing list YAML.

    Accepts the site format (``crossings: [{upper: {name}, lower: {name}}]``) or a
    plain list of ``check_crossing_order`` entries.
    """
    data = yaml.safe_load(Path(path).read_text())
    items = data.get("crossings", []) if isinstance(data, Mapping) else data
    out = []
    for item in items or []:
        if "upper" in item and "lower" in item:
            out.append({"upper_name": item["upper"]["name"], "lower_name": item["lower"]["name"]})
        else:
            out.append(dict(item))
    return out


def export_model(
    model: RoadModel,
    out_dir: str | Path,
    *,
    crossings_ref: Sequence[Mapping] | None = None,
    points: Any = None,
    measured: Mapping | None = None,
    val3dity_exe: str | None = None,
    chord_tol_m: float = 0.01,
) -> tuple[GateReport, dict[str, Path]]:
    """Generate meshes, run all gates and write the unit to published/ or held/."""
    meshes = generate(model, chord_tol_m=chord_tol_m)
    with tempfile.TemporaryDirectory() as scratch:
        runs = []
        for name, run_model in (("a", model), ("b", loads(dumps(model)))):
            run_meshes = meshes if name == "a" else generate(run_model, chord_tol_m=chord_tol_m)
            root = Path(scratch) / name
            write_unit(run_model, run_meshes, GateReport(unit_id=model.unit.unit_id), root)
            runs.append(root / PUBLISHED)
        report = run_gates(
            model,
            meshes,
            points=points,
            crossings_ref=crossings_ref,
            measured=measured,
            export_runs=(runs[0], runs[1]),
            val3dity_exe=val3dity_exe,
        )
    paths = write_unit(model, meshes, report, Path(out_dir), points=points)
    return report, paths
