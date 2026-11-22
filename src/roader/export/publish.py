"""Publish gate and unit writer (FR-EX-1, FR-EX-3, FR-EX-4, AC-H5).

``write_unit`` writes every export of one unit with shared ids. A unit whose hard
gates all pass goes to ``out_dir/published/<unit_id>/``; otherwise it goes to
``out_dir/held/<unit_id>/`` with HELD.json listing the failed gates. Reduced-LOD
publication of failing units arrives with the LOD chain in milestone 2.

The destination is rebuilt on every call, and the unit's copy in the other state
is removed, so a unit that starts failing can never leave a stale published copy.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from roader.export._common import canonical_json, check_meshes
from roader.export.cityjson import write_cityjson
from roader.export.gltf import write_glb
from roader.export.native import write_native
from roader.export.opendrive import write_xodr
from roader.export.report import write_report
from roader.export.usd import write_usda
from roader.model.mesh import Mesh
from roader.model.schema import RoadModel
from roader.validate.report import GateReport

PUBLISHED = "published"
HELD = "held"

FILES = {
    "model": "model.json",
    "glb": "meshes.glb",
    "glb_provenance": "meshes_provenance.glb",
    "usda": "meshes.usda",
    "cityjson": "model.city.json",
    "xodr": "roads.xodr",
}


def _unit_dir_name(unit_id: str) -> str:
    if not unit_id or unit_id in {".", ".."} or "/" in unit_id or "\\" in unit_id:
        raise ValueError(f"unit id {unit_id!r} is not usable as a directory name")
    return unit_id


def held_record(report: GateReport) -> dict[str, Any]:
    failed = [r for r in sorted(report.results, key=lambda r: r.gate) if r.hard and not r.passed]
    return {
        "unit_id": report.unit_id,
        "status": HELD,
        "req": "FR-EX-1",
        "failed_gates": [
            {
                "gate": r.gate,
                "skipped_reason": r.skipped_reason,
                "violation_count": len(r.violations),
                "violations": [
                    {"req": v.req, "element_id": v.element_id, "message": v.message}
                    for v in sorted(r.violations, key=lambda v: (v.req, v.element_id, v.message))[:20]
                ],
            }
            for r in failed
        ],
        "note": "Reduced-LOD publication arrives with the LOD chain in milestone 2.",
    }


def write_unit(
    model: RoadModel,
    meshes: list[Mesh],
    report: GateReport,
    out_dir: Path,
    *,
    points: Any = None,
) -> dict[str, Path]:
    """Write all exports of one unit and route it through the publish gate.

    Returns paths keyed by "model", "glb", "glb_provenance", "usda", "cityjson",
    "xodr", "report_json", "report_md", "figure:<name>", and "held" when held.
    """
    if report.unit_id != model.unit.unit_id:
        raise ValueError(f"gate report is for {report.unit_id}, model is {model.unit.unit_id}")
    meshes = check_meshes(meshes)
    name = _unit_dir_name(model.unit.unit_id)
    out_dir = Path(out_dir)
    state = PUBLISHED if report.publishable else HELD
    dest = out_dir / state / name
    other = out_dir / (HELD if state == PUBLISHED else PUBLISHED) / name
    for stale in (dest, other):
        if stale.is_dir():
            shutil.rmtree(stale)
    dest.mkdir(parents=True)

    paths: dict[str, Path] = {
        "model": write_native(model, dest / FILES["model"]),
        "glb": write_glb(model, meshes, dest / FILES["glb"]),
        "glb_provenance": write_glb(model, meshes, dest / FILES["glb_provenance"], color_by="source"),
        "usda": write_usda(model, meshes, dest / FILES["usda"]),
        "cityjson": write_cityjson(model, meshes, dest / FILES["cityjson"]),
        "xodr": write_xodr(model, dest / FILES["xodr"]),
    }
    paths.update(write_report(model, meshes, report, dest, points=points))
    if state == HELD:
        held = dest / "HELD.json"
        held.write_bytes(canonical_json(held_record(report)))
        paths["held"] = held
    return dict(sorted(paths.items()))
