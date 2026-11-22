"""Per-unit report (FR-EX-4): report.json, report.md and overlay figures.

Contents: inputs and versions (model.unit), coverage from Road.support, model flags,
the gate metric table from the GateReport, triangle counts per mesh, part, road-km
and structure against the FR-AS-2 budgets, and PNG overlays (plan view, profile per
road, fitted surface against points when points are given).

PNGs are drawn with matplotlib's Agg canvas and saved without the Software
metadata key, so the bytes depend only on the inputs and the matplotlib/freetype
build (AC-H5).
"""

from __future__ import annotations

import io
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

import roader
from roader.export._common import (
    canonical_json,
    check_meshes,
    element_index,
    mesh_info,
    rounded,
    safe_name,
)
from roader.model.mesh import Mesh
from roader.model.schema import Road, RoadModel
from roader.model.serialize import model_hash, to_plain
from roader.validate.report import GateReport

# FR-AS-2 finest-LOD budgets.
FREEWAY_TRIANGLES_PER_KM = 150_000
OVERPASS_TRIANGLES = 60_000
OVERRUN_DEFECT = 3.0

OVERLAY_DZ_M = 1.0  # points farther than this from the fitted surface belong to another sheet
OVERLAY_MAX_POINTS = 20_000


def _versions() -> dict[str, str]:
    import matplotlib

    return {
        "model_software_version": "",
        "roader": roader.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }


def coverage(model: RoadModel) -> dict[str, Any]:
    """Observed / occluded / unsupported length per road from Road.support (FR-VC-2, FR-OC-1)."""
    roads = []
    totals = {"length_m": 0.0, "observed_m": 0.0, "occluded_m": 0.0, "unsupported_m": 0.0, "unlisted_m": 0.0}
    for road in sorted(model.roads, key=lambda r: r.id):
        row = {"road_id": road.id, "name": road.name, "length_m": road.length,
               "observed_m": 0.0, "occluded_m": 0.0, "unsupported_m": 0.0}
        for interval in road.support:
            key = f"{interval.kind}_m"
            if key in row:
                row[key] += max(0.0, min(interval.s1, road.length) - max(interval.s0, 0.0))
        row["unlisted_m"] = max(0.0, road.length - row["observed_m"] - row["occluded_m"] - row["unsupported_m"])
        row["observed_fraction"] = row["observed_m"] / road.length if road.length > 0 else 0.0
        for key in totals:
            totals[key] += row[key]
        roads.append({k: rounded(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    totals = {k: rounded(v, 3) for k, v in totals.items()}
    return {"roads": roads, "totals": totals}


def triangle_counts(model: RoadModel, meshes: list[Mesh]) -> dict[str, Any]:
    """FR-AS-2 actuals: per mesh, per part, per road-km and per structure."""
    index = element_index(model)
    per_mesh = {m.id: len(m.faces) for m in meshes}
    per_part: dict[str, int] = {}
    per_road: dict[str, int] = {}
    per_structure: dict[str, int] = {}
    terrain = 0
    for mesh in meshes:
        n = len(mesh.faces)
        per_part[mesh.part] = per_part.get(mesh.part, 0) + n
        info = mesh_info(mesh, index)
        if info is None:
            continue
        if info.kind == "terrain":
            terrain += n
        elif info.structure_id is not None:
            per_structure[info.structure_id] = per_structure.get(info.structure_id, 0) + n
        elif info.road_id is not None:
            per_road[info.road_id] = per_road.get(info.road_id, 0) + n
    roads = []
    for road in sorted(model.roads, key=lambda r: r.id):
        km = road.length / 1000.0
        n = per_road.get(road.id, 0)
        row: dict[str, Any] = {"road_id": road.id, "kind": road.kind, "length_km": rounded(km, 6),
                               "triangles": n, "per_km": rounded(n / km, 1) if km > 0 else None}
        if road.kind == "mainline" and km > 0:
            ratio = n / km / FREEWAY_TRIANGLES_PER_KM
            row.update(budget_per_km=FREEWAY_TRIANGLES_PER_KM, budget_ratio=rounded(ratio, 4),
                       over_3x=ratio > OVERRUN_DEFECT)
        roads.append(row)
    structures = []
    for st in sorted(model.structures, key=lambda s: s.id):
        n = per_structure.get(st.id, 0)
        ratio = n / OVERPASS_TRIANGLES
        structures.append({"structure_id": st.id, "triangles": n, "budget": OVERPASS_TRIANGLES,
                           "budget_ratio": rounded(ratio, 4), "over_3x": ratio > OVERRUN_DEFECT})
    return {
        "total": sum(per_mesh.values()),
        "terrain": terrain,
        "per_mesh": dict(sorted(per_mesh.items())),
        "per_part": dict(sorted(per_part.items())),
        "per_road": roads,
        "per_structure": structures,
        "unmatched_mesh_ids": sorted(m.id for m in meshes if mesh_info(m, index) is None),
    }


def gate_table(gates: GateReport) -> dict[str, Any]:
    rows = []
    for r in sorted(gates.results, key=lambda r: (not r.hard, r.gate)):
        rows.append({
            "gate": r.gate,
            "hard": r.hard,
            "passed": r.passed,
            "skipped_reason": r.skipped_reason,
            "metrics": {k: rounded(v, 6) for k, v in sorted(r.metrics.items())},
            "violation_count": len(r.violations),
            "violations": [to_plain(v) for v in sorted(r.violations, key=lambda v: (v.req, v.element_id, v.message))[:20]],
        })
    return {"unit_id": gates.unit_id, "publishable": gates.publishable, "results": rows}


# ---------------------------------------------------------------- point overlay


def _points_array(points: Any) -> np.ndarray:
    xyz = getattr(points, "xyz", points)
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def road_samples(road: Road, step: float = 0.25) -> dict[str, np.ndarray]:
    count = max(2, math.ceil(road.length / step) + 1)
    s = np.linspace(0.0, road.length, count)
    states = np.array([road.alignment.evaluate(float(v)) for v in s])
    z = np.array([road.profile.z(float(v)) for v in s])
    return {"s": s, "x": states[:, 0], "y": states[:, 1], "h": states[:, 2], "z": z}


def _knots(values: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.array([k.s for k in values.knots]), np.array([k.value for k in values.knots])


def project_points(road: Road, xyz: np.ndarray, samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Station, offset, point z and fitted surface z of points on the road's pavement."""
    from scipy.spatial import cKDTree

    if len(xyz) == 0:
        empty = np.zeros(0)
        return {"s": empty, "u": empty, "z": empty, "fit": empty}
    tree = cKDTree(np.column_stack([samples["x"], samples["y"]]))
    _d, idx = tree.query(xyz[:, :2])
    dx = xyz[:, 0] - samples["x"][idx]
    dy = xyz[:, 1] - samples["y"][idx]
    cos_h, sin_h = np.cos(samples["h"][idx]), np.sin(samples["h"][idx])
    along = dx * cos_h + dy * sin_h
    u = -dx * sin_h + dy * cos_h
    s = samples["s"][idx] + along
    xs = road.cross_section
    wl = np.interp(s, *_knots(xs.width_left))
    wr = np.interp(s, *_knots(xs.width_right))
    inside = (s >= 0.0) & (s <= road.length) & (u <= wl) & (u >= -wr)
    s, u, z = s[inside], u[inside], xyz[inside, 2]
    slope = np.where(u >= 0.0, np.interp(s, *_knots(xs.slope_left)), np.interp(s, *_knots(xs.slope_right)))
    fit = np.interp(s, samples["s"], samples["z"]) + slope * u
    near = np.abs(z - fit) <= OVERLAY_DZ_M
    return {"s": s[near], "u": u[near], "z": z[near], "fit": fit[near]}


# ---------------------------------------------------------------- figures


def _png(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    return buf.getvalue()


def plan_figure(model: RoadModel) -> bytes:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.0, 7.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    for road in sorted(model.roads, key=lambda r: r.id):
        smp = road_samples(road, step=2.0)
        ax.plot(smp["x"], smp["y"], linewidth=1.2, label=road.name or road.id)
    for st in sorted(model.structures, key=lambda s: s.id):
        road = model.road(st.carried_road)
        s = np.linspace(st.s_start, st.s_end, 50)
        pts = np.array([road.alignment.evaluate(float(v))[:2] for v in s])
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=4.0, alpha=0.5)
        for support in sorted(st.supports, key=lambda p: p.id):
            x, y, _h, _k = road.alignment.evaluate(support.station)
            ax.plot([x], [y], marker="s", color="black", markersize=4)
    for c in sorted(model.crossings, key=lambda c: c.id):
        ax.plot([c.x], [c.y], marker="x", color="red", markersize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x east (m, unit frame)")
    ax.set_ylabel("y north (m, unit frame)")
    ax.set_title(f"Plan: {model.unit.unit_id} (structures thick, crossings x)")
    ax.legend(loc="upper right", fontsize=7)
    return _png(fig)


def profile_figure(model: RoadModel, road: Road, overlay: dict[str, np.ndarray] | None) -> bytes:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.0, 4.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    for interval in sorted(road.support, key=lambda i: (i.s0, i.s1, i.kind)):
        if interval.kind != "observed":
            ax.axvspan(interval.s0, interval.s1, color="orange" if interval.kind == "occluded" else "grey",
                       alpha=0.2, linewidth=0)
    for st in sorted(model.structures, key=lambda s: s.id):
        if st.carried_road == road.id:
            ax.axvspan(st.s_start, st.s_end, color="blue", alpha=0.1, linewidth=0)
    if overlay is not None and len(overlay["s"]):
        step = max(1, math.ceil(len(overlay["s"]) / OVERLAY_MAX_POINTS))
        ax.scatter(overlay["s"][::step], overlay["z"][::step], s=1.0, color="grey", alpha=0.5, linewidths=0)
    smp = road_samples(road, step=1.0)
    ax.plot(smp["s"], smp["z"], color="black", linewidth=1.2)
    ax.set_xlabel("station s (m)")
    ax.set_ylabel("elevation z (m)")
    ax.set_title(f"Profile: {road.name or road.id} ({road.id}); occluded orange, structure blue")
    return _png(fig)


# ---------------------------------------------------------------- report


def build_report(model: RoadModel, meshes: list[Mesh], gates: GateReport, *, points: Any = None
                 ) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Report data and figure bytes keyed by relative path."""
    meshes = check_meshes(meshes)
    versions = _versions()
    versions["model_software_version"] = model.unit.software_version
    figures: dict[str, bytes] = {"figures/plan.png": plan_figure(model)}
    overlays: dict[str, Any] = {}
    xyz = _points_array(points) if points is not None else None
    for road in sorted(model.roads, key=lambda r: r.id):
        overlay = None
        if xyz is not None:
            overlay = project_points(road, xyz, road_samples(road))
            resid = overlay["z"] - overlay["fit"]
            overlays[road.id] = {
                "n": len(resid),
                "rms_m": rounded(float(np.sqrt(np.mean(resid**2))), 6) if len(resid) else None,
                "p95_m": rounded(float(np.percentile(np.abs(resid), 95)), 6) if len(resid) else None,
                "mean_m": rounded(float(np.mean(resid)), 6) if len(resid) else None,
            }
        figures[f"figures/profile_{safe_name(road.id)}.png"] = profile_figure(model, road, overlay)
    data = {
        "unit_id": model.unit.unit_id,
        "model_hash": model_hash(model),
        "inputs": to_plain(model.unit),
        "frame": to_plain(model.frame),
        "versions": versions,
        "coverage": coverage(model),
        "flags": [to_plain(f) for f in sorted(model.flags, key=lambda f: (f.code, f.element_id, f.req, f.message))],
        "gates": gate_table(gates),
        "triangles": triangle_counts(model, meshes),
        "point_overlay": {
            "points_given": xyz is not None,
            "max_dz_m": OVERLAY_DZ_M,
            "roads": overlays,
        },
        "figures": sorted(figures),
        "notes": [
            (
                "FR-EX-1: reduced-LOD publication arrives with the LOD chain in milestone 2; "
                "a unit that fails a hard gate is held."
            ),
            "FR-AS-2: the freeway budget is stated for four-lane freeway on grade and is applied to every mainline road.",
        ],
    }
    return data, figures


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "-"
    return str(value)


def report_markdown(data: dict[str, Any]) -> str:
    lines = [f"# Unit report: {data['unit_id']}", "", f"Model hash: `{data['model_hash']}`", ""]
    gates = data["gates"]
    lines += ["## Publish gate (FR-EX-1)", "",
              f"Publishable: **{'yes' if gates['publishable'] else 'no (held)'}**", ""]
    lines += ["| Gate | Hard | Passed | Metrics | Violations | Skipped |", "|---|---|---|---|---|---|"]
    for r in gates["results"]:
        metrics = ", ".join(f"{k}={_fmt(v)}" for k, v in r["metrics"].items()) or "-"
        lines.append(f"| {r['gate']} | {'yes' if r['hard'] else 'no'} | {'yes' if r['passed'] else 'NO'} | "
                     f"{metrics} | {r['violation_count']} | {r['skipped_reason'] or '-'} |")
    inputs = data["inputs"]
    lines += ["", "## Inputs and versions", ""]
    for key, value in sorted(data["versions"].items()):
        lines.append(f"- {key}: {value or '-'}")
    lines.append(f"- AOI (lon/lat W,S,E,N): {', '.join(_fmt(v) for v in inputs['aoi_lonlat'])}")
    lines.append(f"- Imagery enabled: {inputs['imagery_enabled']}")
    lines.append(f"- Frame: `{data['frame']['proj']}`, heights {data['frame']['vertical_datum']} "
                 f"({data['frame']['geoid_model']})")
    for c in inputs["lidar"]:
        lines.append(f"- Lidar {c['name']}: year {_fmt(c['year'])}, {_fmt(c['quality_level'])}, "
                     f"pavement density {_fmt(c['density_pavement_pts_m2'])} pts/m2, {c['point_count']} points")
    for v in inputs["vectors"]:
        lines.append(f"- Vectors {v['name']} ({v['license']}): {'on' if v['enabled'] else 'off'}")
    lines += ["", "## Coverage", "", "| Road | Length m | Observed m | Occluded m | Unsupported m | Observed % |",
              "|---|---|---|---|---|---|"]
    for r in data["coverage"]["roads"]:
        lines.append(f"| {r['name']} (`{r['road_id']}`) | {_fmt(r['length_m'])} | {_fmt(r['observed_m'])} | "
                     f"{_fmt(r['occluded_m'])} | {_fmt(r['unsupported_m'])} | {100 * r['observed_fraction']:.1f} |")
    t = data["coverage"]["totals"]
    lines.append(f"| **total** | {_fmt(t['length_m'])} | {_fmt(t['observed_m'])} | {_fmt(t['occluded_m'])} | "
                 f"{_fmt(t['unsupported_m'])} | |")
    lines += ["", "## Flags", ""]
    if data["flags"]:
        lines += ["| Code | Element | Req | Message |", "|---|---|---|---|"]
        lines += [f"| {f['code']} | `{f['element_id']}` | {f['req']} | {f['message']} |" for f in data["flags"]]
    else:
        lines.append("None.")
    tri = data["triangles"]
    lines += ["", "## Triangles (FR-AS-2 actuals)", "", f"Total {tri['total']}, terrain {tri['terrain']}.", "",
              "| Road | Kind | km | Triangles | Per km | Budget ratio |", "|---|---|---|---|---|---|"]
    for r in tri["per_road"]:
        lines.append(f"| `{r['road_id']}` | {r['kind']} | {_fmt(r['length_km'])} | {r['triangles']} | "
                     f"{_fmt(r['per_km'])} | {_fmt(r.get('budget_ratio'))} |")
    lines += ["", "| Structure | Triangles | Budget ratio | Over 3x |", "|---|---|---|---|"]
    for r in tri["per_structure"]:
        lines.append(f"| `{r['structure_id']}` | {r['triangles']} | {_fmt(r['budget_ratio'])} | {r['over_3x']} |")
    lines += ["", "| Part | Triangles |", "|---|---|"]
    lines += [f"| {part} | {n} |" for part, n in tri["per_part"].items()]
    overlay = data["point_overlay"]
    if overlay["points_given"]:
        lines += ["", f"## Fitted surface against points (within {overlay['max_dz_m']} m)", "",
                  "| Road | Points | RMS m | P95 m | Mean m |", "|---|---|---|---|---|"]
        for road_id, r in sorted(overlay["roads"].items()):
            lines.append(f"| `{road_id}` | {r['n']} | {_fmt(r['rms_m'])} | {_fmt(r['p95_m'])} | {_fmt(r['mean_m'])} |")
    lines += ["", "## Figures", ""]
    lines += [f"![{Path(p).stem}]({p})" for p in data["figures"]]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in data["notes"]]
    return "\n".join(lines) + "\n"


def write_report(model: RoadModel, meshes: list[Mesh], gates: GateReport, out_dir: Path, *, points: Any = None
                 ) -> dict[str, Path]:
    out_dir = Path(out_dir)
    data, figures = build_report(model, meshes, gates, points=points)
    paths: dict[str, Path] = {}
    for rel, content in sorted(figures.items()):
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths[f"figure:{Path(rel).stem}"] = path
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_bytes(canonical_json(data))
    (out_dir / "report.md").write_bytes(report_markdown(data).encode("utf-8"))
    paths["report_json"] = out_dir / "report.json"
    paths["report_md"] = out_dir / "report.md"
    return paths
