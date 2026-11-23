"""Hard gates AC-H1 to AC-H6 and the per-unit gate report (spec section 9, FR-EX-1).

A hard gate that cannot run (no val3dity, no hand-labelled crossing list, no
second export run) is reported with ``skipped_reason`` and ``passed=False``, so
the unit is not publishable until it does run.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from roader.model.mesh import Mesh
from roader.model.provenance import Provenance, Source
from roader.model.schema import ELEMENT_FAILED, Road, RoadModel
from roader.validate import lod
from roader.validate._align import Projector
from roader.validate._geom import points_strictly_inside, vertical_hits, xy_tree
from roader.validate.accuracy import (
    crossing_dz_accuracy,
    horizontal_accuracy,
    structure_station_accuracy,
    surface_accuracy,
)
from roader.validate.assets import check_budgets, check_transitions, check_uv_stretch
from roader.validate.profile import check_profile, check_seam
from roader.validate.realtime import DEFAULT_MIN_EMBED_M, check_realtime
from roader.validate.report import GateReport, GateResult, Violation
from roader.validate.seams import LINK_SLOPE_TOL, curvature_violations, seam_violations
from roader.validate.solid import check_solid
from roader.validate.val3dity import run_val3dity

# AC-H4 joint tolerances inside one road (elements are fitted to chain exactly).
JOINT_POS_TOL = 1e-3  # m
JOINT_HEADING_TOL = 1e-5  # rad
JOINT_Z_TOL = 1e-3  # m
JOINT_GRADE_TOL = 1e-5
STATION_TOL = 1e-6  # m
# AC-H4 across links and junction seams: LINK_Z_TOL and LINK_SLOPE_TOL in roader.validate.seams.
LINK_GRADE_TOL = LINK_SLOPE_TOL
# Export profile -> AC-H1 gate name (spec section 7 table).
PROFILES = {"analytic": "AC-H1", "realtime": "AC-H1:realtime"}


@dataclass
class Seam:
    """A shared tile boundary for FR-SV-8: the meshes on each side and the seam plane."""

    id: str
    side_a: Sequence[Mesh]
    side_b: Sequence[Mesh]
    origin: Sequence[float]
    normal: Sequence[float]


# ---------------------------------------------------------------- AC-H1


def check_solids(
    meshes: Sequence[Mesh],
    *,
    seams: Iterable[Seam] = (),
    val3dity_exe: str | None = None,
    profile: str = "analytic",
    min_embed_m: float = DEFAULT_MIN_EMBED_M,
) -> GateResult:
    """AC-H1: FR-SV-1..6 per mesh, FR-SV-7 between meshes, FR-SV-8 on seams, plus val3dity.

    ``profile`` selects the FR-SV-7 rule: "analytic" (AC-H1 proper) or "realtime"
    (embed by at least ``min_embed_m``, no coplanar overlap), reported as gate
    ``AC-H1:realtime``.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown export profile {profile!r}; expected one of {sorted(PROFILES)}")
    result = GateResult(PROFILES[profile], passed=True, hard=True)
    if not meshes:
        result.passed = False
        result.skipped_reason = "no meshes to validate"
        return result
    for mesh in meshes:
        result.violations.extend(check_solid(mesh))
    if profile == "realtime":
        result.violations.extend(check_realtime(meshes, min_embed_m))
    else:
        result.violations.extend(check_profile(meshes))
    for seam in seams:
        result.violations.extend(check_seam(seam.side_a, seam.side_b, origin=seam.origin, normal=seam.normal, seam_id=seam.id))
    native = len(result.violations)
    oracle = run_val3dity(meshes, exe=val3dity_exe)
    result.violations.extend(oracle.violations)
    result.skipped_reason = oracle.skipped_reason
    result.metrics = {
        "n_meshes": float(len(meshes)),
        "n_native_violations": float(native),
        "n_val3dity_violations": float(len(oracle.violations)),
        "n_triangles": float(sum(len(m.faces) for m in meshes)),
    }
    result.passed = not result.violations and oracle.passed and not oracle.skipped_reason
    return result


# ---------------------------------------------------------------- AC-H2


def check_crossing_order(model: RoadModel, reference: Sequence[Mapping] | None) -> GateResult:
    """AC-H2: vertical order at every crossing matches a hand-labelled list.

    Entries are ``{"upper_road": id, "lower_road": id}`` or
    ``{"upper_name": name, "lower_name": name}``, optionally with ``x``/``y`` to
    pick among several crossings of the same two roads. Every model crossing must
    be labelled and every label must match a model crossing.
    """
    gate = "AC-H2"
    if reference is None:
        return GateResult(gate, passed=False, hard=True, skipped_reason="no hand-labelled crossing reference for this unit")
    names = {r.id: r.name for r in model.roads}
    unmatched = list(model.crossings)
    result = GateResult(gate, passed=True, hard=True)
    correct = 0
    for n, entry in enumerate(reference):
        by_id = "upper_road" in entry
        up = entry["upper_road"] if by_id else entry["upper_name"]
        lo = entry["lower_road"] if by_id else entry["lower_name"]

        def label(road_id: str, by_id: bool = by_id) -> str:
            return road_id if by_id else names.get(road_id, "")

        candidates = [c for c in unmatched if {label(c.upper_road), label(c.lower_road)} == {up, lo}]
        if not candidates:
            result.violations.append(Violation(gate, f"reference[{n}]", f"labelled crossing {up} over {lo} not found in the model"))
            continue
        if "x" in entry and "y" in entry:
            candidates.sort(key=lambda c: math.hypot(c.x - float(entry["x"]), c.y - float(entry["y"])))
        c = candidates[0]
        unmatched.remove(c)
        if label(c.upper_road) == up and label(c.lower_road) == lo:
            correct += 1
        else:
            result.violations.append(Violation(gate, c.id, f"order reversed: model has {label(c.upper_road)} over {label(c.lower_road)}, reference {up} over {lo}"))
    for c in unmatched:
        result.violations.append(
            Violation(gate, c.id, f"model crossing {names.get(c.upper_road, c.upper_road)} over {names.get(c.lower_road, c.lower_road)} is not in the reference list")
        )
    result.metrics = {"n_reference": float(len(reference)), "n_model": float(len(model.crossings)), "n_correct": float(correct)}
    result.passed = not result.violations
    return result


# ---------------------------------------------------------------- AC-H3


def check_voids(
    model: RoadModel,
    meshes: Sequence[Mesh],
    *,
    step_m: float = 0.5,
    lateral_step_m: float = 1.0,
    floor_m: float = 0.05,
    soffit_tol_m: float = 0.01,
    skip_structures: Collection[str] = (),
) -> GateResult:
    """AC-H3: vertical rays from the lower road up to the deck soffit hit nothing.

    For every crossing with a structure, points are sampled on the lower road's
    pavement every ``step_m`` along and ``lateral_step_m`` across, kept where
    they fall under the deck footprint (carried road between the structure ends,
    within the pavement width plus overhang). Each ray runs from the lower
    surface + ``floor_m`` to the soffit (upper surface minus deck depth) less
    ``soffit_tol_m``. Meshes produced by the lower road itself (its id, its
    barriers, or a junction area that includes it, D-097) are ignored. Crossings under a structure in ``skip_structures`` (one
    replaced by its coarsest proxy, FR-AS-1) are not tested.

    Stacked levels (D-140): where another deck that also crosses the lower road passes
    between it and this deck, the ray stops at that deck's soffit, the soffit immediately
    above the road; the void above that deck is tested by its own crossing with this one.
    """
    gate = "AC-H3"
    result = GateResult(gate, passed=True, hard=True)
    tri_parts, owner_parts = [], []
    for k, mesh in enumerate(meshes):
        tri_parts.append(mesh.vertices[mesh.faces])
        owner_parts.append(np.full(len(mesh.faces), k))
    tri = np.concatenate(tri_parts) if tri_parts else np.empty((0, 3, 3))
    owner = np.concatenate(owner_parts) if owner_parts else np.empty(0, np.int64)
    tree = xy_tree(tri) if len(tri) else None
    n_rays = 0
    min_clear = math.inf
    for c in model.crossings:
        if not c.structure_id or c.structure_id in skip_structures:
            continue
        st = model.structure(c.structure_id)
        upper, lower = model.road(c.upper_road), model.road(c.lower_road)
        xy, z_low = _lower_samples(c, st, upper, lower, step_m, lateral_step_m)
        proj = Projector(upper.alignment)
        su, uu = proj.project(xy)
        xs = upper.cross_section
        # Per side: a twin deck's joint side stops at the joint (FR-CR-3), not at the outer overhang.
        half_l = np.array([xs.width_left(float(s)) for s in su]) + st.deck.overhang("left")
        half_r = np.array([xs.width_right(float(s)) for s in su]) + st.deck.overhang("right")
        under = (su >= st.s_start) & (su <= st.s_end) & (uu <= half_l) & (uu >= -half_r)
        if not under.any():
            result.violations.append(Violation(gate, c.id, "no lower-road samples under the deck footprint"))
            continue
        xy, z_low, su, uu = xy[under], z_low[under], su[under], uu[under]
        soffit = np.array([upper.surface_z(float(s), float(u)) for s, u in zip(su, uu, strict=True)]) - st.deck.depth_m
        n_rays += len(xy)
        min_clear = min(min_clear, float(np.min(soffit - z_low)))
        blocked = soffit - z_low <= floor_m
        if blocked.any():
            result.violations.append(Violation(gate, c.id, f"{int(blocked.sum())} samples with soffit at or below the lower road"))
        if tree is None:
            continue
        # A junction area covering the lower road carries that road's surface there (D-097).
        own = {lower.id, *(b.id for b in lower.cross_section.barriers), *(a.id for a in model.junctions if lower.id in a.road_ids)}
        pi, ti, z = vertical_hits(tri, xy, tree=tree)
        mine = np.array([meshes[o].element_id in own for o in owner[ti]], dtype=bool)
        ceiling = _lowest_soffit(model, c, lower, xy, z_low + floor_m, soffit)
        hit = ~mine & (z > z_low[pi] + floor_m) & (z < ceiling[pi] - soffit_tol_m)
        if hit.any():
            ids = sorted({meshes[o].id for o in owner[ti[hit]]})
            result.violations.append(
                Violation(gate, c.id, f"{len(np.unique(pi[hit]))} rays under the deck hit {', '.join(ids[:5])}")
            )
        # A solid enclosing a whole ray crosses none of it: test the ray starts for containment.
        start = np.column_stack([xy, z_low + floor_m])
        for mesh in meshes:
            if not mesh.closed or mesh.element_id in own or not len(mesh.faces):
                continue
            lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
            near = np.all((start > lo) & (start < hi), axis=1)
            if not near.any():
                continue
            inside = points_strictly_inside(mesh.vertices, mesh.faces, start[near])
            if inside.any():
                result.violations.append(Violation(gate, c.id, f"{int(inside.sum())} rays under the deck start inside {mesh.id}"))
    result.metrics = {"n_rays": float(n_rays), "min_clearance_m": min_clear if n_rays else math.nan}
    result.passed = not result.violations
    return result


def _lowest_soffit(model: RoadModel, c, lower: Road, xy: np.ndarray, floor: np.ndarray, soffit: np.ndarray) -> np.ndarray:
    """Per ray, the soffit immediately above the lower road: this deck's, or a lower deck that also crosses the road (D-140)."""
    ceiling = soffit.copy()
    for other in model.crossings:
        if other.id == c.id or other.lower_road != lower.id or not other.structure_id or other.upper_road == c.upper_road:
            continue
        st = model.structure(other.structure_id)
        mid = model.road(other.upper_road)
        sm, um = Projector(mid.alignment).project(xy)
        half_l = np.array([_deck_half(mid, st, "left", float(s)) for s in sm])
        half_r = np.array([_deck_half(mid, st, "right", float(s)) for s in sm])
        under = (sm >= st.s_start) & (sm <= st.s_end) & (um <= half_l) & (um >= -half_r)
        if not under.any():
            continue
        z = np.full(len(xy), np.inf)
        z[under] = np.array([mid.surface_z(float(s), float(u)) for s, u in zip(sm[under], um[under], strict=True)]) - st.deck.depth_m
        lower_deck = (z > floor) & (z < ceiling)
        ceiling[lower_deck] = z[lower_deck]
    return ceiling


def _deck_half(road: Road, st, side: str, s: float) -> float:
    """Deck edge offset as the solids build it: the pavement or the parapets standing on the deck, whichever is wider, plus the overhang."""
    width = road.cross_section.width_left(s) if side == "left" else road.cross_section.width_right(s)
    zone = max((b.offset_m + b.base_width_m for b in road.cross_section.barriers if b.side == side and b.s0 < st.s_end and b.s1 > st.s_start), default=0.0)
    return max(width, zone) + st.deck.overhang(side)


def _lower_samples(c, st, upper: Road, lower: Road, step: float, lateral: float) -> tuple[np.ndarray, np.ndarray]:
    deck_half = max(_max_knot(upper.cross_section.width_left), _max_knot(upper.cross_section.width_right)) + st.deck.overhang_m
    reach = (st.s_end - st.s_start) + 2.0 * deck_half / max(math.cos(c.skew_rad), 0.1) + 5.0
    s0, s1 = max(0.0, c.lower_s - reach), min(lower.length, c.lower_s + reach)
    xy, z = [], []
    for s in np.arange(s0, s1 + 1e-9, step):
        s = float(s)
        wl = lower.cross_section.width_left(s)
        wr = lower.cross_section.width_right(s)
        for u in np.arange(-wr + 0.25, wl - 0.25 + 1e-9, lateral):
            xy.append(lower.alignment.offset_point(s, float(u)))
            z.append(lower.surface_z(s, float(u)))
    return np.array(xy).reshape(-1, 2), np.array(z)


def _max_knot(piece) -> float:
    return max(k.value for k in piece.knots)


# ---------------------------------------------------------------- AC-H4


def check_continuity(model: RoadModel) -> GateResult:
    """AC-H4: elevation, grade, position and heading continuous along roads; curvature
    continuous along mainlines; elevation, grade and cross slope matching across every
    link and junction-area seam (``roader.validate.seams``)."""
    gate = "AC-H4"
    result = GateResult(gate, passed=True, hard=True)
    add = result.violations.append
    n_joints = 0
    for road in model.roads:
        els = road.alignment.elements
        for k, (a, b) in enumerate(itertools.pairwise(els)):
            n_joints += 1
            x, y, h, _k = a.end_state()
            where = f"alignment joint {k}/{k + 1} at s={b.s0:.3f}"
            if abs(a.s0 + a.length - b.s0) > STATION_TOL:
                add(Violation(gate, road.id, f"{where}: station gap {b.s0 - a.s0 - a.length:.6g} m"))
            gap = math.hypot(x - b.x0, y - b.y0)
            if gap > JOINT_POS_TOL:
                add(Violation(gate, road.id, f"{where}: position gap {gap:.4f} m"))
            dh = math.remainder(h - b.heading0, 2.0 * math.pi)
            if abs(dh) > JOINT_HEADING_TOL:
                add(Violation(gate, road.id, f"{where}: heading break {dh:.6g} rad"))
        pels = road.profile.elements
        for k, (a, b) in enumerate(itertools.pairwise(pels)):
            n_joints += 1
            where = f"profile joint {k}/{k + 1} at s={b.s0:.3f}"
            if abs(a.s0 + a.length - b.s0) > STATION_TOL:
                add(Violation(gate, road.id, f"{where}: station gap {b.s0 - a.s0 - a.length:.6g} m"))
            dz = b.z0 - a.z(a.length)
            if abs(dz) > JOINT_Z_TOL:
                add(Violation(gate, road.id, f"{where}: elevation step {dz:.4f} m"))
            dg = b.g0 - a.g1
            if abs(dg) > JOINT_GRADE_TOL:
                add(Violation(gate, road.id, f"{where}: grade step {dg:.6f}"))
        result.violations.extend(curvature_violations(road))
    seam_v, n_seam = seam_violations(model)
    result.violations.extend(seam_v)
    n_joints += len(model.links)
    result.metrics = {"n_joints": float(n_joints), "n_seam_samples": float(n_seam), "n_junctions": float(len(model.junctions))}
    result.passed = not result.violations
    return result


# ---------------------------------------------------------------- AC-H5


def hash_exports(root: str | os.PathLike) -> dict[str, str]:
    """sha256 of every file under root, keyed by POSIX relative path, sorted."""
    base = Path(root)
    out = {}
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[path.relative_to(base).as_posix()] = h.hexdigest()
    return dict(sorted(out.items()))


def compare_runs(a: str | os.PathLike | Mapping[str, str], b: str | os.PathLike | Mapping[str, str]) -> GateResult:
    """AC-H5: two runs on the same inputs produce identical export hashes."""
    gate = "AC-H5"
    ha = dict(a) if isinstance(a, Mapping) else hash_exports(a)
    hb = dict(b) if isinstance(b, Mapping) else hash_exports(b)
    result = GateResult(gate, passed=True, hard=True)
    for path in sorted(set(ha) | set(hb)):
        if path not in hb:
            result.violations.append(Violation(gate, path, "file only in run A"))
        elif path not in ha:
            result.violations.append(Violation(gate, path, "file only in run B"))
        elif ha[path] != hb[path]:
            result.violations.append(Violation(gate, path, "content differs between runs"))
    if not ha and not hb:
        result.skipped_reason = "no exported files to compare"
    result.metrics = {"n_files": float(len(set(ha) | set(hb)))}
    result.passed = not result.violations and not result.skipped_reason
    return result


# ---------------------------------------------------------------- AC-H6


def _walk(obj, path: str):
    if isinstance(obj, (Provenance, Enum)):
        return
    if is_dataclass(obj) and not isinstance(obj, type):
        yield path, obj
        for f in fields(obj):
            yield from _walk(getattr(obj, f.name), f"{path}.{f.name}")
    elif isinstance(obj, (list, tuple)):
        for k, item in enumerate(obj):
            yield from _walk(item, f"{path}[{k}]")
    elif isinstance(obj, dict):
        for k, item in obj.items():
            yield from _walk(item, f"{path}[{k!r}]")


def model_element_ids(model: RoadModel) -> set[str]:
    """Ids a mesh may name as its producing element (FR-DM-3)."""
    ids = {obj.id for _p, obj in _walk(model, "model") if isinstance(getattr(obj, "id", None), str)}
    if model.terrain is not None:
        ids.add(model.terrain.id)
    return ids


def check_provenance(model: RoadModel, meshes: Sequence[Mesh]) -> GateResult:
    """AC-H6: every element with a ``prov`` field has a valid Provenance; every mesh names
    an existing model element and carries a Source."""
    gate = "AC-H6"
    result = GateResult(gate, passed=True, hard=True)
    n_prov = 0
    for path, obj in _walk(model, "model"):
        prov_field = next((f for f in fields(obj) if f.name == "prov"), None)
        if prov_field is None:
            continue
        prov = obj.prov
        if prov is None and prov_field.default is None:
            continue  # optional provenance, derived from the element's parameters (D-034)
        n_prov += 1
        label = getattr(obj, "id", "") or path
        problem = ""
        if not isinstance(prov, Provenance):
            problem = "missing provenance"
        elif not isinstance(prov.source, Source):
            problem = f"invalid source {prov.source!r}"
        elif not (isinstance(prov.confidence, (int, float)) and 0.0 <= float(prov.confidence) <= 1.0):
            problem = f"confidence {prov.confidence!r} outside [0, 1]"
        elif prov.source == Source.RULE and not prov.rule_id:
            problem = "rule-sourced provenance without a rule_id"
        if problem:
            result.violations.append(Violation(gate, label, f"{path}: {problem}"))
    ids = model_element_ids(model)
    for mesh in meshes:
        if not mesh.element_id:
            result.violations.append(Violation(gate, mesh.id, "mesh has no element_id"))
        elif mesh.element_id not in ids:
            result.violations.append(Violation(gate, mesh.id, f"mesh element_id {mesh.element_id!r} is not in the model"))
        elif not isinstance(mesh.source, Source):
            result.violations.append(Violation(gate, mesh.id, f"mesh source {mesh.source!r} is not a Source"))
    result.metrics = {"n_prov": float(n_prov), "n_meshes": float(len(meshes))}
    result.passed = not result.violations
    return result


# ---------------------------------------------------------------- FR-EX-1 degraded elements


def check_degraded(model: RoadModel) -> GateResult | None:
    """FR-EX-1: elements replaced or dropped because building them raised (D-119) hold the unit.

    None when the model carries no ``ELEMENT_FAILED`` flag, so complete units report as before.
    """
    flags = sorted((f for f in model.flags if f.code == ELEMENT_FAILED), key=lambda f: (f.element_id, f.req, f.message))
    if not flags:
        return None
    return GateResult(
        "FR-EX-1",
        passed=False,
        hard=True,
        metrics={"n_elements": float(len({f.element_id for f in flags}))},
        violations=[Violation(f.req, f.element_id, f.message) for f in flags],
    )


# ---------------------------------------------------------------- report


def run_gates(
    model: RoadModel,
    meshes: Sequence[Mesh],
    *,
    points: np.ndarray | None = None,
    crossings_ref: Sequence[Mapping] | None = None,
    measured: Mapping | None = None,
    export_runs: tuple[str | os.PathLike, str | os.PathLike] | None = None,
    seams: Iterable[Seam] = (),
    val3dity_exe: str | None = None,
    profile: str = "analytic",
    lods: Mapping[int, Sequence[Mesh]] | None = None,
    min_embed_m: float = DEFAULT_MIN_EMBED_M,
) -> GateReport:
    """All section 9 gates that apply to one unit.

    ``points``: (N, 3) pavement and deck returns for AC-A1 (held out internally).
    ``measured``: optional ``edges`` (AC-A3), ``crossing_dz`` with its
    ``crossing_points`` (AC-A4, D-096) and ``structure_stations`` (AC-A5) as
    produced by the pipeline.
    ``export_runs``: two export directories from identical runs (AC-H5).
    ``profile``: export profile of ``meshes`` ("analytic" or "realtime", FR-SV-7).
    ``lods``: the LOD chain, level -> meshes (FR-AS-1, hard when given). AC-P3 and
    AC-P4 then run on level 0, otherwise on ``meshes``.
    """
    report = GateReport(unit_id=model.unit.unit_id)
    degraded = check_degraded(model)
    if degraded is not None:
        report.results.append(degraded)
    report.results.append(check_solids(meshes, seams=seams, val3dity_exe=val3dity_exe, profile=profile, min_embed_m=min_embed_m))
    report.results.append(check_crossing_order(model, crossings_ref))
    report.results.append(check_voids(model, meshes))
    report.results.append(check_continuity(model))
    if export_runs is None:
        report.results.append(GateResult("AC-H5", passed=False, hard=True, skipped_reason="no second export run to compare"))
    else:
        report.results.append(compare_runs(*export_runs))
    report.results.append(check_provenance(model, meshes))
    if points is not None:
        report.results.append(surface_accuracy(model, points))
    measured = measured or {}
    if measured.get("edges"):
        report.results.append(horizontal_accuracy(model, measured["edges"]))
    if measured.get("crossing_dz"):
        report.results.append(crossing_dz_accuracy(model, measured["crossing_dz"], at=measured.get("crossing_points")))
    if measured.get("structure_stations"):
        report.results.append(structure_station_accuracy(model, measured["structure_stations"]))
    if lods is not None:
        report.results.append(lod.check_lods(model, lods))
    report.results.append(check_transitions(model))
    finest = list(lods[0]) if lods is not None and 0 in lods else list(meshes)
    if finest:
        report.results.append(check_budgets(model, finest))
        if any(getattr(m, "uv0", None) is not None for m in finest):
            report.results.append(check_uv_stretch(finest))
    return report
