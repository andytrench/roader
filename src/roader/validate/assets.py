"""Plausibility and asset targets AC-P2, AC-P3 (FR-AS-2) and AC-P4 (FR-AS-4).

These are targets, not hard gates (spec section 9): results carry ``hard=False``.

AC-P2: in every road's cross-section the pavement widths and the cross slopes
(banking) change only over a transition of at least ``MIN_TRANSITION_M``; lane
groups that abut on one side keep their lane width; profile grades do not step
at element joints and every vertical curve is at least ``MIN_TRANSITION_M`` long.

AC-P3 budget definitions, counted on the finest level:

- Freeway on grade: one asset per ``kind == "mainline"`` road. Its triangles are
  those of meshes produced by the road itself, its barriers and its side slopes,
  whose centroid lies outside the deck footprint of every structure the road
  carries. Its length is the road length minus those structures' spans, scaled
  by lanes / 4 (lane counts from the lane groups at mid-road, 2 when none), so
  a 2-lane carriageway is half a kilometre of four-lane freeway per kilometre.
- Ordinary overpass: one asset per ``kind == "bridge"`` structure: meshes
  produced by the structure, its supports, and barriers of the carried road
  lying within the structure's stations.

The target is met when the median ratio actual / budget is at most 1; any asset
over ``DEFECT_FACTOR`` times its budget is a defect.

AC-P4 contract (lane ``solids``): ``Mesh.uv0`` is an optional (N, 2) array of
(station m, offset m), or (M, 3, 2) per face corner (generated solids, which need
UV seams at creases without splitting shared vertices). Per triangle the affine map from uv0 to 3D gives the 3D
length of a unit step in station and in offset; stretch is how far each is from
1. A triangle whose uv0 image has no area while its 3D area is positive has
unbounded stretch.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence

import numpy as np
import shapely

from roader.model.mesh import Mesh
from roader.model.schema import Road, RoadModel, Structure
from roader.validate.report import GateResult, Violation

MIN_TRANSITION_M = 1.0  # m: a value change over less than this is a step (AC-P2)
VALUE_EPS = 1e-9
GRADE_EPS = 1e-5  # D-020 within-road grade tolerance
FREEWAY_BUDGET_PER_KM = 150_000.0  # triangles per km of four-lane freeway on grade (FR-AS-2)
OVERPASS_BUDGET = 60_000.0  # triangles per ordinary overpass incl. substructure (FR-AS-2)
DEFECT_FACTOR = 3.0
UV_STRETCH_MAX = 0.05  # AC-P4
UV_AREA_EPS = 1e-12  # m^2


# ---------------------------------------------------------------- AC-P2


def check_transitions(model: RoadModel, *, min_transition_m: float = MIN_TRANSITION_M) -> GateResult:
    """AC-P2: no lane-width, banking or grade step changes; transitions of nonzero length."""
    gate = "AC-P2"
    result = GateResult(gate, passed=True, hard=False)
    add = result.violations.append
    n = 0
    for road in model.roads:
        xs = road.cross_section
        for name in ("width_left", "width_right", "slope_left", "slope_right"):
            knots = getattr(xs, name).knots
            for a, b in itertools.pairwise(knots):
                n += 1
                ds, dv = b.s - a.s, b.value - a.value
                if ds <= 0.0:
                    add(Violation(gate, road.id, f"{name}: knots at s={a.s:.3f} and {b.s:.3f} do not increase (step)"))
                elif abs(dv) > VALUE_EPS and ds < min_transition_m:
                    add(Violation(gate, road.id, f"{name}: change {dv:+.4f} over {ds:.3f} m at s={a.s:.3f}, under {min_transition_m:g} m (step)"))
        for side in ("left", "right"):
            groups = sorted((g for g in xs.lanes if g.side == side), key=lambda g: (g.s0, g.s1))
            for a, b in itertools.pairwise(groups):
                n += 1
                if abs(b.s0 - a.s1) <= 1e-6 and abs(b.lane_width_m - a.lane_width_m) > VALUE_EPS:
                    add(Violation(gate, road.id, f"{side} lane width steps {a.lane_width_m:.3f} -> {b.lane_width_m:.3f} m at s={b.s0:.3f}"))
        pels = road.profile.elements
        for k, el in enumerate(pels):
            n += 1
            if abs(el.g1 - el.g0) > GRADE_EPS and el.length < min_transition_m:
                add(Violation(gate, road.id, f"vertical curve {k} changes grade {el.g1 - el.g0:+.4f} over {el.length:.3f} m (step)"))
            if k and abs(el.g0 - pels[k - 1].g1) > GRADE_EPS:
                add(Violation(gate, road.id, f"grade steps {el.g0 - pels[k - 1].g1:+.5f} at s={el.s0:.3f} with no vertical curve"))
    result.metrics = {"n_transitions": float(n)}
    result.passed = not result.violations
    return result


# ---------------------------------------------------------------- AC-P3


def _deck_polygon(road: Road, st: Structure, margin: float = 1.0) -> shapely.Polygon:
    """Plan footprint of a structure on its carried road, widened by overhang and margin."""
    xs = road.cross_section
    pad = st.deck.overhang_m + margin
    skew = max((abs(math.tan(sp.skew_rad)) for sp in st.supports), default=0.0)
    half = max(max(k.value for k in xs.width_left.knots), max(k.value for k in xs.width_right.knots)) + pad
    s0 = max(0.0, st.s_start - margin - skew * half)
    s1 = min(road.length, st.s_end + margin + skew * half)
    stations = np.linspace(s0, s1, max(2, math.ceil((s1 - s0) / 2.0) + 1))
    left = [road.alignment.offset_point(float(s), xs.width_left(float(s)) + pad) for s in stations]
    right = [road.alignment.offset_point(float(s), -xs.width_right(float(s)) - pad) for s in stations]
    return shapely.Polygon(left + right[::-1]).buffer(0)


def _lane_count(road: Road) -> int:
    mid = 0.5 * road.length
    count = sum(g.count for g in road.cross_section.lanes if g.s0 <= mid <= g.s1)
    return count or 2


def budget_assets(model: RoadModel, meshes: Sequence[Mesh]) -> list[dict]:
    """Per asset: kind, id, triangles, measure (four-lane km or 1 per overpass), budget."""
    by_element: dict[str, list[Mesh]] = {}
    for m in meshes:
        by_element.setdefault(m.element_id, []).append(m)
    assets = []
    for road in model.roads:
        if road.kind != "mainline":
            continue
        ids = {road.id, *(b.id for b in road.cross_section.barriers), *(ss.id for ss in model.side_slopes if ss.road_id == road.id)}
        carried = [st for st in model.structures if st.carried_road == road.id]
        decks = shapely.union_all([_deck_polygon(road, st) for st in carried]) if carried else None
        n_tri = 0
        for eid in sorted(ids):
            for m in by_element.get(eid, []):
                if decks is None:
                    n_tri += len(m.faces)
                    continue
                c = m.vertices[m.faces].mean(axis=1)
                n_tri += int((~shapely.contains_xy(decks, c[:, 0], c[:, 1])).sum())
        on_grade = road.length - sum(st.s_end - st.s_start for st in carried)
        km4 = max(on_grade, 0.0) * _lane_count(road) / 4.0 / 1000.0
        if km4 > 0.0:
            assets.append({"kind": "freeway", "id": road.id, "triangles": n_tri, "measure": km4, "budget": FREEWAY_BUDGET_PER_KM})
    for st in model.structures:
        if st.kind != "bridge":
            continue
        ids = {st.id, *(sp.id for sp in st.supports)}
        try:
            road = model.road(st.carried_road)
            ids |= {b.id for b in road.cross_section.barriers if b.s0 >= st.s_start - 0.5 and b.s1 <= st.s_end + 0.5}
        except KeyError:
            pass
        n_tri = sum(len(m.faces) for eid in ids for m in by_element.get(eid, []))
        assets.append({"kind": "overpass", "id": st.id, "triangles": n_tri, "measure": 1.0, "budget": OVERPASS_BUDGET})
    return assets


def check_budgets(model: RoadModel, meshes: Sequence[Mesh]) -> GateResult:
    """AC-P3 / FR-AS-2: triangle budgets met at the median, no asset over three times budget."""
    gate = "AC-P3"
    result = GateResult(gate, passed=True, hard=False)
    assets = budget_assets(model, meshes)
    if not assets:
        result.passed = False
        result.skipped_reason = "no mainline on grade and no bridge to budget"
        return result
    ratios = []
    for a in assets:
        actual = a["triangles"] / a["measure"]
        ratio = actual / a["budget"]
        ratios.append(ratio)
        key = "tri_per_4lane_km" if a["kind"] == "freeway" else "tri_per_overpass"
        result.metrics[f"{key}:{a['id']}"] = actual
        if ratio > DEFECT_FACTOR:
            result.violations.append(
                Violation(gate, a["id"], f"{a['kind']} {actual:.0f} triangles{' per km' if a['kind'] == 'freeway' else ''}, {ratio:.2f}x budget (defect over {DEFECT_FACTOR:g}x)")
            )
    median = statistics.median(ratios)
    result.metrics.update({"median_ratio": median, "max_ratio": max(ratios), "n_assets": float(len(assets))})
    if median > 1.0:
        result.violations.append(Violation(gate, model.unit.unit_id, f"median asset at {median:.2f}x its triangle budget"))
    result.passed = not result.violations
    return result


# ---------------------------------------------------------------- AC-P4


def uv_stretch(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per triangle: (along stretch, across stretch, degenerate uv) for a mesh with uv0."""
    uv = np.asarray(mesh.uv0, dtype=np.float64)
    if uv.shape == (len(mesh.faces), 3, 2):
        q = uv
    else:
        uv = uv.reshape(-1, 2)
        if len(uv) != len(mesh.vertices):
            raise ValueError(f"{mesh.id}: uv0 has {len(uv)} rows for {len(mesh.vertices)} vertices")
        q = uv[mesh.faces]
    p = mesh.vertices[mesh.faces]
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=2)  # (M, 3, 2)
    d = np.stack([q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]], axis=2)  # (M, 2, 2)
    area3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    det = d[:, 0, 0] * d[:, 1, 1] - d[:, 0, 1] * d[:, 1, 0]
    live = area3 > UV_AREA_EPS
    degenerate = live & (np.abs(0.5 * det) <= UV_AREA_EPS)
    ok = live & ~degenerate
    safe = np.where(ok, det, 1.0)
    inv = np.empty_like(d)
    inv[:, 0, 0], inv[:, 0, 1] = d[:, 1, 1] / safe, -d[:, 0, 1] / safe
    inv[:, 1, 0], inv[:, 1, 1] = -d[:, 1, 0] / safe, d[:, 0, 0] / safe
    jac = np.einsum("mij,mjk->mik", e, inv)  # columns: dX/dstation, dX/doffset
    along = np.where(ok, np.abs(np.linalg.norm(jac[:, :, 0], axis=1) - 1.0), 0.0)
    across = np.where(ok, np.abs(np.linalg.norm(jac[:, :, 1], axis=1) - 1.0), 0.0)
    return along, across, degenerate


def check_uv_stretch(meshes: Sequence[Mesh], *, max_stretch: float = UV_STRETCH_MAX) -> GateResult:
    """AC-P4: uv0 stretch along and across the road under 5% at the finest LOD."""
    gate = "AC-P4"
    result = GateResult(gate, passed=True, hard=False)
    with_uv = [m for m in meshes if getattr(m, "uv0", None) is not None and len(m.faces)]
    if not with_uv:
        result.passed = False
        result.skipped_reason = "no mesh carries uv0 (FR-AS-4)"
        return result
    worst_along = worst_across = 0.0
    n_tri = 0
    for m in with_uv:
        try:
            along, across, degenerate = uv_stretch(m)
        except ValueError as exc:
            result.violations.append(Violation(gate, m.id, str(exc)))
            continue
        n_tri += len(along)
        a, c = float(along.max(initial=0.0)), float(across.max(initial=0.0))
        worst_along, worst_across = max(worst_along, a), max(worst_across, c)
        if degenerate.any():
            result.violations.append(Violation(gate, m.id, f"{int(degenerate.sum())} triangles with zero uv0 area (unbounded stretch)"))
        if a >= max_stretch or c >= max_stretch:
            bad = int(((along >= max_stretch) | (across >= max_stretch)).sum())
            result.violations.append(Violation(gate, m.id, f"uv0 stretch along {a:.1%}, across {c:.1%} (limit {max_stretch:.0%}) on {bad} triangles"))
    result.metrics = {
        "max_stretch_along": worst_along,
        "max_stretch_across": worst_across,
        "n_meshes": float(len(with_uv)),
        "n_meshes_without_uv0": float(sum(1 for m in meshes if getattr(m, "uv0", None) is None)),
        "n_triangles": float(n_tri),
    }
    result.passed = not result.violations
    return result
