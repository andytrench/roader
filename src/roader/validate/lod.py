"""FR-AS-1 level-of-detail chain checks (milestone 2).

Contract (lane ``solids``): ``roader.solids.generate_lods(model) -> dict[int, list[Mesh]]``
with levels 0 to 3 at chord tolerances ``LOD_CHORD_TOL``. A structure replaced
by its coarsest proxy has one mesh with ``part == "proxy"`` and
``element_id == structure.id``; a proxy is allowed only at the coarsest level.
Deck meshes (and proxies) carry the structure id as ``element_id``.

Per level: every mesh passes FR-SV-1..6; AC-H3 void rays pass under every
structure that is not a proxy; the top of every deck (topmost hit of a vertical
ray among the structure's meshes, sampled inside the deck) is within the level's
chord tolerance of the model's carried-road surface; the total triangle count
does not grow from one level to the next.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from roader.model.mesh import Mesh
from roader.model.schema import RoadModel, Structure
from roader.validate import gates
from roader.validate._geom import vertical_hits
from roader.validate.report import GateResult, Violation
from roader.validate.solid import check_solid

GATE = "FR-AS-1"
LOD_CHORD_TOL = {0: 0.01, 1: 0.05, 2: 0.25, 3: 1.0}  # m
PROXY_PART = "proxy"
DECK_EDGE_INSET = 0.5  # m kept inside the deck outline when sampling its top
Z_EPS = 1e-3  # m allowed on top of the chord tolerance


def check_lods(
    model: RoadModel,
    lods: Mapping[int, Sequence[Mesh]],
    *,
    tolerances: Mapping[int, float] = LOD_CHORD_TOL,
    deck_step_m: float = 2.0,
    deck_lateral_m: float = 1.0,
) -> GateResult:
    """FR-AS-1 over the whole chain; violations name the level."""
    result = GateResult(GATE, passed=True, hard=True)
    add = result.violations.append
    expected = sorted(tolerances)
    if sorted(lods) != expected:
        add(Violation(GATE, "lods", f"levels {sorted(lods)} differ from the required {expected}"))
    coarsest = expected[-1]
    levels = [k for k in expected if k in lods]
    previous: tuple[int, int] | None = None
    for level in levels:
        meshes = list(lods[level])
        tag = f"LOD {level}"
        if not meshes:
            add(Violation(GATE, "lods", f"{tag}: no meshes"))
            continue
        for mesh in meshes:
            for v in check_solid(mesh):
                add(Violation(v.req, v.element_id, f"{tag}: {v.message}"))

        proxies = {m.element_id for m in meshes if m.part == PROXY_PART}
        if level != coarsest:
            for sid in sorted(proxies):
                add(Violation(GATE, sid, f"{tag}: structure replaced by a proxy before the coarsest level {coarsest}"))
        voids = gates.check_voids(model, meshes, skip_structures=proxies if level == coarsest else set())
        for v in voids.violations:
            add(Violation(v.req, v.element_id, f"{tag}: {v.message}"))
        result.metrics[f"n_rays_lod{level}"] = voids.metrics.get("n_rays", 0.0)

        tol = float(tolerances[level])
        worst = 0.0
        for st in model.structures:
            err, missing, n = deck_top_error(model, st, meshes, deck_step_m, deck_lateral_m)
            if n == 0:
                continue
            if missing:
                add(Violation(GATE, st.id, f"{tag}: {missing}/{n} deck-top samples hit no mesh of the structure"))
            if err > tol + Z_EPS:
                add(Violation(GATE, st.id, f"{tag}: deck top off the model by {err:.3f} m, over the {tol:g} m chord tolerance"))
            worst = max(worst, err)
        result.metrics[f"deck_dz_max_lod{level}"] = worst

        n_tri = sum(len(m.faces) for m in meshes)
        result.metrics[f"n_triangles_lod{level}"] = float(n_tri)
        if previous is not None and n_tri > previous[1]:
            add(Violation(GATE, "lods", f"{tag}: {n_tri} triangles, more than {previous[1]} at LOD {previous[0]}"))
        previous = (level, n_tri)
    result.passed = not result.violations
    return result


def deck_samples(model: RoadModel, st: Structure, step_m: float = 2.0, lateral_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Plan points inside the deck of ``st`` and the model's road surface z there."""
    road = model.road(st.carried_road)
    xs = road.cross_section
    skew = max((abs(math.tan(sp.skew_rad)) for sp in st.supports), default=0.0)
    half = max(max(k.value for k in xs.width_left.knots), max(k.value for k in xs.width_right.knots))
    margin = DECK_EDGE_INSET + skew * half
    s0, s1 = st.s_start + margin, st.s_end - margin
    xy, z = [], []
    if s1 < s0:
        return np.empty((0, 2)), np.empty(0)
    for s in np.linspace(s0, s1, max(2, math.ceil((s1 - s0) / step_m) + 1)):
        s = float(s)
        wl, wr = xs.width_left(s), xs.width_right(s)
        for u in np.arange(-wr + DECK_EDGE_INSET, wl - DECK_EDGE_INSET + 1e-9, lateral_m):
            xy.append(road.alignment.offset_point(s, float(u)))
            z.append(road.surface_z(s, float(u)))
    return np.array(xy).reshape(-1, 2), np.array(z)


def deck_top_error(
    model: RoadModel, st: Structure, meshes: Sequence[Mesh], step_m: float = 2.0, lateral_m: float = 1.0
) -> tuple[float, int, int]:
    """(largest |z_top - z_model|, samples with no hit, samples) for one structure's deck."""
    xy, z_model = deck_samples(model, st, step_m, lateral_m)
    if not len(xy):
        return 0.0, 0, 0
    own = [m for m in meshes if m.element_id == st.id and len(m.faces)]
    if not own:
        return 0.0, len(xy), len(xy)
    tri = np.concatenate([m.vertices[m.faces] for m in own])
    pi, _ti, z = vertical_hits(tri, xy)
    top = np.full(len(xy), -np.inf)
    np.maximum.at(top, pi, z)
    hit = np.isfinite(top)
    err = float(np.max(np.abs(top[hit] - z_model[hit]))) if hit.any() else 0.0
    return err, int((~hit).sum()), len(xy)
