"""Joint L2 fitting over connected components (FR-AF-3, AC-H4, D-020).

Roads are first fitted one by one (``roader.fit.road``). Here every connected
component (roads joined by links or crossings) is re-solved together, subject to:

- equal elevation and equal surface slope where roads join (each link seam);
- the measured surface-to-surface height difference kept at every crossing;
- profile smoothness, by keeping each road's PVI stations and curve lengths.

With PVIs and curve lengths fixed each profile is linear in its start elevation
and tangent grades (``roader.fit.profile._design``), so the component is one
equality-constrained robust least-squares problem: the constraints are solved
exactly on their null space and the returns are fitted on what is left.

A seam joins a *child* road end (a ramp leaving or joining) to a *parent* road
(the one it leaves or joins). The child takes the parent's surface plane at the
seam: its cross slope on each side is held at the parent's surface slope across
the child's direction for ``SEAM_HOLD_M``, then moves to its own measured values
over ``SEAM_TRANSITION_M`` (AC-P2: nonzero transition). Its grade at the seam is
the parent's surface slope along the child's direction. A child that misses its
seam, or whose returns cannot be met with the seam fixed (its RMS rises by more
than ``RMS_RISE_M``), gets two adjacent vertical curves filling the first
``SEAM_WINDOW_M`` from the seam (curves crowding the seam are replaced): they
carry the seam's elevation and grade to the child's own tangent, so the seam
does not bend the rest of the road. A parent that cannot follow its returns gets
a short curve ``SEAM_PVI_M`` from the seam.

A crossing on a child where it still lies on its parent's pavement, with the
parent crossing the same road, is the parent's grade separation: the child's
recorded height difference there comes from its own extrapolated profile and is
not a separate constraint (flagged ``FR-AF-3:crossing_on_parent``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from roader.fit import profile as fp
from roader.model.geometry import Knot, Piecewise
from roader.model.provenance import Provenance, Source
from roader.model.schema import Crossing, Flag, Link, Road
from roader.rules import RuleBook

SEAM_HOLD_M = 5.0
SEAM_TRANSITION_M = 20.0
SEAM_PVI_M = 30.0
SEAM_WINDOW_M = 60.0  # a child's seam transition curves lie within this of the seam
SEAM_MISS_Z_M = 0.005  # half the D-020 link tolerances: a seam missed by more gets its transition up front
SEAM_MISS_GRADE = 0.0005
RMS_RISE_M = 0.005
OVERLAP_STEP_M = 2.0  # an unobserved child on its parent's pavement follows the parent's surface (_overlap_points)
OVERLAP_REACH_M = 200.0
OVERLAP_OWN_M = 1.5
OVERLAP_WEIGHT = 25.0  # per 2 m row; a full 1 m return bin weighs 1: the parent's surface is the child's there
RESIDUAL_CAP_M = 1.0  # a residual counts at most this much in a road's RMS, so a diverging solve shows
SEAM_INSET_M = 0.25  # cross-slope samples inset from the edges and the crown, as the AC-H4 test samples
SEAM_LATERAL_M = 1.0
BAND_M = 0.15  # surface band around the profile for the returns used (as in roader.fit.road)
DIFF_M = 0.05
REG = 1e-6  # ties unobserved parameters to their independent values
INDEPENDENT_TOL = 1e-7
CONSISTENT_TOL = 1e-6
CONSTRAINT_TOL = 1e-5
REVERT_RMS_M = 0.05  # a component whose solve would worsen any road's returns by this much is left as fitted
IRLS_STEPS = 5
JOINT_FLAG = "FR-AF-3:joint"
SEAM_FLAG = "FR-AF-3:seam"


@dataclass
class JointRoad:
    """One road in the joint solve: the model road, its L2 chain and the returns it was fitted to."""

    road: Road
    vert: fp._Vert
    s: np.ndarray  # station, offset, raw height of the surface returns used for L2
    u: np.ndarray
    z: np.ndarray
    gaps: list[tuple[float, float]] = field(default_factory=list)
    strip_factor: float = 1.0
    profile_flags: list[Flag] | None = None  # set when the joint solve rebuilt the profile
    bins: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(default=None, repr=False)


@dataclass
class Seam:
    """Where a child road's end meets its parent (FR-AF-3); stations in each road's own frame."""

    link: Link
    child: str
    child_s: float
    parent: str
    parent_s: float
    parent_u: float


def _project(road: Road, x: float, y: float) -> tuple[float, float]:
    """``road.alignment.project``: the alignment's cached exact projector (D-088, D-125)."""
    return road.alignment.project(x, y)


def seam_of(link: Link, roads: dict[str, Road]) -> Seam:
    """The child is the road whose seam station is one of its own ends; ``to_road`` when both are."""
    src, dst = roads[link.from_road], roads[link.to_road]
    s_from = link.from_s if link.from_s is not None else (src.length if link.from_end == "end" else 0.0)
    to_end_s = 0.0 if link.to_end == "start" else dst.length
    s_to = link.to_s if link.to_s is not None else to_end_s

    def at_end(road: Road, s: float) -> bool:
        return min(abs(s), abs(road.length - s)) < 1e-6

    if at_end(dst, s_to):
        child, cs, parent = dst, (0.0 if abs(s_to) < 1e-6 else dst.length), src
    else:
        child, cs, parent = src, (0.0 if abs(s_from) < 1e-6 else src.length), dst
    x, y, _h, _k = child.alignment.evaluate(cs)
    ps, pu = _project(parent, x, y)
    return Seam(link, child.id, cs, parent.id, float(ps), float(pu))


# ---------------------------------------------------------------- surface geometry


def _slope(road: Road, s: float, u: float) -> float:
    xs = road.cross_section
    return xs.slope_left(s) if u >= 0.0 else xs.slope_right(s)


def _dslope(road: Road, s: float, u: float) -> float:
    a, b = max(0.0, s - DIFF_M), min(road.length, s + DIFF_M)
    return (_slope(road, b, u) - _slope(road, a, u)) / (b - a) if b - a > 1e-9 else 0.0


def _frame(road: Road, s: float) -> tuple[np.ndarray, np.ndarray, float]:
    _x, _y, h, k = road.alignment.evaluate(s)
    return np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)]), k


def surface_gradient(road: Road, s: float, u: float, grade: float | None = None) -> np.ndarray:
    """Plan gradient (dz/dx, dz/dy) of the pavement at (s, u), as AC-H4 measures it (``validate.seams``)."""
    g = road.profile.grade(s) if grade is None else grade
    t, n, k = _frame(road, s)
    along = (g + _dslope(road, s, u) * u) / max(1.0 - k * u, 1e-6)
    return along * t + _slope(road, s, u) * n


def _side_offsets(road: Road, s: float, side: float) -> list[float]:
    width = road.cross_section.width_left(s) if side > 0 else road.cross_section.width_right(s)
    far = width - SEAM_INSET_M
    if far < SEAM_INSET_M:
        return [side * SEAM_INSET_M]
    us = list(np.arange(SEAM_INSET_M, far + 1e-9, SEAM_LATERAL_M))
    if far - us[-1] > 1e-6:
        us.append(far)
    return [side * float(v) for v in us]


# ---------------------------------------------------------------- cross slope at seams


def _hold(pw: Piecewise, s_c: float, length: float, values: tuple[float, float], prov: Provenance) -> Piecewise:
    """``pw`` following ``values`` (at the seam, at the hold's end) over SEAM_HOLD_M from the road end at s_c, then a transition back."""
    d = 1.0 if s_c < 0.5 * length else -1.0
    hold = min(SEAM_HOLD_M, 0.25 * length)
    trans = min(SEAM_TRANSITION_M, 0.25 * length)
    near, far = s_c + d * hold, s_c + d * (hold + trans)
    resume = pw(far)
    kept = [k for k in pw.knots if d * (k.s - far) > 1e-6]
    new = [Knot(s_c, values[0], prov), Knot(near, values[1], prov), Knot(far, resume, prov)]
    knots = sorted(new + kept, key=lambda k: k.s)
    return Piecewise(knots)


def _slope_across(child: Road, parent: Road, s: float, side: float, fallback: tuple[float, float]) -> float:
    """Mean surface slope of the parent across the child's direction under one side of the child at its station s."""
    _t, n_c, _k = _frame(child, s)
    vals = []
    for u in _side_offsets(child, s, side):
        x, y = child.alignment.offset_point(s, u)
        ps, pu = _project(parent, x, y)
        wl, wr = parent.cross_section.width_left(ps), parent.cross_section.width_right(ps)
        if not (-wr <= pu <= wl):
            continue
        vals.append(float(surface_gradient(parent, ps, pu) @ n_c))
    if not vals:
        vals = [float(surface_gradient(parent, *fallback) @ n_c)]
    return float(np.mean(vals))


def _seam_slopes(child: Road, parent: Road, seam: Seam) -> None:
    """Child cross slope per side = the parent's surface slope across the child, at the seam and at the hold's end.

    Following the parent over the hold (not a constant) keeps the change of cross
    slope along the road equal on both sides of the seam, so the grade matches at
    every offset where the parent's cross slope is in transition (AC-H4).
    """
    xs = child.cross_section
    d = 1.0 if seam.child_s < 0.5 * child.length else -1.0
    near = seam.child_s + d * min(SEAM_HOLD_M, 0.25 * child.length)
    fallback = (seam.parent_s, seam.parent_u)
    targets = {side: (_slope_across(child, parent, seam.child_s, side, fallback), _slope_across(child, parent, near, side, fallback)) for side in (1.0, -1.0)}
    ref = xs.slope_left.knots[0].prov if xs.slope_left.knots else Provenance(Source.MEASURED, 0.5)
    pprov = parent.cross_section.slope_left.knots[0].prov if parent.cross_section.slope_left.knots else ref
    prov = Provenance(
        pprov.source, min(ref.confidence, pprov.confidence), rms_m=pprov.rms_m, n_points=pprov.n_points, rule_id=pprov.rule_id, flags=[SEAM_FLAG]
    )
    xs.slope_left = _hold(xs.slope_left, seam.child_s, child.length, targets[1.0], prov)
    xs.slope_right = _hold(xs.slope_right, seam.child_s, child.length, targets[-1.0], prov)


# ---------------------------------------------------------------- linear rows


def _z_row(v: fp._Vert, s: float) -> np.ndarray:
    return fp._design(np.array([s]), v.pvi, v.lengths)[0]


def _g_row(v: fp._Vert, s: float) -> np.ndarray:
    """d/ds of the design row: grade as a linear function of [z0, g_0 .. g_n]."""
    n = len(v.pvi)
    knots = np.concatenate([[0.0], v.pvi, [np.inf]])
    row = np.zeros(n + 2)
    for j in range(n + 1):
        if knots[j] <= s < knots[j + 1] or (j == n and s >= knots[j]):
            row[1 + j] = 1.0
    if s < 0.0:
        row[1] = 1.0
    for k in range(n):
        L = v.lengths[k]
        t = s - (v.pvi[k] - 0.5 * L)
        if 0.0 < t < L:
            q = t / L if t < 0.5 * L else -(L - t) / L
            row[1 + k + 1] += q
            row[1 + k] -= q
    return row


def _params(v: fp._Vert) -> np.ndarray:
    return np.concatenate([[v.z0], v.grades])


def _with(v: fp._Vert, x: np.ndarray) -> fp._Vert:
    return fp._Vert(float(x[0]), x[1:].copy(), v.pvi.copy(), v.lengths.copy())


def _insert_pvi(v: fp._Vert, p: float, length: float, road_length: float) -> fp._Vert | None:
    """A curve of ``length`` centered at p, on the tangent there; None if it would overlap another."""
    a, b = p - 0.5 * length, p + 0.5 * length
    if a < 0.0 or b > road_length:
        return None
    for q, L in zip(v.pvi, v.lengths, strict=True):
        if a < q + 0.5 * L and b > q - 0.5 * L:
            return None
    k = int(np.searchsorted(v.pvi, p))
    grades = np.insert(v.grades, k + 1, v.grades[k])
    return fp._Vert(v.z0, grades, np.insert(v.pvi, k, p), np.insert(v.lengths, k, length))


# ---------------------------------------------------------------- the solve


def _components(ids: list[str], pairs: list[tuple[str, str]]) -> list[list[str]]:
    parent = {i: i for i in ids}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for _r, g in sorted(groups.items())]


def _pw(pw: Piecewise, s: np.ndarray) -> np.ndarray:
    return np.interp(s, [k.s for k in pw.knots], [k.value for k in pw.knots])


def _corrected(item: JointRoad) -> np.ndarray:
    xs = item.road.cross_section
    return item.z - np.where(item.u >= 0.0, _pw(xs.slope_left, item.s), _pw(xs.slope_right, item.s)) * item.u


def _bins_of(item: JointRoad) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1 m bins of cross-slope-corrected returns within the surface band of the road's own fit.

    The band is taken around ``item.vert`` (the road fitted on its own), so every
    trial of a component is scored on the same returns.
    """
    if item.bins is None:
        zc = _corrected(item)
        keep = np.abs(zc - item.vert.z(item.s)) <= BAND_M + 0.02 * np.abs(item.u)
        item.bins = fp._bins(item.s[keep], zc[keep], item.road.length)
    return item.bins


def _data(item: JointRoad, v: fp._Vert) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(design, z, weight, station) of the road's binned returns for chain ``v``."""
    sb, zb, cnt = _bins_of(item)
    wb = np.sqrt(np.minimum(cnt, 25.0)) / 5.0
    return fp._design(sb, v.pvi, v.lengths), zb, wb, sb


def _assemble(ids, verts, rows):
    off, n = {}, 0
    for i in ids:
        off[i] = n
        n += len(verts[i].grades) + 1
    c = np.zeros((len(rows), n))
    d = np.zeros(len(rows))
    for r, (coef, rhs, _tag) in enumerate(rows):
        for rid, vec in coef.items():
            c[r, off[rid] : off[rid] + len(vec)] += vec
        d[r] = rhs
    return off, n, c, d


def _admit(c: np.ndarray, d: np.ndarray) -> list[int]:
    """Rows kept in order: each is independent of those before it, or redundant and consistent with them.

    Incremental Gram-Schmidt over the normalized rows, carrying each basis vector's
    right-hand side, so a dependent row's implied value is read off directly.
    """
    kept: list[int] = []
    basis: list[np.ndarray] = []
    brhs: list[float] = []
    for r in range(len(c)):
        norm = float(np.linalg.norm(c[r]))
        if norm == 0.0:
            continue
        row, rhs = c[r] / norm, d[r] / norm
        if basis:
            q = np.array(basis)
            proj = q @ row
            rest = row - q.T @ proj
            rest = rest - q.T @ (q @ rest)
            left = float(np.linalg.norm(rest))
            implied = float(proj @ np.array(brhs))
        else:
            rest, left, implied = row, 1.0, 0.0
        if left <= INDEPENDENT_TOL:
            if abs(rhs - implied) > CONSISTENT_TOL:
                continue
            kept.append(r)
            continue
        basis.append(rest / left)
        brhs.append((rhs - implied) / left)
        kept.append(r)
    return kept


def _overlap_points(seam: Seam, roads: dict[str, Road], items: dict[str, JointRoad]) -> list[tuple[float, float, float]]:
    """(child station, parent station, parent cross-slope height at the child's crown) where an unobserved child lies on its parent.

    From the seam inward every ``OVERLAP_STEP_M`` until the child's crown leaves the
    parent's pavement (inset ``SEAM_INSET_M``), over the unobserved stretch of the
    child that starts at the seam (inside one of its gaps, no return bin within
    ``OVERLAP_OWN_M``): there the child's crown height is the parent's surface.
    From the first station with the child's own returns they decide.
    """
    child, parent = roads[seam.child], roads[seam.parent]
    own = _bins_of(items[seam.child])[0]
    d = 1.0 if seam.child_s < 0.5 * child.length else -1.0
    out = []
    xs = parent.cross_section
    for t in np.arange(OVERLAP_STEP_M, min(OVERLAP_REACH_M, child.length), OVERLAP_STEP_M):
        s = seam.child_s + d * t
        x, y = child.alignment.offset_point(s, 0.0)
        ps, pu = _project(parent, x, y)
        if not (-xs.width_right(ps) + SEAM_INSET_M <= pu <= xs.width_left(ps) - SEAM_INSET_M) or not (0.0 < ps < parent.length):
            break
        unobserved = any(a <= s <= b for a, b in items[seam.child].gaps)
        if not unobserved or (len(own) and float(np.min(np.abs(own - s))) <= OVERLAP_OWN_M):
            break
        out.append((float(s), float(ps), _slope(parent, ps, pu) * pu))
    return out


def _overlap_rows(seam: Seam, points, verts: dict[str, fp._Vert]):
    """Soft rows of ``_overlap_points``: child crown height minus the parent's height there."""
    out = []
    for s, ps, rhs in points:
        coef = {seam.child: _z_row(verts[seam.child], s)}
        coef[seam.parent] = coef.get(seam.parent, 0.0) - _z_row(verts[seam.parent], ps)
        out.append((coef, rhs))
    return out


def _solve(items: dict[str, JointRoad], verts: dict[str, fp._Vert], rows: list, soft: list = ()):
    """Equality-constrained robust least squares over the component.

    The returns of each road touch only its own parameters, so the normal matrix is
    built road by road and the constraints are solved exactly with it as one small
    KKT system (redundant consistent rows are handled by least squares). Returns
    (params per road, worst constraint residual).
    """
    ids = sorted(items)
    off, n, c, d = _assemble(ids, verts, rows)
    x0 = np.concatenate([_params(verts[i]) for i in ids])
    blocks = []
    for i in ids:
        a_i, z_i, w_i, _s = _data(items[i], verts[i])
        blocks.append((slice(off[i], off[i] + a_i.shape[1]), a_i, z_i, w_i))
    if soft:
        extra = np.zeros((len(soft), n))
        for r, (coef, _rhs) in enumerate(soft):
            for rid, vec in coef.items():
                extra[r, off[rid] : off[rid] + len(vec)] += vec
        blocks.append((slice(0, n), extra, np.array([rhs for _c, rhs in soft]), np.full(len(soft), OVERLAP_WEIGHT)))
    # Columns are scaled so stations in the hundreds of meters do not swamp z0.
    sq = np.zeros(n)
    for cols, a_i, _z, _w in blocks:
        sq[cols] += np.sum(a_i**2, axis=0)
    scale = np.maximum(np.sqrt(sq), 1.0)
    c_s = c / scale
    norms = np.maximum(np.linalg.norm(c_s, axis=1), 1e-30)
    c_s, d_s = c_s / norms[:, None], d / norms
    y0 = x0 * scale
    scaled = [(cols, a_i / scale[cols], z_i, w_i) for cols, a_i, z_i, w_i in blocks]
    # Squared row weights: the soft L1 IRLS weights of roader.fit.profile._inner.
    wt = [w_i.copy() for _cols, _a, _z, w_i in scaled]
    x = x0.copy()
    m = len(c_s)
    for _ in range(IRLS_STEPS):
        h = REG * np.eye(n)
        g = REG * y0
        for (cols, a_i, z_i, _w), w2 in zip(scaled, wt, strict=True):
            aw = a_i * w2[:, None]
            h[cols, cols] += a_i.T @ aw
            g[cols] += aw.T @ z_i
        kkt = np.zeros((n + m, n + m))
        kkt[:n, :n] = h
        kkt[:n, n:] = c_s.T
        kkt[n:, :n] = c_s
        sol, *_ = np.linalg.lstsq(kkt, np.concatenate([g, d_s]), rcond=None)
        y = sol[:n]
        x = y / scale
        wt = []
        for (cols, a_i, z_i, w_i), (_c2, _a2, _z2, wb_i) in zip(scaled, blocks, strict=True):
            res = a_i @ y[cols] - z_i
            wt.append(wb_i / np.sqrt(1.0 + (res / fp.F_SCALE_M) ** 2))
    resid = float(np.max(np.abs(c @ x - d))) if len(rows) else 0.0
    return {i: x[off[i] : off[i] + len(verts[i].grades) + 1] for i in ids}, resid


def _rms(item: JointRoad, v: fp._Vert) -> float:
    a, zb, _wb, _sb = _data(item, v)
    if not len(zb):
        return 0.0
    r = np.clip(a @ np.concatenate([[v.z0], v.grades]) - zb, -RESIDUAL_CAP_M, RESIDUAL_CAP_M)
    return float(np.sqrt(np.mean(r**2)))


def _seam_rows(seam: Seam, roads: dict[str, Road], verts: dict[str, fp._Vert]):
    child, parent = roads[seam.child], roads[seam.parent]
    vc, vp = verts[seam.child], verts[seam.parent]
    ps, pu = seam.parent_s, seam.parent_u
    t_c, _n_c, _ = _frame(child, seam.child_s)
    t_p, n_p, k_p = _frame(parent, ps)
    slope_p = _slope(parent, ps, pu)
    z_rows = {seam.child: _z_row(vc, seam.child_s)}
    z_rows[seam.parent] = z_rows.get(seam.parent, 0.0) - _z_row(vp, ps)
    c1 = float(t_p @ t_c) / max(1.0 - k_p * pu, 1e-6)
    c2 = c1 * _dslope(parent, ps, pu) * pu + slope_p * float(n_p @ t_c)
    g_rows = {seam.child: _g_row(vc, seam.child_s)}
    g_rows[seam.parent] = g_rows.get(seam.parent, 0.0) - c1 * _g_row(vp, ps)
    return [(z_rows, slope_p * pu, (seam.link.id, "elevation")), (g_rows, c2, (seam.link.id, "grade"))]


def _crossing_rows(c: Crossing, roads: dict[str, Road], verts: dict[str, fp._Vert], dz0: float):
    up, lo = roads[c.upper_road], roads[c.lower_road]
    su, uu = _project(up, c.x, c.y)
    sl, ul = _project(lo, c.x, c.y)
    rows = {c.upper_road: _z_row(verts[c.upper_road], su)}
    rows[c.lower_road] = rows.get(c.lower_road, 0.0) - _z_row(verts[c.lower_road], sl)
    return [(rows, dz0 - _slope(up, su, uu) * uu + _slope(lo, sl, ul) * ul, (c.id, "clearance"))]


def _seam_transition(v: fp._Vert, s0: float, length: float, data_s: np.ndarray | None = None) -> fp._Vert:
    """``v`` with two adjacent curves filling the first SEAM_WINDOW_M from a child's seam at s0 (D-050).

    With the seam's elevation and grade fixed, the two curves' grades carry them to
    the child's own tangent beyond, so the seam bends only this window. Where the
    seam lies in an unobserved stretch (no ``data_s`` station within it) at least two
    shortest curves long, the window is that stretch, so the observed returns keep
    their fit. A curve starting closer to the seam than two shortest curves is
    shortened to start at the window's end (giving up at most its near half), or
    removed when too little of it is left.
    """
    d = 1.0 if s0 < 0.5 * length else -1.0
    window = min(SEAM_WINDOW_M, 0.5 * length)
    least = 2.0 * fp.MIN_CURVE_M
    if data_s is not None and len(data_s):
        unobserved = float(np.min(np.abs(np.asarray(data_s) - s0)))
        if unobserved >= least:
            window = min(window, unobserved)
    if window < least:
        return v
    room = window
    pvi, lengths = v.pvi.copy(), v.lengths.copy()
    gone = []
    for k in sorted(range(len(pvi)), key=lambda k: d * (pvi[k] - s0)):
        mid = d * (pvi[k] - s0)
        a, b = mid - 0.5 * lengths[k], mid + 0.5 * lengths[k]
        if a >= least:
            room = min(room, a)
            break
        room = min(window, max(least, mid))
        if b - room >= fp.MIN_CURVE_M:
            pvi[k], lengths[k] = s0 + d * 0.5 * (room + b), b - room
            break
        gone.append(k)
    v = fp._Vert(v.z0, v.grades.copy(), pvi, lengths)
    for k in sorted(gone, reverse=True):
        v = fp._remove(v, k)
    q = 0.25 * room
    for c in (s0 + d * q, s0 + d * 3.0 * q):
        k = int(np.searchsorted(v.pvi, c))
        seg = min(k, len(v.grades) - 1)
        v = fp._Vert(v.z0, np.insert(v.grades, seg + 1, v.grades[seg]), np.insert(v.pvi, k, c), np.insert(v.lengths, k, 2.0 * q))
    return v


def _misses(seam: Seam, roads: dict[str, Road], verts: dict[str, fp._Vert]) -> bool:
    """The seam's elevation or grade row is off by more than half the D-020 link tolerance with the current chains."""
    rows = _seam_rows(seam, roads, verts)
    tol = (SEAM_MISS_Z_M, SEAM_MISS_GRADE)
    for (coef, rhs, _tag), t in zip(rows, tol, strict=True):
        val = sum(float(np.asarray(vec) @ _params(verts[rid])) for rid, vec in coef.items())
        if abs(val - rhs) > t:
            return True
    return False


def _rounds(sub, verts, cseams, ccross, roads, dz0, base, eased, overlap):
    """Solve, and add freedom near the seams of any road whose returns it cannot follow; up to 3 rounds.

    A child gets its seam transition (``_seam_transition``, once per seam, recorded
    in ``eased``); a parent a short curve ``SEAM_PVI_M`` from the seam.
    """
    verts = dict(verts)
    for _round in range(3):
        rows = [r for s in cseams for r in _seam_rows(s, roads, verts)]
        rows += [r for c in ccross for r in _crossing_rows(c, roads, verts, dz0[c.id])]
        _off, _n, cm, dm = _assemble(sorted(sub), verts, rows)
        keep = set(_admit(cm, dm))
        dropped = [rows[r][2] for r in range(len(rows)) if r not in keep]
        soft = [r for s in cseams for r in _overlap_rows(s, overlap[s.link.id], verts)]
        params, resid = _solve(sub, verts, [rows[r] for r in sorted(keep)], soft)
        trial = {i: _with(verts[i], params[i]) for i in sub}
        grown = False
        for i in sorted(sub):
            if _rms(sub[i], trial[i]) - base[i] <= RMS_RISE_M:
                continue
            for s in cseams:
                if s.child == i and s.link.id not in eased:
                    eased.add(s.link.id)
                    verts[i], grown = _seam_transition(verts[i], s.child_s, roads[i].length, _bins_of(sub[i])[0]), True
            for s0 in sorted(s.parent_s for s in cseams if s.parent == i):
                v = _insert_pvi(verts[i], s0 + _inward(s0, roads[i]) * 1.5 * SEAM_PVI_M, SEAM_PVI_M, roads[i].length)
                if v is not None:
                    verts[i], grown = v, True
        if not grown:
            break
    return trial, dropped, resid


def _inward(s0: float, road: Road) -> float:
    return 1.0 if s0 < 0.5 * road.length else -1.0


def _ensure_freedom(verts: dict[str, fp._Vert], cseams: list[Seam], ccross: list[Crossing], roads: dict[str, Road]) -> None:
    """Short vertical curves near a child's seams until it has two parameters beyond its constraints."""
    for i in sorted(verts):
        need = 2 * sum(s.child == i for s in cseams) + sum(i in (c.upper_road, c.lower_road) for c in ccross)
        stations = sorted(s.child_s for s in cseams if s.child == i)
        k = 1
        while len(verts[i].grades) + 1 < need + 2 and stations and k <= 4:
            for s0 in stations:
                v = _insert_pvi(verts[i], s0 + _inward(s0, roads[i]) * 1.5 * k * SEAM_PVI_M, SEAM_PVI_M, roads[i].length)
                if v is not None and len(verts[i].grades) + 1 < need + 2:
                    verts[i] = v
            k += 1


def _rebuild(item: JointRoad, v: fp._Vert, rules: RuleBook) -> list[Flag]:
    from roader.fit.road import profile_provenance

    road = item.road
    a, zb, _wb, sb = _data(item, v)
    del a
    xs = road.cross_section
    zc = _corrected(item)
    keep = np.abs(zc - v.z(item.s)) <= BAND_M + 0.02 * np.abs(item.u)
    st = np.linspace(0.0, road.length, 11)
    paved = float(np.median([xs.width_left(float(q)) + xs.width_right(float(q)) for q in st]))
    provs, flags = profile_provenance(road.id, road.road_class, v, road.length, item.s[keep], zc[keep], sb, item.gaps, paved, rules, item.strip_factor)
    for p in provs:
        p.flags.append(JOINT_FLAG)
    road.profile = fp.to_profile(v, road.length, provs)
    item.vert = v
    item.profile_flags = flags
    del zb
    return flags


def _on_parent_crossings(crossings: list[Crossing], seams: list[Seam], roads: dict[str, Road]) -> list[tuple[Crossing, Crossing, str, str]]:
    """(crossing, parent's crossing, child, parent) where a child lies on its parent's pavement at a crossing with a road the parent also crosses."""
    out = []
    used: set[str] = set()
    for c in sorted(crossings, key=lambda c: c.id):
        if c.id in used:
            continue
        found = None
        for road_id, other in ((c.lower_road, c.upper_road), (c.upper_road, c.lower_road)):
            for sm in seams:
                if found is not None or sm.child != road_id or sm.parent == other:
                    continue
                parent = roads[sm.parent]
                ps, pu = _project(parent, c.x, c.y)
                if not (-parent.cross_section.width_right(ps) <= pu <= parent.cross_section.width_left(ps)):
                    continue
                twins = [d for d in crossings if d.id != c.id and {d.upper_road, d.lower_road} == {other, sm.parent}]
                if twins:
                    found = (min(twins, key=lambda d: (math.hypot(d.x - c.x, d.y - c.y), d.id)), road_id, sm.parent)
        if found is not None and found[0].id not in {x.id for x, *_ in out}:
            used.add(found[0].id)
            out.append((c, *found))
    return out


def joint_fit_roads(fits, links: list[Link], crossings: list[Crossing], rules: RuleBook) -> list[Flag]:
    """``joint_fit`` on pipeline ``RoadFit``s; their roads, L2 chains and profile flags are updated in place.

    Returns the component flags (FR-AF-3); the FR-AF-4 overrides of every profile
    are in ``RoadFit.profile_flags``.
    """
    items = []
    for f in fits:
        s, u, z = f.surface if f.surface is not None else (np.zeros(0), np.zeros(0), np.zeros(0))
        items.append(JointRoad(f.road, f.vert, s, u, z, list(f.gaps), f.strip_factor))
    flags = joint_fit(items, links, crossings, rules)
    for f, it in zip(fits, items, strict=True):
        f.vert = it.vert
        if it.profile_flags is not None:
            f.profile_flags = it.profile_flags
    return [fl for fl in flags if fl.code.startswith("FR-AF-3")]


def joint_fit(items: list[JointRoad], links: list[Link], crossings: list[Crossing], rules: RuleBook) -> list[Flag]:
    """Re-solve L2 (and the cross slope at seams) of every component in place (FR-AF-3).

    Only components holding at least one seam are re-solved; crossings inside them
    are constraints. A component of crossings alone already keeps its measured
    height differences, so it is left exactly as fitted. Returns flags:
    ``FR-AF-3:infeasible`` when a component's constraints cannot all hold, and the
    rule overrides of the refitted elements (FR-AF-4).
    """
    return _joint_fit(items, links, crossings, rules)


def _joint_fit(items: list[JointRoad], links: list[Link], crossings: list[Crossing], rules: RuleBook) -> list[Flag]:
    by_id = {it.road.id: it for it in items}
    roads = {i: it.road for i, it in by_id.items()}
    links = [lk for lk in links if lk.from_road in roads and lk.to_road in roads]
    crossings = [c for c in crossings if c.upper_road in roads and c.lower_road in roads]
    seams = [seam_of(lk, roads) for lk in sorted(links, key=lambda lk: lk.id)]
    dz0 = {c.id: float(c.dz_m) for c in crossings}
    # Parents before children, so a fork's slopes follow an already-held parent.
    depth = {i: 0 for i in roads}
    for _ in range(len(seams)):
        for sm in seams:
            depth[sm.child] = max(depth[sm.child], depth[sm.parent] + 1)
    for sm in sorted(seams, key=lambda q: (depth[q.child], q.link.id)):
        _seam_slopes(roads[sm.child], roads[sm.parent], sm)

    flags: list[Flag] = []
    for it in items:
        it.bins = None
    implied = _on_parent_crossings(crossings, seams, roads)
    for c, parent_c, child, parent in implied:
        flags.append(
            Flag(
                "FR-AF-3:crossing_on_parent",
                c.id,
                "FR-AF-3",
                f"{roads[child].name} lies on the pavement of {roads[parent].name} here; "
                f"the height difference is that of crossing {parent_c.id} and is not enforced separately",
            )
        )
    skip = {c.id for c, *_rest in implied}
    pairs = [(s.child, s.parent) for s in seams] + [(c.upper_road, c.lower_road) for c in crossings]
    seamed = {i for s in seams for i in (s.child, s.parent)}
    for comp in _components(sorted(roads), pairs):
        if not seamed.intersection(comp):
            continue
        sub = {i: by_id[i] for i in comp}
        verts = {i: by_id[i].vert for i in comp}
        base = {i: _rms(by_id[i], verts[i]) for i in comp}
        cseams = [s for s in seams if s.child in sub]
        ccross = [c for c in crossings if c.upper_road in sub and c.id not in skip]
        eased = set()
        overlap = {s.link.id: _overlap_points(s, roads, sub) for s in cseams}
        for s in cseams:
            if _misses(s, roads, verts) or overlap[s.link.id]:
                eased.add(s.link.id)
                verts[s.child] = _seam_transition(verts[s.child], s.child_s, roads[s.child].length, _bins_of(sub[s.child])[0])
        _ensure_freedom(verts, cseams, ccross, roads)
        while True:
            trial, dropped, resid = _rounds(sub, verts, cseams, ccross, roads, dz0, base, eased, overlap)
            rise = {i: _rms(by_id[i], trial[i]) - base[i] for i in comp}
            worst = max(rise, key=lambda i: (rise[i], i))
            if rise[worst] <= REVERT_RMS_M:
                break
            # A seam on a road its returns cannot follow is dropped, not forced.
            off = [s for s in cseams if worst in (s.child, s.parent)]
            for s in off:
                flags.append(
                    Flag(
                        "FR-AF-3:seam_dropped",
                        s.link.id,
                        "FR-AF-3",
                        f"seam not enforced: with it the joint solve worsens {roads[worst].name} by {rise[worst]:.3f} m RMS",
                    )
                )
            cseams = [s for s in cseams if s not in off]
            if not off:
                # Its clearances then: a crossing measured on a road its returns cannot follow.
                gone = [c for c in ccross if worst in (c.upper_road, c.lower_road)]
                for c in gone:
                    flags.append(Flag("FR-AF-3:constraint_dropped", c.id, "FR-AF-3", f"clearance not enforced: with it the joint solve worsens {roads[worst].name} by {rise[worst]:.3f} m RMS"))
                ccross = [c for c in ccross if c not in gone]
                if not gone:
                    break
            if not cseams:
                trial = None
                break
        if trial is None:
            flags.append(Flag("FR-AF-3:reverted", comp[0], "FR-AF-3", f"component of {len(comp)} roads left as fitted one by one"))
            continue
        for tag, what in dropped:
            flags.append(Flag("FR-AF-3:constraint_dropped", tag, "FR-AF-3", f"{what} constraint conflicts with the others in its component and is not enforced"))
        if resid > CONSTRAINT_TOL:
            flags.append(Flag("FR-AF-3:infeasible", comp[0], "FR-AF-3", f"component of {len(comp)} roads: constraints miss by {resid:.2e}"))
        for i in comp:
            flags.extend(_rebuild(by_id[i], trial[i], rules))
    return flags
