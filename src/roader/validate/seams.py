"""AC-H4 across links and L6 junction areas, and curvature continuity on mainlines (D-020).

A seam is where two drivable surfaces meet: the end (or a declared station) of
one road against another road, or a road against a junction area. Along the
seam the two surfaces must agree in elevation within ``LINK_Z_TOL`` and in
slope within ``LINK_SLOPE_TOL``, both along the road (grade) and across it
(cross slope). Slopes are compared as plan gradients of the surface, so roads
meeting at an angle or in opposite directions compare correctly.

Milestone 2 contracts (lane ``model``), read with ``getattr`` so older models
still load:

- ``Link.from_s`` / ``Link.to_s`` (float or None): seam station on each road.
  ``from_s`` None is the ``from_end`` end; ``to_s`` None is the projection of
  the ``from`` seam point onto the ``to`` road.
- ``JunctionArea.seams``: items with ``road_id: str`` and ``s: float``, the
  station where that road's surface meets the area.
- ``JunctionArea.surface_z(x, y) -> float``: top of pavement inside the area
  (NaN or an exception outside it).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import numpy as np
import shapely

from roader.model.schema import Link, Road, RoadModel
from roader.validate.report import Violation

GATE = "AC-H4"
LINK_Z_TOL = 0.01  # m (D-020)
LINK_SLOPE_TOL = 0.001  # grade and cross slope (D-020)
JOINT_CURVATURE_TOL = 1e-6  # 1/m: mainline element joints are fitted curvature-continuous
SEAM_INSET = 0.25  # m kept clear of pavement edges and of the crown
SEAM_LATERAL = 1.0  # m between samples across the road
DIFF_STEP = 0.05  # m, finite-difference step on road and junction surfaces


def road_gradient(road: Road, s: float, u: float) -> np.ndarray:
    """Plan gradient (dz/dx, dz/dy) of the pavement surface at (s, u).

    Along the road: dz/ds at fixed u (central difference, one-sided at the ends)
    over the plan speed 1 - k u of the offset line. Across: the side's cross slope.
    """
    length = road.length
    s0, s1 = max(0.0, s - DIFF_STEP), min(length, s + DIFF_STEP)
    if s1 - s0 > 1e-9:
        dzds = (road.surface_z(s1, u) - road.surface_z(s0, u)) / (s1 - s0)
    else:
        dzds = road.profile.grade(s)
    _x, _y, h, k = road.alignment.evaluate(s)
    along = dzds / max(1.0 - k * u, 1e-6)
    xs = road.cross_section
    cross = xs.slope_left(s) if u >= 0.0 else xs.slope_right(s)
    t = np.array([math.cos(h), math.sin(h)])
    n = np.array([-math.sin(h), math.cos(h)])
    return along * t + cross * n


def _offsets(road: Road, s: float) -> list[float]:
    """Sample offsets across the pavement at s: the crown line plus every SEAM_LATERAL, inset from the edges."""
    wl = road.cross_section.width_left(s)
    wr = road.cross_section.width_right(s)
    out = [0.0]
    for sign, width in ((1.0, wl), (-1.0, wr)):
        far = width - SEAM_INSET
        if far < SEAM_INSET:
            continue
        us = list(np.arange(SEAM_INSET, far + 1e-9, SEAM_LATERAL))
        if far - us[-1] > 1e-6:
            us.append(far)
        out.extend(sign * float(u) for u in us)
    return out


class _SeamStats:
    def __init__(self) -> None:
        self.dz: list[float] = []
        self.d_grade: list[float] = []
        self.d_cross: list[float] = []
        self.missing = 0

    def add(self, dz: float, dgrad: np.ndarray | None, t: np.ndarray, n: np.ndarray, cross_ok: bool) -> None:
        self.dz.append(dz)
        if dgrad is None:
            return
        self.d_grade.append(float(dgrad @ t))
        if cross_ok:
            self.d_cross.append(float(dgrad @ n))

    def violations(self, element_id: str, where: str) -> list[Violation]:
        out = []
        if self.missing:
            out.append(Violation(GATE, element_id, f"{where}: {self.missing} seam samples have no surface on the other side"))
        for name, values, tol, unit in (
            ("elevation", self.dz, LINK_Z_TOL, " m"),
            ("grade", self.d_grade, LINK_SLOPE_TOL, ""),
            ("cross-slope", self.d_cross, LINK_SLOPE_TOL, ""),
        ):
            if not values:
                continue
            worst = max(values, key=abs)
            bad = sum(abs(v) > tol for v in values)
            if bad:
                out.append(Violation(GATE, element_id, f"{where}: {name} mismatch up to {worst:.5f}{unit} at {bad}/{len(values)} samples"))
        return out


# ---------------------------------------------------------------- links


def link_violations(link: Link, roads: Mapping[str, Road]) -> tuple[list[Violation], int]:
    """Elevation, grade and cross slope across one link; returns (violations, samples)."""
    if link.from_road not in roads or link.to_road not in roads:
        return [Violation(GATE, link.id, "link refers to a missing road")], 0
    src, dst = roads[link.from_road], roads[link.to_road]
    from_s = getattr(link, "from_s", None)
    to_s = getattr(link, "to_s", None)
    s_from = float(from_s) if from_s is not None else (src.length if link.from_end == "end" else 0.0)
    if to_s is None:
        x, y, _h, _k = src.alignment.evaluate(s_from)
        s_to, _u = dst.alignment.project(x, y)
    else:
        s_to = float(to_s)
    _x, _y, h, _k = dst.alignment.evaluate(s_to)
    t = np.array([math.cos(h), math.sin(h)])
    n = np.array([-math.sin(h), math.cos(h)])
    stats = _SeamStats()
    for u in _offsets(dst, s_to):
        qx, qy = dst.alignment.offset_point(s_to, u)
        sq, uq = src.alignment.project(qx, qy)
        wl, wr = src.cross_section.width_left(sq), src.cross_section.width_right(sq)
        if not (-wr - 1e-6 <= uq <= wl + 1e-6):
            continue  # this part of dst's seam is not over src's pavement
        dz = dst.surface_z(s_to, u) - src.surface_z(sq, uq)
        dgrad = road_gradient(dst, s_to, u) - road_gradient(src, sq, uq)
        cross_ok = abs(u) >= SEAM_INSET - 1e-9 and abs(uq) >= SEAM_INSET - 1e-9
        stats.add(dz, dgrad, t, n, cross_ok)
    where = f"{src.id}@{s_from:.2f} to {dst.id}@{s_to:.2f}"
    if not stats.dz:
        return [Violation(GATE, link.id, f"{where}: the two pavements do not overlap at the seam")], 0
    return stats.violations(link.id, where), len(stats.dz)


# ---------------------------------------------------------------- junction areas


def _junction_side(junction, road: Road, s: float) -> float:
    """+1 if the junction area lies ahead of station s on the road, -1 if behind."""
    outline = list(getattr(junction, "outline", []) or [])
    if len(outline) >= 3:
        poly = shapely.Polygon(outline)
        ahead = shapely.Point(road.alignment.offset_point(min(s + 0.5, road.length), 0.0))
        x, y, h, _k = road.alignment.evaluate(s)
        if s + 0.5 > road.length:
            ahead = shapely.Point(x + 0.5 * math.cos(h), y + 0.5 * math.sin(h))
        behind = shapely.Point(road.alignment.offset_point(max(s - 0.5, 0.0), 0.0))
        if s - 0.5 < 0.0:
            behind = shapely.Point(x - 0.5 * math.cos(h), y - 0.5 * math.sin(h))
        return 1.0 if poly.distance(ahead) <= poly.distance(behind) else -1.0
    return 1.0 if s >= 0.5 * road.length else -1.0


def _jz(junction, x: float, y: float) -> float:
    try:
        return float(junction.surface_z(float(x), float(y)))
    except (ValueError, KeyError, IndexError):
        return math.nan


def junction_violations(junction, roads: Mapping[str, Road]) -> tuple[list[Violation], int]:
    """Elevation, grade and cross slope along every seam of one junction area."""
    seams = getattr(junction, "seams", None)
    if seams is None or not callable(getattr(junction, "surface_z", None)):
        return [Violation(GATE, junction.id, "junction area has no seams or surface_z (L6 contract), so its seams cannot be checked")], 0
    out: list[Violation] = []
    n = 0
    for seam in seams:
        if seam.road_id not in roads:
            out.append(Violation(GATE, junction.id, f"seam refers to a missing road {seam.road_id}"))
            continue
        road = roads[seam.road_id]
        s = float(seam.s)
        sigma = _junction_side(junction, road, s)
        _x, _y, h, _k = road.alignment.evaluate(s)
        t = np.array([math.cos(h), math.sin(h)])
        nrm = np.array([-math.sin(h), math.cos(h)])
        stats = _SeamStats()
        for u in _offsets(road, s):
            q = np.array(road.alignment.offset_point(s, u))
            zj = _jz(junction, *q)
            if not math.isfinite(zj):
                stats.missing += 1
                continue
            z_ahead = _jz(junction, *(q + sigma * DIFF_STEP * t))
            jg_along = sigma * (z_ahead - zj) / DIFF_STEP
            side = 1.0 if u >= 0.0 else -1.0
            z_out = _jz(junction, *(q + side * DIFF_STEP * nrm))
            jg_cross = side * (z_out - zj) / DIFF_STEP
            dz = zj - road.surface_z(s, u)
            if not (math.isfinite(z_ahead) and math.isfinite(z_out)):
                stats.add(dz, None, t, nrm, False)
                continue
            dgrad = jg_along * t + jg_cross * nrm - road_gradient(road, s, u)
            stats.add(dz, dgrad, t, nrm, abs(u) >= SEAM_INSET - 1e-9)
        n += len(stats.dz)
        out.extend(stats.violations(junction.id, f"seam with {road.id}@{s:.2f}"))
    return out, n


# ---------------------------------------------------------------- curvature


def curvature_violations(road: Road) -> list[Violation]:
    """Curvature continuity at alignment joints of a mainline (heading is checked for every road)."""
    if road.kind != "mainline":
        return []
    out = []
    els = road.alignment.elements
    for k, (a, b) in enumerate(itertools.pairwise(els)):
        dk = b.k0 - a.k1
        if abs(dk) > JOINT_CURVATURE_TOL:
            out.append(Violation(GATE, road.id, f"alignment joint {k}/{k + 1} at s={b.s0:.3f}: curvature break {dk:.3g} 1/m"))
    return out


def seam_violations(model: RoadModel) -> tuple[list[Violation], int]:
    """Every link and junction-area seam in the model; returns (violations, samples)."""
    roads = {r.id: r for r in model.roads}
    out: list[Violation] = []
    n = 0
    for link in model.links:
        v, k = link_violations(link, roads)
        out.extend(v)
        n += k
    for junction in model.junctions:
        v, k = junction_violations(junction, roads)
        out.extend(v)
        n += k
    return out, n
