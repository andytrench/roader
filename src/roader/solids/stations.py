"""FR-SG-2 sweep stations and the skew field, shared by every body on one road.

A station line is the line s = s0 + t * u in the road's (s, u) parameter space.
Square stations have t = 0. Inside a structure t = tan(skew) of the supports, so
deck ends, abutments and pier caps follow the skew of what they cross (FR-ST-4).
Outside a structure t ramps back to 0 over a length chosen so neighbouring station
lines never cross within the road's lateral extent.

Skew convention: a support with skew_rad k lies along s = station + tan(k) * u.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from roader.model.schema import Road, Structure, Support

MAX_STATION_SPACING_M = 10.0
STATION_MERGE_M = 0.01
RAMP_FACTOR = 2.4
RAMP_MIN_M = 1.0

PRIORITY_STRUCTURE = 0
PRIORITY_OTHER = 1


@dataclass(frozen=True)
class StationLine:
    s: float  # station where the line crosses the reference line
    t: float  # ds/du along the line

    def at(self, u: float) -> float:
        return self.s + self.t * u


@dataclass(frozen=True)
class SkewKnot:
    s: float
    t: float
    ramp: bool


def support_block(structure: Structure, support: Support) -> tuple[float, float]:
    """Station range of a support's block (abutment or pier cap) on the carried road.

    Abutments sit at the deck ends (FR-ST-3) with thickness cap_width_m along the
    road; pier caps are centred on the support station with width cap_width_m.
    """
    if support.cap_width_m <= 0.0:
        raise ValueError(f"support {support.id}: cap_width_m must be positive")
    if support.kind == "abutment":
        if abs(support.station - structure.s_start) <= abs(support.station - structure.s_end):
            return structure.s_start, structure.s_start + support.cap_width_m
        return structure.s_end - support.cap_width_m, structure.s_end
    if support.kind == "pier":
        half = 0.5 * support.cap_width_m
        return support.station - half, support.station + half
    raise NotImplementedError(f"support {support.id}: kind {support.kind!r} is not generated in milestone 1")


def skew_knots(structures: list[Structure], umax: float) -> list[SkewKnot]:
    """Piecewise-linear skew field t(s) for one road (structures sorted by s_start)."""
    knots: list[SkewKnot] = []
    for st in structures:
        supports = sorted(st.supports, key=lambda sp: (sp.station, sp.id))
        if supports:
            t_a = math.tan(min(supports, key=lambda sp: abs(sp.station - st.s_start)).skew_rad)
            t_b = math.tan(min(supports, key=lambda sp: abs(sp.station - st.s_end)).skew_rad)
        else:
            t_a = t_b = 0.0
        pts = [SkewKnot(st.s_start, t_a, False)]
        for sp in supports:
            if st.s_start + STATION_MERGE_M < sp.station < st.s_end - STATION_MERGE_M:
                pts.append(SkewKnot(sp.station, math.tan(sp.skew_rad), False))
        pts.append(SkewKnot(st.s_end, t_b, False))
        if t_a != 0.0:
            pts.insert(0, SkewKnot(st.s_start - (RAMP_FACTOR * abs(t_a) * umax + RAMP_MIN_M), 0.0, True))
        if t_b != 0.0:
            pts.append(SkewKnot(st.s_end + (RAMP_FACTOR * abs(t_b) * umax + RAMP_MIN_M), 0.0, True))
        if knots and pts[0].s <= knots[-1].s:
            # Ramps of neighbouring structures overlap: interpolate between their ends.
            if knots[-1].ramp:
                knots.pop()
            if pts[0].ramp:
                pts.pop(0)
            if knots and pts[0].s <= knots[-1].s:
                raise ValueError(f"structure {st.id} overlaps the previous structure on its road")
        knots.extend(pts)
    return knots


def skew_at(knots: list[SkewKnot], s: float) -> float:
    if not knots:
        return 0.0
    if s <= knots[0].s:
        return knots[0].t
    for left, right in pairwise(knots):
        if s <= right.s:
            f = (s - left.s) / (right.s - left.s)
            return left.t + f * (right.t - left.t)
    return knots[-1].t


def merge_breakpoints(points: list[tuple[float, int]], length: float) -> list[float]:
    """Sorted breakpoints; any closer than STATION_MERGE_M collapse to the highest priority one."""
    pts = sorted((min(max(s, 0.0), length), prio) for s, prio in points if -STATION_MERGE_M <= s <= length + STATION_MERGE_M)
    pts += [(0.0, -1), (length, -1)]
    pts.sort()
    clusters: list[list[tuple[float, int]]] = []
    for s, prio in pts:
        if clusters and s - clusters[-1][-1][0] <= STATION_MERGE_M:
            clusters[-1].append((s, prio))
        else:
            clusters.append([(s, prio)])
    return [min(cluster, key=lambda item: (item[1], item[0]))[0] for cluster in clusters]


def max_step(road: Road, a: float, b: float, tol: float, umax: float) -> float:
    """Longest station spacing in (a, b) that keeps chord deviation under tol (FR-SG-2).

    (a, b) contains no parameter change, so curvature, grade change and cross-slope
    rate are linear or constant inside it.
    """
    step = MAX_STATION_SPACING_M
    eps = 1e-6 * (b - a)
    sa, sb = a + eps, b - eps
    k = max(abs(road.alignment.evaluate(sa)[3]), abs(road.alignment.evaluate(sb)[3]))
    if k > 1e-12:
        denom = 1.0 - k * umax
        if denom < 0.05:
            raise ValueError(f"road {road.id}: lateral extent {umax:.1f} m reaches the centre of curvature near s={a:.1f}")
        step = min(step, math.sqrt(8.0 * tol * denom / k))
    zpp = abs(road.profile.grade(sb) - road.profile.grade(sa)) / (sb - sa)
    xs = road.cross_section
    for slope, width in ((xs.slope_left, xs.width_left), (xs.slope_right, xs.width_right)):
        rate = abs(slope(b) - slope(a)) / (b - a)
        if rate > 1e-12:
            step = min(step, 4.0 * tol / (rate * umax))
            zpp += 2.0 * rate * abs(width(b) - width(a)) / (b - a)
    if zpp > 1e-12:
        step = min(step, math.sqrt(8.0 * tol / zpp))
    return step


def kinks(road: Road, umax: float) -> list[tuple[float, float]]:
    """Stations where an edge or surface line changes slope, with the size of the change (m/m).

    A skewed station line meets these kinks away from a vertex except on the reference
    line, so spacing near them is limited instead (chord error <= jump * spacing / 4).
    """
    xs = road.cross_section
    out = []
    for pw, scale in ((xs.width_left, 1.0), (xs.width_right, 1.0), (xs.slope_left, umax), (xs.slope_right, umax)):
        knots = pw.knots
        rates = [0.0] + [(b.value - a.value) / (b.s - a.s) for a, b in pairwise(knots)] + [0.0]
        for k, knot in enumerate(knots):
            jump = abs(rates[k + 1] - rates[k]) * scale
            if jump > 1e-12:
                out.append((knot.s, jump))
    return out


def stations(road: Road, breakpoints: list[float], tol: float, umax: float, knots: list[SkewKnot]) -> list[float]:
    skewed = [k for k in knots if k.t != 0.0]
    kinked = kinks(road, umax) if skewed else []
    reach = max((abs(k.t) for k in skewed), default=0.0) * umax
    out = [breakpoints[0]]
    for a, b in pairwise(breakpoints):
        step = max_step(road, a, b, tol, umax)
        if skewed and (abs(skew_at(knots, a)) > 0.0 or abs(skew_at(knots, b)) > 0.0):
            for s, jump in kinked:
                if a - reach <= s <= b + reach:
                    step = min(step, 4.0 * tol / jump)
        n = max(1, math.ceil((b - a) / step - 1e-9))
        out.extend(a + (b - a) * k / n for k in range(1, n))
        out.append(b)
    return out
