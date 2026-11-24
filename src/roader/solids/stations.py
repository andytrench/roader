"""FR-SG-2 sweep stations and the skew field, shared by every body on one road.

A station line is the line s = s0 + t * u in the road's (s, u) parameter space.
Square stations have t = 0. Inside a structure t = tan(skew) of the supports, so
deck ends, abutments and pier caps follow the skew of what they cross (FR-ST-4).
Outside a structure t ramps back to 0 over a length chosen so neighbouring station
lines never cross within the road's lateral extent.

Skew convention: a support with skew_rad k lies along s = station + tan(k) * u.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

from roader.model.schema import Road, Structure, Support

MAX_STATION_SPACING_M = 10.0  # at the finest chord tolerance (D-028)
SPACING_REF_TOL_M = 0.01
STRUCTURE_TOL_M = 0.01  # chord tolerance cap along a structure span, every LOD
STATION_MERGE_M = 0.01
RAMP_FACTOR = 2.4
RAMP_MIN_M = 1.0
SKEW_LINE_EDGE_M = 0.3  # extra deck lines stay this far inside the pavement edge
TOE_SKEW_MAX = 0.5  # |tan skew| * |du/ds| of an active embankment toe stays at or below this (D-161)
TOE_SAMPLE_M = 1.0

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
        thickness = support.thickness_m if support.thickness_m is not None else support.cap_width_m
        if thickness <= 0.0:
            raise ValueError(f"support {support.id}: abutment thickness must be positive")
        if abs(support.station - structure.s_start) <= abs(support.station - structure.s_end):
            return structure.s_start, structure.s_start + thickness
        return structure.s_end - thickness, structure.s_end
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


def end_limited(t: float, s: float, length: float, umax: float) -> float:
    """Skew squared off toward the road's ends, so no station line leaves the road (FR-SG-2, D-118).

    A deck end at (or its skew ramp past) a road end would put the line s0 + t u outside
    [0, length] within the lateral reach. |t| is capped at the distance to the nearer end
    over RAMP_FACTOR * umax, which ramps to 0 at the end no faster than a skew ramp, so
    neighbouring lines never cross.
    """
    lim = max(min(s, length - s), 0.0) / (RAMP_FACTOR * umax)
    return max(-lim, min(t, lim))


def toe_skew_limit(samples: list[tuple[float, float, float]], sweep: list[float], umax: float, tmax: float) -> list[float]:
    """Largest |t| each station line in ``sweep`` may take near steep embankment toes (FR-SG-2, D-161).

    ``samples``: (station, |du/ds|, |u|) along the active toes. A line s0 + t u meets a toe
    at s0 + t u_toe; where |t| |du/ds| nears 1 that intersection is ill conditioned and
    jumps between rows. A line with |t| <= L meets the toe only within L |u| of s0, so a
    sample bounds L by max(TOE_SKEW_MAX / rate, |s0 - station| / |u|). The bounds then grow
    along the road no faster than a skew ramp, so neighbouring lines never cross.
    """
    caps = [math.inf] * len(sweep)
    for s, rate, u in samples:
        cap = TOE_SKEW_MAX / rate
        if cap >= tmax or u <= 0.0:
            continue
        reach = tmax * u
        for k in range(bisect.bisect_left(sweep, s - reach), bisect.bisect_right(sweep, s + reach)):
            caps[k] = min(caps[k], max(cap, abs(sweep[k] - s) / u))
    slope = 1.0 / (RAMP_FACTOR * umax)
    for k in range(1, len(sweep)):
        caps[k] = min(caps[k], caps[k - 1] + slope * (sweep[k] - sweep[k - 1]))
    for k in range(len(sweep) - 2, -1, -1):
        caps[k] = min(caps[k], caps[k + 1] + slope * (sweep[k + 1] - sweep[k]))
    return caps


def toe_samples(toes: list[tuple[Callable[[float], float], float, float]], windows: list[tuple[float, float]], tmax: float) -> list[tuple[float, float, float]]:
    """(station, |du/ds|, |u|) every TOE_SAMPLE_M along each toe (func, s0, s1) where skewed lines can meet it."""
    out = []
    for func, a, b in toes:
        for w0, w1 in windows:
            lo, hi = max(a, w0), min(b, w1)
            if hi - lo < 2.0 * TOE_SAMPLE_M:
                continue
            n = math.ceil((hi - lo) / TOE_SAMPLE_M)
            ss = [lo + (hi - lo) * k / n for k in range(n + 1)]
            us = [func(s) for s in ss]
            for k in range(n):
                rate = abs(us[k + 1] - us[k]) / (ss[k + 1] - ss[k])
                if rate * tmax > TOE_SKEW_MAX:
                    out.append((0.5 * (ss[k] + ss[k + 1]), rate, max(abs(us[k]), abs(us[k + 1]))))
    return out


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


def spacing_cap(tol: float) -> float:
    """Longest station spacing: 10 m at 1 cm, growing with sqrt(tol) for coarser LODs (FR-AS-1).

    The square root keeps the cap in step with the curvature limit sqrt(8 tol R).
    """
    return MAX_STATION_SPACING_M * math.sqrt(max(tol, SPACING_REF_TOL_M) / SPACING_REF_TOL_M)


def max_step(road: Road, a: float, b: float, tol: float, umax: float, sides: Callable[[float, float], tuple[float, float]] | None = None) -> float:
    """Longest station spacing in (a, b) that keeps chord deviation under tol (FR-SG-2).

    (a, b) contains no parameter change, so curvature, grade change and cross-slope
    rate are linear or constant inside it. ``sides``: lateral reach on the left and
    the right of the lines over (a, b); only the side toward the centre of curvature
    can fold over it.
    """
    step = spacing_cap(tol)
    eps = 1e-6 * (b - a)
    sa, sb = a + eps, b - eps
    ks = [road.alignment.evaluate(sa)[3], road.alignment.evaluate(sb)[3]]
    k = max(abs(v) for v in ks)
    if k > 1e-12:
        reach = sides(a, b) if sides is not None else (umax, umax)
        inner = max(reach[0] if v > 0.0 else reach[1] for v in ks)
        # ``sides`` pads the reach; an actual fold is caught on the grid (D-142), so the padded reach only shortens the step.
        denom = max(1.0 - k * inner, 0.05)
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


def _max_rate(pw, a: float, b: float) -> float:
    """Largest |d value / ds| of a piecewise-linear parameter over [a, b]."""
    knots = pw.knots
    return max((abs(q.value - p.value) / (q.s - p.s) for p, q in pairwise(knots) if q.s > a and p.s < b and q.s > p.s), default=0.0)


def skew_offsets(road: Road, structure: Structure, side: str, tol: float) -> list[float]:
    """Offsets of extra u-lines on one side of a skewed deck (FR-SG-2).

    A skewed station line s = s0 + t u runs along the road as it crosses it, so the
    surface bends along it by z'' = t^2 p'' + 2 t dslope/ds (profile curvature and the
    cross-slope rate). Station spacing cannot reduce that chord error; lines along the
    road every sqrt(8 (tol / 2) / z'') do. The lines stay inside the narrowest pavement
    edge so they never cross it; none are needed on a square or gently warped deck.
    """
    t = max((abs(math.tan(sp.skew_rad)) for sp in structure.supports), default=0.0)
    if t == 0.0:
        return []
    a, b = structure.s_start, structure.s_end
    xs = road.cross_section
    slope, width = (xs.slope_left, xs.width_left) if side == "left" else (xs.slope_right, xs.width_right)
    curv = max((abs(e.g1 - e.g0) / e.length for e in road.profile.elements if e.length > 0.0 and e.s0 < b and e.s0 + e.length > a), default=0.0)
    bend = t * t * curv + 2.0 * t * _max_rate(slope, a, b)
    if bend <= 1e-12:
        return []
    h = math.sqrt(8.0 * 0.5 * tol / bend)
    narrow = min([width(a), width(b)] + [k.value for k in width.knots if a < k.s < b])
    full = narrow + structure.deck.overhang(side)
    n = math.ceil(full / h)
    sign = 1.0 if side == "left" else -1.0
    return [sign * full * k / n for k in range(1, n) if full * k / n < narrow - SKEW_LINE_EDGE_M]


def kinks(road: Road, umax: float) -> list[tuple[float, float, float, float]]:
    """Stations where an edge or surface line changes slope: (s, size of the change in m/m, u range).

    A skewed station line meets these kinks away from a vertex except on the reference
    line, so spacing near them is limited instead (chord error <= jump * spacing / 4).
    A kink bends only the lines on its own side of the reference line: a width kink the
    edge and everything outside it, a cross-slope kink that side's surface (D-122).
    """
    xs = road.cross_section
    out = []
    for pw, scale, lo, hi in (
        (xs.width_left, 1.0, 0.0, umax),
        (xs.width_right, 1.0, -umax, 0.0),
        (xs.slope_left, umax, 0.0, umax),
        (xs.slope_right, umax, -umax, 0.0),
    ):
        knots = pw.knots
        rates = [0.0] + [(b.value - a.value) / (b.s - a.s) for a, b in pairwise(knots)] + [0.0]
        for k, knot in enumerate(knots):
            jump = abs(rates[k + 1] - rates[k]) * scale
            if jump > 1e-12:
                out.append((knot.s, jump, lo, hi))
    return out


def stations(
    road: Road,
    breakpoints: list[float],
    tol: float,
    umax: float,
    knots: list[SkewKnot],
    spans: list[tuple[float, float]] = (),  # type: ignore[assignment]
    sides: Callable[[float, float], tuple[float, float]] | None = None,
) -> list[float]:
    """Sweep stations; inside ``spans`` (structures) the tolerance never exceeds STRUCTURE_TOL_M.

    A deck on a crest curve sags below the model by up to the chord tolerance, so
    structures keep their finest tessellation at every LOD: the deck top and soffit,
    and the void under them, are preserved (FR-AS-1).
    """
    skewed = [k for k in knots if k.t != 0.0]
    kinked = kinks(road, umax) if skewed else []
    out = [breakpoints[0]]
    for a, b in pairwise(breakpoints):
        mid = 0.5 * (a + b)
        tol_ab = min(tol, STRUCTURE_TOL_M) if any(s0 <= mid <= s1 for s0, s1 in spans) else tol
        step = max_step(road, a, b, tol_ab, umax, sides)
        ta, tb = skew_at(knots, a), skew_at(knots, b)
        if skewed and (abs(ta) > 0.0 or abs(tb) > 0.0):
            for s, jump, lo, hi in kinked:
                # Lines s0 + t u over (a, b) meet the kink's side between these stations.
                ends = [s0 + t * u for s0, t in ((a, ta), (b, tb), (a, tb), (b, ta)) for u in (lo, hi)]
                if min(ends) < s < max(ends):
                    step = min(step, 4.0 * tol_ab / jump)
        n = max(1, math.ceil((b - a) / step - 1e-9))
        out.extend(a + (b - a) * k / n for k in range(1, n))
        out.append(b)
    return out
