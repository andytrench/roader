"""Structure ends from data (FR-VC-4, FR-CR-3).

Tagged bridge extents are ignored. Along each fitted road the ground just beyond
both pavement edges is compared with the road surface: where the lowest returns
in a band 1.5 to 4 m outside the edge are more than ``SEPARATION_M`` below the
surface, the carrying sheet has left the ground and a void or a lower sheet is
beside it. Each side gives a run; on a skewed bridge the two sides' runs are
shifted by +-offset * tan(skew), so the centerline ends are their means.
Returns classified as bridge deck (ASPRS 17) on the road surface give a second,
sharper estimate. It is used only where the geometric run confirms a structure,
since vendor classes are untrusted evidence on their own (FR-IN-3), and there
it gives the ends (D-031).

A road leaving or joining another on that road's deck rides the same deck
(``absorb_diverging``): it gets no structure of its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from roader.evidence import Evidence
from roader.fit.road import RoadFit
from roader.model.schema import Flag, Link, Road

STEP_M = 1.0
BAND_INNER_M = 1.5
BAND_OUTER_M = 4.0
SEPARATION_M = 2.5
BAND_PERCENTILE = 10.0
BAND_MIN_POINTS = 4
GAP_BRIDGE_M = 4.0  # elevated runs separated by less than this are one run
MIN_STRUCTURE_M = 18.0
DECK_CLASS = 17
DECK_CLASS_MIN_POINTS = 3
DECK_SURFACE_TOL_M = 0.5
CLASS17_AGREE_M = 5.0
ABOVE_SURFACE_M = 1.5  # returns higher than this above the road are something over it (parapets stay in)


@dataclass
class StructureExtent:
    road_id: str
    s_start: float
    s_end: float
    method: str  # "class17" | "sheet_break" | "class17_only" | "rule"
    geom: tuple[float, float] | None
    class17: tuple[float, float] | None
    one_sided: bool = False
    # FR-CR-3: the (s_start, s_end, method) runs chained into this extent; empty when it is one run.
    chained: list[tuple[float, float, str]] = field(default_factory=list)
    clipped: bool = False  # an end was clamped to the end of the road

    @property
    def length(self) -> float:
        return self.s_end - self.s_start


METHOD_TRUST = {"class17": 3, "sheet_break": 2, "class17_only": 1, "rule": 0}
REJOIN_M = 5.0  # the carrying sheet is back on the ground over at least this length


def rejoins_ground(drop: tuple[np.ndarray, np.ndarray, np.ndarray], s0: float, s1: float) -> bool:
    """True when, somewhere in (s0, s1), the road is back on the ground for ``REJOIN_M``.

    A station is on the ground when at least one side was seen and no side
    drops by more than ``SEPARATION_M`` (the inverse of the elevated test in
    ``find_extents``; an embankment between two bridges reads as ground, a twin
    carriageway's deck beside one side does not hide the void beyond the other).
    Unseen stations (both sides NaN, e.g. under a higher deck) are not evidence
    of rejoining.
    """
    st, dl, dr = drop
    m = (st > s0) & (st < s1)
    if not m.any():
        return False
    left, right = dl[m], dr[m]
    seen = np.isfinite(left) | np.isfinite(right)
    deep = (np.nan_to_num(left, nan=0.0) < -SEPARATION_M) | (np.nan_to_num(right, nan=0.0) < -SEPARATION_M)
    ground = seen & ~deep
    run = best = 0
    for g in ground.tolist():
        run = run + 1 if g else 0
        best = max(best, run)
    return best * STEP_M >= REJOIN_M


def chain_extents(
    extents: list[StructureExtent],
    drops: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    lengths: dict[str, float] | None = None,
) -> list[StructureExtent]:
    """FR-CR-3: one candidate structure per run of a road that never rejoins the ground.

    Extents on one road that overlap, or whose gap never shows the carrying
    sheet back on the ground (``rejoins_ground``), become one extent: a viaduct
    over several roads is one structure, not one per crossing. Each end keeps the
    station and method of the run it came from; the merged method is the less
    trusted of the two end methods, and ``chained`` lists the runs. With
    ``lengths`` (road id to length), ends are clamped to the road and the
    extent is marked ``clipped`` (a deck running past where the road was cut).
    """
    out: list[StructureExtent] = []
    if lengths:
        extents = [c for c in (_clamped(e, lengths[e.road_id]) for e in extents) if c.length >= STEP_M]
    for road_id in sorted({e.road_id for e in extents}):
        runs = sorted((e for e in extents if e.road_id == road_id), key=lambda e: (e.s_start, e.s_end))
        groups: list[list[StructureExtent]] = []
        for e in runs:
            if groups:
                last_end = max(g.s_end for g in groups[-1])
                if e.s_start <= last_end + GAP_BRIDGE_M or not rejoins_ground(drops[road_id], last_end, e.s_start):
                    groups[-1].append(e)
                    continue
            groups.append([e])
        for g in groups:
            if len(g) == 1:
                out.append(g[0])
                continue
            first = min(g, key=lambda e: (e.s_start, -METHOD_TRUST[e.method]))
            last = max(g, key=lambda e: (e.s_end, METHOD_TRUST[e.method]))
            method = min((first.method, last.method), key=lambda m: METHOD_TRUST[m])
            geom = [e.geom for e in g if e.geom]
            c17 = [e.class17 for e in g if e.class17]
            out.append(
                StructureExtent(
                    road_id,
                    first.s_start,
                    last.s_end,
                    method,
                    (min(a for a, _ in geom), max(b for _, b in geom)) if geom else None,
                    (min(a for a, _ in c17), max(b for _, b in c17)) if c17 else None,
                    first.one_sided or last.one_sided,
                    [(e.s_start, e.s_end, e.method) for e in g],
                    first.clipped or last.clipped,
                )
            )
    return sorted(out, key=lambda e: (e.road_id, e.s_start))


SEAM_REACH_M = 2.0  # a run starting this close to its road's seam starts on the parent's deck
SHARED_OVERRUN_M = 10.0  # a run ending this far past the parent's deck end is a deck of its own


@dataclass
class Share:
    """A road riding on another road's deck from its seam with that road (FR-CR-3)."""

    parent_road: str
    parent_s: float  # seam station on the parent
    road_id: str
    s_seam: float  # seam station on this road
    toward: float  # +1: the road leaves the seam toward increasing s (a diverge), -1: arrives there (a merge)


def absorb_diverging(
    extents: list[StructureExtent], links: list[Link], roads: dict[str, Road]
) -> tuple[list[StructureExtent], list[Share], list[Flag]]:
    """FR-CR-3: a road diverging from (or merging into) another on that road's deck shares the deck.

    Every diverge or merge whose seam lies on a structure run of the parent gives a
    ``Share``: the child rides the parent's deck from its seam. A run of the child
    starting at that seam is dropped, so the child gets no deck or abutments of its
    own. A run reaching more than ``SHARED_OVERRUN_M`` past the parent's deck end is
    a deck of its own continuing from the shared one; it is kept, no share is made,
    and it is flagged for review.
    """
    dropped: set[int] = set()
    shares: list[Share] = []
    flags: list[Flag] = []
    for lk in sorted(links, key=lambda k: k.id):
        if lk.kind == "diverge":
            parent, p_end, p_s, child, c_end, c_s = lk.from_road, lk.from_end, lk.from_s, lk.to_road, lk.to_end, lk.to_s
        elif lk.kind == "merge":
            parent, p_end, p_s, child, c_end, c_s = lk.to_road, lk.to_end, lk.to_s, lk.from_road, lk.from_end, lk.from_s
        else:
            continue
        if parent not in roads or child not in roads:
            continue
        road = roads[child]
        seam = c_s if c_s is not None else (0.0 if c_end == "start" else road.length)
        p_seam = p_s if p_s is not None else (0.0 if p_end == "start" else roads[parent].length)
        host = [p for p in extents if p.road_id == parent and p.s_start <= p_seam <= p.s_end]
        if not host:
            continue
        p = host[0]
        toward = 1.0 if c_end == "start" else -1.0
        own = [
            k
            for k, e in enumerate(extents)
            if e.road_id == child and (e.s_start <= seam + SEAM_REACH_M if toward > 0 else e.s_end >= seam - SEAM_REACH_M)
            and (e.s_end >= seam - SEAM_REACH_M if toward > 0 else e.s_start <= seam + SEAM_REACH_M)
        ]
        long = []
        for k in own:
            e = extents[k]
            far = e.s_end if toward > 0 else e.s_start
            ps, _pu = roads[parent].alignment.project(*road.alignment.offset_point(far, 0.0))
            overrun = max(ps - p.s_end, p.s_start - ps)
            if overrun > SHARED_OVERRUN_M:
                long.append((e, overrun))
        if long:
            for e, overrun in long:
                flags.append(
                    Flag(
                        "FR-CR-3:diverges_on_deck",
                        child,
                        "FR-CR-3",
                        f"{e.method} run s={e.s_start:.1f}-{e.s_end:.1f} starts on the deck of {parent} (s={p.s_start:.1f}-{p.s_end:.1f}) and runs {overrun:.1f} m past it; kept as its own structure",
                    )
                )
            continue
        dropped.update(own)
        shares.append(Share(parent, p_seam, child, seam, toward))
    return [e for k, e in enumerate(extents) if k not in dropped], shares, flags


def _clamped(e: StructureExtent, length: float) -> StructureExtent:
    a, b = max(e.s_start, 0.0), min(e.s_end, length)
    if a == e.s_start and b == e.s_end:
        return e
    return StructureExtent(e.road_id, a, b, e.method, e.geom, e.class17, e.one_sided, list(e.chained), True)


def _runs(mask: np.ndarray, s: np.ndarray, bridge: float, min_len: float, unknown: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Runs of True stations. Gaps up to ``bridge`` m, or gaps with no data at all
    (``unknown``, e.g. a held-out block), do not end a run."""
    out: list[list[float]] = []
    idx = np.flatnonzero(mask)
    for i in idx:
        if out:
            j = int(np.searchsorted(s, out[-1][1]))
            gap_unknown = unknown is not None and bool(np.all(unknown[j + 1 : i]))
            if s[i] - out[-1][1] <= bridge or gap_unknown:
                out[-1][1] = s[i]
                continue
        out.append([s[i], s[i]])
    return [(a, b) for a, b in out if b - a >= min_len]


def side_drop(fit: RoadFit, ev: Evidence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stations and the ground drop beyond the left and right edges (negative = below), NaN if empty.

    Returns above the road surface are a higher deck or canopy, not what the road
    stands on (FR-RS-2), so they never count: beside a stacked deck where only
    the decks above were seen, the drop is unknown, not zero.
    """
    road = fit.road
    lp = fit.points
    length = road.length
    st = np.arange(0.0, length + 1e-9, STEP_M)
    k = np.clip(np.rint(lp.s / STEP_M).astype(np.int64), 0, len(st) - 1)
    xs = road.cross_section
    wl = np.interp(lp.s, [kn.s for kn in xs.width_left.knots], [kn.value for kn in xs.width_left.knots])
    wr = np.interp(lp.s, [kn.s for kn in xs.width_right.knots], [kn.value for kn in xs.width_right.knots])
    zc = np.array([road.profile.z(float(v)) for v in st])
    below = lp.z <= zc[k] + ABOVE_SURFACE_M
    out = []
    for sign, w in ((1.0, wl), (-1.0, wr)):
        d = sign * lp.u - w
        band = (d >= BAND_INNER_M) & (d <= BAND_OUTER_M) & below
        kb, zb = k[band], lp.z[band]
        drop = np.full(len(st), np.nan)
        if kb.size:
            order = np.lexsort((zb, kb))
            kb, zb = kb[order], zb[order]
            uk, start, cnt = np.unique(kb, return_index=True, return_counts=True)
            q = zb[start + np.floor(BAND_PERCENTILE / 100.0 * (cnt - 1)).astype(np.int64)]
            good = cnt >= BAND_MIN_POINTS
            drop[uk[good]] = q[good] - zc[uk[good]]
        out.append(drop)
    return st, out[0], out[1]


def deck_class_runs(fit: RoadFit, ev: Evidence, skews: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    road = fit.road
    lp = fit.points
    cls = ev.classification[lp.index]
    xs = road.cross_section
    wl = np.interp(lp.s, [kn.s for kn in xs.width_left.knots], [kn.value for kn in xs.width_left.knots])
    wr = np.interp(lp.s, [kn.s for kn in xs.width_right.knots], [kn.value for kn in xs.width_right.knots])
    on = (cls == DECK_CLASS) & (lp.u <= wl) & (lp.u >= -wr)
    if on.sum() < DECK_CLASS_MIN_POINTS:
        return []
    idx = np.flatnonzero(on)
    surf = np.array([road.surface_z(float(a), float(b)) for a, b in zip(lp.s[idx], lp.u[idx], strict=True)])
    idx = idx[np.abs(lp.z[idx] - surf) <= DECK_SURFACE_TOL_M]
    if idx.size < DECK_CLASS_MIN_POINTS:
        return []
    st = np.arange(0.0, road.length + 1e-9, STEP_M)
    counts = np.bincount(np.clip(np.rint(lp.s[idx] / STEP_M).astype(np.int64), 0, len(st) - 1), minlength=len(st))
    # Only returns at this road's height say whether its surface was seen; a deck
    # above it (FR-RS-2) leaves the stations unobserved, so the run is not ended there.
    pz = np.interp(lp.s, st, [road.profile.z(float(v)) for v in st])
    on_road = (lp.u <= wl) & (lp.u >= -wr) & (np.abs(lp.z - pz) <= ABOVE_SURFACE_M)
    any_pts = np.bincount(np.clip(np.rint(lp.s[on_road] / STEP_M).astype(np.int64), 0, len(st) - 1), minlength=len(st))
    sparse = any_pts < 0.2 * max(float(np.median(any_pts[any_pts > 0])) if (any_pts > 0).any() else 0.0, 1.0)
    runs = _runs(counts >= DECK_CLASS_MIN_POINTS, st, GAP_BRIDGE_M + 1.0, MIN_STRUCTURE_M * 0.5, sparse)
    # Run ends at the outermost deck-class returns, not at the station grid. On a
    # skewed deck the corners reach past the centerline ends by u tan(skew): each
    # return is moved to its centerline station first. The skew sign is the one
    # that gives the crisper (shorter) run, so no sign convention is assumed.
    out = []
    for a, b in runs:
        m = (lp.s[idx] >= a - STEP_M) & (lp.s[idx] <= b + STEP_M)
        s, u = lp.s[idx][m], lp.u[idx][m]
        inside = [k for c, k in skews or [] if a - 5.0 <= c <= b + 5.0]
        t = math.tan(float(np.mean(inside))) if inside else 0.0
        ends = min(
            ((float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))) for v in (s - u * t, s + u * t)),
            key=lambda e: e[1] - e[0],
        )
        out.append(ends)
    return out


def _side_runs(mask: np.ndarray, unknown: np.ndarray, st: np.ndarray) -> list[tuple[float, float, bool, bool]]:
    """Elevated runs on one side with flags telling whether each end touches missing data."""
    out = []
    for a, b in _runs(mask, st, GAP_BRIDGE_M, 0.5 * MIN_STRUCTURE_M, unknown):
        ia, ib = int(np.searchsorted(st, a)), int(np.searchsorted(st, b))
        cen_a = ia == 0 or bool(unknown[max(ia - 1, 0)])
        cen_b = ib >= len(st) - 1 or bool(unknown[min(ib + 1, len(st) - 1)])
        out.append((a, b, cen_a, cen_b))
    return out


def _combine(left, right, offset: float, skew: float) -> tuple[float, float, bool]:
    """Centerline ends from the two sides' runs.

    A side run at lateral offset u sees the abutment line at s_c + u tan(skew)
    (abutments parallel to the crossed road), so each uncensored side end gives
    an estimate of the centerline end; censored ends (touching missing data) are
    used only when nothing else is available.
    """
    shift = offset * math.tan(skew)
    ends = []
    for k in (0, 1):
        good, poor = [], []
        for run, sign in ((left, 1.0), (right, -1.0)):
            if run is None:
                continue
            value = run[k] - sign * shift
            (poor if run[2 + k] else good).append(value)
        pick = good or poor
        ends.append(float(np.mean(pick)))
    return ends[0], ends[1], left is None or right is None


def find_extents(fit: RoadFit, ev: Evidence, skews: list[tuple[float, float]] | None = None) -> list[StructureExtent]:
    """Structure runs along one road. ``skews``: (station, skew) of crossings this road carries."""
    st, drop_l, drop_r = side_drop(fit, ev)
    elev_l = np.nan_to_num(drop_l, nan=0.0) < -SEPARATION_M
    elev_r = np.nan_to_num(drop_r, nan=0.0) < -SEPARATION_M
    runs_l = _side_runs(elev_l, np.isnan(drop_l), st)
    runs_r = _side_runs(elev_r, np.isnan(drop_r), st)
    xs = fit.road.cross_section
    geom: list[tuple[float, float, bool]] = []
    used_r: set[int] = set()
    pairs = []
    for run in runs_l:
        partners = [j for j, other in enumerate(runs_r) if other[0] <= run[1] + GAP_BRIDGE_M and other[1] >= run[0] - GAP_BRIDGE_M]
        used_r.update(partners)
        if partners:
            c = min(runs_r[j][0] for j in partners)
            d = max(runs_r[j][1] for j in partners)
            cen_c = runs_r[min(partners, key=lambda j: runs_r[j][0])][2]
            cen_d = runs_r[max(partners, key=lambda j: runs_r[j][1])][3]
            pairs.append((run, (c, d, cen_c, cen_d)))
        else:
            pairs.append((run, None))
    for j, run in enumerate(runs_r):
        if j not in used_r:
            pairs.append((None, run))
    for left, right in pairs:
        a0 = min(r[0] for r in (left, right) if r is not None)
        b0 = max(r[1] for r in (left, right) if r is not None)
        mid = 0.5 * (a0 + b0)
        skew = 0.0
        if skews:
            inside = [k for s, k in skews if a0 - 5.0 <= s <= b0 + 5.0]
            skew = float(np.mean(inside)) if inside else 0.0
        offset = 0.5 * (xs.width_left(mid) + xs.width_right(mid)) + 0.5 * (BAND_INNER_M + BAND_OUTER_M)
        a, b, one = _combine(left, right, offset, skew)
        if b - a >= MIN_STRUCTURE_M:
            geom.append((a, b, one))
    geom.sort()
    deck = deck_class_runs(fit, ev)
    skewed = deck_class_runs(fit, ev, skews) if skews else deck
    out = []
    used_deck: set[int] = set()
    for a, b, one in geom:
        match = [i for i, (c, d) in enumerate(deck) if c <= b and d >= a]
        c17 = (min(deck[i][0] for i in match), max(deck[i][1] for i in match)) if match else None
        used_deck.update(match)
        if c17 and abs(c17[0] - a) <= CLASS17_AGREE_M and abs(c17[1] - b) <= CLASS17_AGREE_M:
            out.append(StructureExtent(fit.road.id, c17[0], c17[1], "class17", (a, b), c17, one))
        elif c17:
            # D-031: off by more than CLASS17_AGREE_M, class 17 on the road surface still
            # wins: with one side unseen (a twin deck beside it, FR-RS-2) the side drop only
            # guesses the far end, and a two-sided sheet break runs on over wingwalls and
            # abutment fills. Read at the centerline, corrected for the skew.
            if len(skewed) == len(deck):
                c17 = (min(skewed[i][0] for i in match), max(skewed[i][1] for i in match))
            out.append(StructureExtent(fit.road.id, c17[0], c17[1], "class17", (a, b), c17, one))
        else:
            out.append(StructureExtent(fit.road.id, a, b, "sheet_break", (a, b), c17, one))
    for i, (c, d) in enumerate(deck):
        if i not in used_deck and d - c >= MIN_STRUCTURE_M:
            out.append(StructureExtent(fit.road.id, c, d, "class17_only", None, (c, d)))
    return sorted(out, key=lambda e: e.s_start)
