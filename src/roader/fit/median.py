"""Medians and twin carriageways (FR-XS-4).

Twin carriageways are two one-way mainlines running in opposite directions with
each on the other's left. Every ``STATION_M`` along the first of a pair the
median is classified from the returns between the two inner pavement edges,
against the straight line joining the two edge heights:

- ``barrier``: linear raised returns (FR-RS-4 evidence) within ``BARRIER_EDGE_M``
  of an inner edge, or anywhere in a median narrower than ``WIDE_M`` (a pier
  standing in a wide median is not a median barrier); in a narrow median any
  returns more than ``BARRIER_DZ_M`` above the edge line count as barrier too;
- ``none``: edges closer than ``NONE_GAP_M`` and no barrier;
- ``depressed`` / ``raised``: the median ground is more than ``DEPRESSED_M``
  below / ``RAISED_M`` above that line; an unpaved median ``WIDE_M`` or wider is
  ``depressed`` whatever its relief (a natural median, drained to its low line);
- ``painted``: flush and as smooth as pavement (robust spread under
  ``SMOOTH_M``); flush but rough ground (grass) counts as ``depressed`` and the
  element carries ``FR-XS-4:flush_unpaved``.

Station kinds are smoothed by a majority over ``MAJORITY`` stations, runs
shorter than ``MIN_RUN_M`` join their longer neighbour, and each run becomes one
``Median`` on both roads (stations in each road's own frame), with the median
gap between inner edges as its width.

A divided highway drawn as one two-way way down its median (``split_divided``)
is split before fitting into two one-way roads on the two paved bands either
side of a non-paved or barrier gap, so each carriageway gets its own alignment,
profile and cross-section. The split is an L0 topology change and is flagged.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np

from roader.conflate.graph import EdgeSpan, RoadSeed, _bound, _route_label
from roader.conflate.polyline import Polyline
from roader.model.provenance import Provenance, Source
from roader.model.schema import Flag, Median, Road
from roader.model.serialize import stable_id

STATION_M = 10.0
WINDOW_M = 2.5
EDGE_INSET_M = 0.15
TWIN_MIN_M = 3.0  # centerline to centerline
TWIN_MAX_M = 60.0
TWIN_COS = -0.9
NONE_GAP_M = 0.5
WIDE_M = 12.0
BARRIER_EDGE_M = 3.0
BARRIER_DZ_M = 0.5
DEPRESSED_M = 0.25
RAISED_M = 0.10
SMOOTH_M = 0.05
MIN_POINTS = 5
BARRIER_MIN_POINTS = 5
MAJORITY = 5
MIN_RUN_M = 30.0
MIN_OVERLAP_M = 60.0
# One-way-mapped divided highways (split_divided)
SPLIT_REACH_M = 30.0
SPLIT_BIN_M = 0.5
SPLIT_PAVED_P = 0.5
SPLIT_MIN_BAND_M = 5.0
SPLIT_MIN_GAP_M = 0.5
SPLIT_FRACTION = 0.6
LINEAR = 1  # Evidence.layer_label of linear raised features
SURFACE = 0


def _points_near(ev, line: Polyline, reach: float):
    """(s, u, z, label) of every evidence return within ``reach`` of the line, transients excluded."""
    lo, hi = line.xy.min(axis=0) - reach, line.xy.max(axis=0) + reach
    xy = ev.xyz[:, :2]
    box = np.flatnonzero(np.all((xy >= lo) & (xy <= hi), axis=1) & ~ev.transient)
    s, u, ok = line.project(xy[box], max_dist=reach)
    idx = box[ok]
    return s[ok], u[ok], ev.xyz[idx, 2], ev.layer_label[ev.point_layer[idx]], ev.point_prob[idx]


# ---------------------------------------------------------------- twins


def _partners(a: Road, la: Polyline, others: list[tuple[Road, Polyline]]) -> list[tuple[float, Road | None, float, float]]:
    """Per station of ``a``: (s, partner road, partner station, partner centerline offset in a's frame)."""
    out = []
    for s in np.arange(0.5 * STATION_M, a.length, STATION_M):
        p = la.point(s)[0]
        t = np.array([math.cos(la.heading(s)[0]), math.sin(la.heading(s)[0])])
        n = np.array([-t[1], t[0]])
        best = (s, None, 0.0, math.inf)
        for b, lb in others:
            sb, _ub, ok = lb.project(p[None, :], max_dist=TWIN_MAX_M)
            if not ok[0]:
                continue
            q = lb.point(sb[0])[0]
            hb = lb.heading(sb[0])[0]
            if float(t @ np.array([math.cos(hb), math.sin(hb)])) > TWIN_COS:
                continue
            u = float((q - p) @ n)
            if TWIN_MIN_M <= u <= TWIN_MAX_M and u < best[3]:
                best = (s, b, float(sb[0]), u)
        out.append(best)
    return out


def _classify(gap, z, u, label, z_line, u0) -> tuple[str | None, int, bool]:
    """(kind or None if unobserved, returns used, flush_unpaved); ``u0`` is the near inner edge offset."""
    at_edge = (u - u0 <= BARRIER_EDGE_M) | (u0 + gap - u <= BARRIER_EDGE_M) | (gap < WIDE_M)
    barrier = (label == LINEAR) & at_edge
    if gap < WIDE_M:
        # A wall-like rise in a narrow median is a barrier even where the cells were not labelled linear.
        barrier |= z - z_line(u) > BARRIER_DZ_M
    if int(barrier.sum()) >= BARRIER_MIN_POINTS:
        return "barrier", int(barrier.sum()), False
    if gap < NONE_GAP_M:
        return "none", 0, False
    ground = label == SURFACE
    if ground.sum() < MIN_POINTS:
        return None, 0, False
    dz = z[ground] - z_line(u[ground])
    med = float(np.median(dz))
    if med < -DEPRESSED_M:
        return "depressed", int(ground.sum()), False
    if gap >= WIDE_M:
        return "depressed", int(ground.sum()), med > -DEPRESSED_M
    if med > RAISED_M:
        return "raised", int(ground.sum()), False
    spread = 1.4826 * float(np.median(np.abs(dz - med)))
    if spread <= SMOOTH_M:
        return "painted", int(ground.sum()), False
    return "depressed", int(ground.sum()), True


def _runs(rows: list[tuple[float, str, float, int, bool]], length: float) -> list[tuple[float, float, str, float, int, float, bool]]:
    """Station rows (s, kind, gap, n, flush) to runs (s0, s1, kind, width, n, agreement, flush)."""
    if not rows:
        return []
    kinds = [r[1] for r in rows]
    half = MAJORITY // 2
    smooth = []
    for i in range(len(rows)):
        window = Counter(kinds[max(0, i - half) : i + half + 1])
        top = max(window.values())
        smooth.append(kinds[i] if window[kinds[i]] == top else min(k for k, c in window.items() if c == top))
    groups: list[list[int]] = []
    for i in range(len(rows)):
        contiguous = groups and rows[i][0] - rows[groups[-1][-1]][0] <= 2.5 * STATION_M
        if contiguous and smooth[groups[-1][-1]] == smooth[i]:
            groups[-1].append(i)
        else:
            groups.append([i])
    # Short runs join the longer contiguous neighbour.
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for g, idx in enumerate(groups):
            span = rows[idx[-1]][0] - rows[idx[0]][0] + STATION_M
            if span >= MIN_RUN_M:
                continue
            nbrs = [h for h in (g - 1, g + 1) if 0 <= h < len(groups) and abs(rows[groups[h][0 if h > g else -1]][0] - rows[idx[-1 if h > g else 0]][0]) <= 2.5 * STATION_M]
            if not nbrs:
                continue
            h = max(nbrs, key=lambda k: (len(groups[k]), -k))
            for i in idx:
                smooth[i] = smooth[groups[h][0]]
            groups[h] = sorted(groups[h] + idx)
            del groups[g]
            changed = True
            break
    out = []
    for idx in groups:
        kind = smooth[idx[0]]
        s0 = max(0.0, rows[idx[0]][0] - 0.5 * STATION_M)
        s1 = min(length, rows[idx[-1]][0] + 0.5 * STATION_M)
        gaps = [rows[i][2] for i in idx]
        agree = sum(rows[i][1] == kind for i in idx) / len(idx)
        flush = any(rows[i][4] for i in idx if rows[i][1] == kind)
        out.append((s0, s1, kind, float(np.median(gaps)), sum(rows[i][3] for i in idx), agree, flush))
    return out


def _median(s0, s1, kind, width, n, agree, flush) -> Median:
    flags = ["FR-XS-4"] + (["FR-XS-4:flush_unpaved"] if flush and kind == "depressed" else [])
    prov = Provenance(Source.MEASURED, round(0.9 * agree, 6), n_points=int(n), observed_fraction=round(agree, 6), flags=flags)
    return Median(float(s0), float(s1), kind, max(0.0, float(width)), prov)


def classify_medians(fits, ev) -> list[Flag]:
    """Median elements on every twin carriageway pair (FR-XS-4), set in place on the fitted roads."""
    roads = sorted([f.road for f in fits if f.road.kind == "mainline" and f.road.oneway], key=lambda r: r.id)
    lines = {r.id: Polyline.from_alignment(r.alignment) for r in roads}
    flags: list[Flag] = []
    done: set[tuple[str, str]] = set()
    for a in roads:
        la = lines[a.id]
        others = [(b, lines[b.id]) for b in roads if b.id != a.id]
        partners = _partners(a, la, others)
        by_partner: dict[str, list] = {}
        for row in partners:
            if row[1] is not None:
                by_partner.setdefault(row[1].id, []).append(row)
        pts = None
        for bid in sorted(by_partner):
            rows_b = by_partner[bid]
            if (min(a.id, bid), max(a.id, bid)) in done or len(rows_b) * STATION_M < MIN_OVERLAP_M:
                continue
            done.add((min(a.id, bid), max(a.id, bid)))
            b = rows_b[0][1]
            if pts is None:
                pts = _points_near(ev, la, TWIN_MAX_M)
            ps, pu, pz, plab, _pp = pts
            rows = []
            for s, _b, sb, ub in rows_b:
                wa = a.cross_section.width_left(s)
                wb = b.cross_section.width_left(sb)
                gap = ub - wb - wa
                u0, u1 = wa + EDGE_INSET_M, ub - wb - EDGE_INSET_M
                m = (np.abs(ps - s) <= WINDOW_M) & (pu > u0) & (pu < u1) if u1 > u0 else np.zeros(len(ps), dtype=bool)
                za, zb = a.surface_z(s, wa), b.surface_z(sb, wb)

                def z_line(u, za=za, zb=zb, wa=wa, gap=gap):
                    return za + (u - wa) / max(gap, 1e-6) * (zb - za)

                kind, n, flush = _classify(gap, pz[m], pu[m], plab[m], z_line, wa)
                if kind is not None:
                    rows.append((s, kind, gap, n, flush))
            lb = lines[b.id]
            meds_a, meds_b = [], []
            for run in _runs(rows, a.length):
                meds_a.append(_median(*run))
                ends = lb.project(la.point(np.array([run[0], run[1]])))[0]
                s0b, s1b = float(np.nanmin(ends)), float(np.nanmax(ends))
                if s1b - s0b > 1.0:
                    meds_b.append(_median(s0b, s1b, *run[2:]))
            a.cross_section.medians = sorted(a.cross_section.medians + meds_a, key=lambda m: m.s0)
            b.cross_section.medians = sorted(b.cross_section.medians + meds_b, key=lambda m: m.s0)
            flags.extend(_overlaps(a) + _overlaps(b))
    return flags


def _overlaps(road: Road) -> list[Flag]:
    """A road between two opposite carriageways has two medians at a station; that is logged, not merged."""
    meds = road.cross_section.medians
    return [
        Flag("FR-XS-4:overlap", road.id, "FR-XS-4", f"medians overlap at s={b.s0:.1f}-{min(a.s1, b.s1):.1f}")
        for a, b in itertools.pairwise(meds)
        if b.s0 < a.s1 - 1e-6
    ]


# ---------------------------------------------------------------- one-way-mapped divided highways


def _bands(u: np.ndarray, paved: np.ndarray, raised: np.ndarray) -> tuple[float, float] | None:
    """Centers of the paved bands right and left of a central gap, or None if the section is not divided."""
    nb = round(2 * SPLIT_REACH_M / SPLIT_BIN_M)
    b = np.clip(np.floor((u + SPLIT_REACH_M) / SPLIT_BIN_M).astype(np.int64), 0, nb - 1)
    count = np.bincount(b, weights=paved.astype(np.float64), minlength=nb)
    total = np.bincount(b, minlength=nb)
    is_paved = (count >= 2) & (count >= 0.5 * np.maximum(total, 1))
    has_barrier = np.bincount(b, weights=raised.astype(np.float64), minlength=nb) >= 2
    uc = -SPLIT_REACH_M + (np.arange(nb) + 0.5) * SPLIT_BIN_M
    runs, start = [], None
    for i in range(nb + 1):
        on = i < nb and is_paved[i]
        if on and start is None:
            start = i
        if not on and start is not None:
            runs.append((start, i - 1))
            start = None
    wide = [(a, c) for a, c in runs if (c - a + 1) * SPLIT_BIN_M >= SPLIT_MIN_BAND_M]
    right = [r for r in wide if uc[r[1]] < 0.0 or uc[(r[0] + r[1]) // 2] < 0.0]
    left = [r for r in wide if uc[r[0]] > 0.0 or uc[(r[0] + r[1]) // 2] > 0.0]
    if not right or not left:
        return None
    r = max(right, key=lambda q: q[1])
    lft = min(left, key=lambda q: q[0])
    if r == lft:
        return None
    gap = (lft[0] - r[1] - 1) * SPLIT_BIN_M
    barrier = bool(has_barrier[r[1] : lft[0] + 1].any())
    if gap < SPLIT_MIN_GAP_M and not barrier:
        return None
    return float(0.5 * (uc[r[0]] + uc[r[1]])), float(0.5 * (uc[lft[0]] + uc[lft[1]]))


def split_divided(seeds: list[RoadSeed], ev) -> tuple[list[RoadSeed], list[Flag]]:
    """Two-way mainline seeds over a divided section become twin one-way seeds (FR-XS-4, L0 change logged)."""
    out, flags = [], []
    for seed in seeds:
        two_way = not seed.oneway or any(ref.tags.get("oneway") == "no" for ref in seed.vector_refs)
        if seed.kind != "mainline" or not two_way:
            out.append(seed)
            continue
        line = seed.line
        s, u, _z, lab, prob = _points_near(ev, line, SPLIT_REACH_M)
        paved = (lab == SURFACE) & (prob >= SPLIT_PAVED_P)
        raised = lab == LINEAR
        centers = []
        stations = np.arange(0.5 * STATION_M, line.length, STATION_M)
        observed = 0
        for st in stations:
            m = np.abs(s - st) <= WINDOW_M
            if m.sum() < MIN_POINTS:
                continue
            observed += 1
            bands = _bands(u[m], paved[m], raised[m])
            if bands is not None:
                centers.append(bands)
        if not observed or len(centers) < SPLIT_FRACTION * observed:
            out.append(seed)
            continue
        c_r = float(np.median([c[0] for c in centers]))
        c_l = float(np.median([c[1] for c in centers]))
        grid = np.linspace(0.0, line.length, max(2, math.ceil(line.length / 2.0) + 1))
        right = line.shifted(grid, np.full(len(grid), c_r))
        left_line = line.shifted(grid, np.full(len(grid), c_l))
        left = Polyline(left_line.xy[::-1])
        ref = seed.vector_refs[0].tags.get("ref") or seed.name
        for tag, pl, flip in (("right", right, False), ("left", left, True)):
            h = math.atan2(pl.xy[-1, 1] - pl.xy[0, 1], pl.xy[-1, 0] - pl.xy[0, 0])
            spans = [EdgeSpan(sp.source, sp.feature_id, pl.length - sp.s1, pl.length - sp.s0, sp.tags) if flip else sp for sp in seed.spans]
            out.append(
                RoadSeed(
                    id=stable_id("rd", seed.id, tag),
                    name=f"{_route_label(ref)} {_bound(h)}",
                    road_class=seed.road_class,
                    kind=seed.kind,
                    oneway=True,
                    line=pl,
                    vector_refs=seed.vector_refs,
                    spans=sorted(spans, key=lambda sp: sp.s0),
                )
            )
        flags.append(
            Flag(
                "FR-XS-4:split",
                seed.id,
                "FR-XS-4",
                f"{seed.name}: one two-way way over a divided section ({len(centers)}/{observed} stations); "
                f"modelled as twin carriageways at offsets {c_r:+.1f} m and {c_l:+.1f} m (L0 topology change)",
            )
        )
    return sorted(out, key=lambda sd: sd.id), flags


# ---------------------------------------------------------------- twin median split (L3)

SPLIT_STEP_M = 5.0
TWIN_REACH_M = 40.0  # D-075 envelope: within 40 m, within 35 degrees, under 1 m height difference
TWIN_PARALLEL_DEG = 35.0
TWIN_DZ_M = 1.0
FACE_WINDOW_M = 2.5  # barrier returns this close in station set the face at a station
FACE_MIN_POINTS = 3
FACE_INSET_M = 0.5  # a barrier face nearer either crown than this is not in the median
CAP_SMOOTH_M = 12.5
BARRIER_BELOW_M = 0.5
BARRIER_ABOVE_M = 2.0
SPLIT_TOL_M = 0.15


def split_twin_medians(fits, barrier_xyz: np.ndarray) -> list[Flag]:
    """No inner edge of twin one-way mainlines passes their median (FR-XS-1, FR-XS-4, D-063).

    For every pair of one-way mainlines running alongside within the D-075 envelope,
    each road's facing width is capped, station by station, at the face on its side
    of the median barrier when ``barrier_xyz`` (linear raised returns, standing on
    the road: from ``BARRIER_BELOW_M`` below to ``BARRIER_ABOVE_M`` above its surface)
    lie between the two, else at the line midway between the two inner edges where those overlap.
    Widths short of the cap are kept. Each changed side is flagged.
    """
    fits = sorted([f for f in fits if f.road.oneway and f.road.kind == "mainline"], key=lambda f: f.road.id)
    caps: dict[str, dict[int, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for a, b in itertools.permutations(fits, 2):
        res = _caps(a, b, barrier_xyz)
        if res is not None:
            side, st, cap = res
            caps.setdefault(a.road.id, {}).setdefault(side, []).append((st, cap))
    flags = []
    for f in fits:
        for side, parts in sorted(caps.get(f.road.id, {}).items()):
            if _apply(f.road, side, parts):
                msg = f"{'left' if side > 0 else 'right'} width capped at the median of its twin carriageway"
                flags.append(Flag("FR-XS-4:median_split", f.road.id, "FR-XS-4", msg))
                f.flags.append(flags[-1])
    return flags


def _caps(a, b, barrier_xyz: np.ndarray):
    """(facing side of a, stations, cap) where b runs alongside a; None when it never does."""
    ra, rb, la, lb = a.road, b.road, a.line, b.line
    st = np.arange(0.0, ra.length, SPLIT_STEP_M)
    pa = la.point(st)
    sb, ub, okb = lb.project(pa, max_dist=TWIN_REACH_M)
    if not okb.any():
        return None
    sb = np.clip(np.nan_to_num(sb), 0.0, rb.length)
    qb = lb.point(sb)
    sa2, ua, oka = la.project(qb, max_dist=TWIN_REACH_M)
    ua = np.nan_to_num(ua)
    parallel = np.abs(np.cos(la.heading(st) - lb.heading(sb))) >= math.cos(math.radians(TWIN_PARALLEL_DEG))
    za = np.array([ra.profile.z(float(s)) for s in st])
    dz = np.abs(za - np.array([rb.profile.z(float(s)) for s in sb]))
    near = okb & oka & parallel & (dz < TWIN_DZ_M) & (np.abs(ua) > 1.0) & (np.abs(np.nan_to_num(sa2) - st) < TWIN_REACH_M)
    if near.sum() < 3:
        return None
    side = 1 if np.median(ua[near]) > 0 else -1
    near &= np.sign(ua) == side
    dist = np.abs(ua)
    wa = _width(ra, side, st)
    side_b = 1 if np.median(np.nan_to_num(ub)[near]) > 0 else -1
    wb = _width(rb, side_b, sb)
    edge_b = dist - wb  # b's inner edge, measured from a's line toward b
    cap = np.where(wa > edge_b, 0.5 * (wa + edge_b), np.inf)
    if len(barrier_xyz):
        s_bar, u_bar, ok_bar = la.project(barrier_xyz[:, :2], max_dist=TWIN_REACH_M)
        s_bar = np.nan_to_num(s_bar)
        above = barrier_xyz[:, 2] - np.interp(s_bar, st, za)
        keep = ok_bar & (np.sign(u_bar) == side) & (above >= -BARRIER_BELOW_M) & (above <= BARRIER_ABOVE_M)
        s_bar, t_bar = s_bar[keep], np.abs(u_bar[keep])
        order = np.argsort(s_bar)
        s_bar, t_bar = s_bar[order], t_bar[order]
        lo = np.searchsorted(s_bar, st - FACE_WINDOW_M)
        hi = np.searchsorted(s_bar, st + FACE_WINDOW_M)
        for i in np.flatnonzero(near):
            t = t_bar[lo[i] : hi[i]]
            t = t[(t > FACE_INSET_M) & (t < dist[i] - FACE_INSET_M)]
            if len(t) >= FACE_MIN_POINTS:
                cap[i] = min(float(np.min(t)), cap[i]) if np.isfinite(cap[i]) else float(np.min(t))
    cap = np.where(near, cap, np.inf)
    if not np.isfinite(cap).any():
        return None
    return side, st, _smooth_cap(st, cap)


def _smooth_cap(st: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Running minimum-preserving median of the finite caps, so one noisy station does not notch the edge."""
    out = cap.copy()
    fin = np.isfinite(cap)
    for i in np.flatnonzero(fin):
        w = fin & (np.abs(st - st[i]) <= CAP_SMOOTH_M)
        out[i] = float(np.median(cap[w]))
    return out


def _width(road: Road, side: int, s: np.ndarray) -> np.ndarray:
    pw = road.cross_section.width_left if side > 0 else road.cross_section.width_right
    return np.interp(s, [k.s for k in pw.knots], [k.value for k in pw.knots])


def _apply(road: Road, side: int, parts) -> bool:
    """Width on ``side`` = min(width, caps); True when it changed by more than SPLIT_TOL_M anywhere."""
    from roader.fit.piecewise import simplify
    from roader.model.geometry import Knot, Piecewise

    xs = road.cross_section
    pw = xs.width_left if side > 0 else xs.width_right
    st = np.unique(np.concatenate([np.arange(0.0, road.length, SPLIT_STEP_M), [road.length], [k.s for k in pw.knots]]))
    st = st[(st >= 0.0) & (st <= road.length)]
    old = _width(road, side, st)
    cap = np.full(len(st), np.inf)
    for s_c, c in parts:
        fin = np.isfinite(c)
        if fin.sum() < 2:
            continue
        inside = (st >= s_c[fin][0]) & (st <= s_c[fin][-1])
        cap = np.where(inside, np.minimum(cap, np.interp(st, s_c[fin], c[fin])), cap)
    new = np.minimum(old, np.maximum(cap, 0.0))
    if float(np.max(old - new)) <= SPLIT_TOL_M:
        return False
    kn, vals = simplify(st, new, 0.5 * SPLIT_TOL_M)
    base = pw.knots[0].prov
    prov = Provenance(base.source, base.confidence, rms_m=base.rms_m, n_points=base.n_points, observed_fraction=base.observed_fraction, flags=[*base.flags, "FR-XS-4:median_split"])
    piece = Piecewise([Knot(float(k), float(v), prov) for k, v in zip(kn, vals, strict=True)])
    if side > 0:
        xs.width_left = piece
    else:
        xs.width_right = piece
    return True
