"""One road from evidence: registration, L1, L2 and L3 with per-element provenance.

Order (spec stages 4, 6, 7): track the input line onto its sheet, register it to
the measured pavement (twice, so the second pass measures on a line that already
follows the road), fit the alignment, re-measure everything on the alignment,
fit cross slopes and the profile jointly (two passes), then the edge lines.

A registered line that strays more than ``REGISTER_MAX_M`` from the input vector
has followed another pavement (a ramp, the other carriageway); that stretch is
bridged from its neighbours and its measured centers are ignored (flagged
``FR-VC-3:off_vector``). Junction anchors pin the alignment's ends to a parent
road (``roader.fit.links``, FR-AF-3).

Confidence of a measured element (FR-EX-2 input, recorded per element):
``0.95 * observed_fraction ** 0.5 / (1 + (rms / 0.10) ** 2) * strip_factor``,
where ``strip_factor`` is 0.8 when flight strips disagree by more than 5 cm and
were not corrected (FR-PN-2), else 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from roader.conflate.graph import RoadSeed
from roader.conflate.polyline import Polyline
from roader.conflate.register import (
    LinePoints,
    Profiles,
    cross_profiles,
    near_points,
    register_line,
)
from roader.conflate.track import Track, track_line
from roader.evidence import Evidence
from roader.fit import alignment as fa
from roader.fit import profile as fp
from roader.fit.section import (
    EdgeFit,
    fit_cross_slopes,
    fit_edges,
    infer_lanes,
    piecewise_from,
)
from roader.model.geometry import Knot, Piecewise
from roader.model.provenance import Provenance, Source
from roader.model.schema import CrossSection, Flag, Road
from roader.rules import RuleBook

REGISTER_PASSES = 2
SURFACE_BAND_M = 0.15
EDGE_MARGIN_M = 0.3
OVERRIDE = "FR-AF-4:override"
PIN_BLEND_M = 60.0
REGISTER_MAX_M = 6.0  # a registered centerline further than this from the input vector followed another pavement
GUARD_MARGIN_M = 10.0
HINT_STEP_M = 10.0
HINT_WEIGHT = 0.3


@dataclass
class RoadFit:
    seed: RoadSeed
    road: Road
    line: Polyline  # alignment sampled at 0.5 m
    track: Track
    profiles: Profiles
    points: LinePoints
    edges: EdgeFit | None
    vert: object
    flags: list[Flag] = field(default_factory=list)
    gaps: list[tuple[float, float]] = field(default_factory=list)
    # Surface returns the L2 fit used (station, offset, raw height), for the joint solve (FR-AF-3).
    surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    strip_factor: float = 1.0
    profile_flags: list[Flag] = field(default_factory=list)  # FR-AF-4 overrides of the current L2 elements


def confidence(observed: float, rms: float | None, strip_factor: float = 1.0) -> float:
    obs = min(max(observed, 0.0), 1.0)
    r = 0.0 if rms is None or not math.isfinite(rms) else rms
    return float(min(1.0, max(0.0, 0.95 * math.sqrt(obs) / (1.0 + (r / 0.10) ** 2) * strip_factor)))


def _stats(res: np.ndarray) -> tuple[float | None, float | None]:
    res = res[np.isfinite(res)]
    if res.size == 0:
        return None, None
    return float(np.sqrt(np.mean(res**2))), float(np.percentile(np.abs(res), 95.0))


def _within(seed_line: Polyline, xy: np.ndarray) -> np.ndarray:
    """Points within REGISTER_MAX_M of the input line, past its ends included."""
    s, u, _ok = seed_line.project(xy)
    s, u = np.nan_to_num(s, nan=-np.inf), np.nan_to_num(u, nan=np.inf)
    return (np.abs(u) <= REGISTER_MAX_M) & (s >= -REGISTER_MAX_M) & (s <= seed_line.length + REGISTER_MAX_M)


def _guard(reg: Polyline, seed_line: Polyline) -> Polyline:
    """The registered line, with stretches more than REGISTER_MAX_M off the input vector bridged.

    Input vectors are a few meters off (spec section 3); a registered line further
    away has followed another pavement (a ramp, the other carriageway). Those
    stretches take the offsets of the good stretches either side, interpolated
    along the input line (FR-VC-3: a smooth correction).
    """
    s, u, _ok = seed_line.project(reg.xy)
    good = _within(seed_line, reg.xy)
    if good.all():
        return reg
    # The way off and back is not a measurement either: keep clear of it.
    s_bad = np.sort(np.nan_to_num(s[~good], nan=-np.inf))
    pos = np.clip(np.searchsorted(s_bad, s), 1, len(s_bad) - 1) if len(s_bad) > 1 else np.zeros(len(s), dtype=np.int64)
    near_bad = np.minimum(np.abs(s - s_bad[pos - 1 if len(s_bad) > 1 else 0]), np.abs(s - s_bad[pos])) <= GUARD_MARGIN_M
    good &= ~near_bad
    if good.sum() < max(3, 0.3 * len(good)):
        return seed_line
    order = np.argsort(s[good], kind="stable")
    sg, ug = s[good][order], u[good][order]
    st = np.linspace(0.0, seed_line.length, max(2, math.ceil(seed_line.length / 2.0) + 1))
    return seed_line.shifted(st, np.interp(st, sg, ug))


def _pin(data_xy: np.ndarray, check_xy: np.ndarray, anchors) -> tuple[np.ndarray, np.ndarray]:
    """Move the measured centerline onto junction nodes, blended out over PIN_BLEND_M (FR-AF-3, FR-VC-3)."""
    line = Polyline(check_xy, step=0.5)
    s_check = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(check_xy, axis=0).T))])
    s_data, _u, _ok = line.project(data_xy)
    s_data = np.nan_to_num(s_data, nan=0.0)
    check, data = check_xy.copy(), data_xy.copy()
    for i, anchor in enumerate(anchors):
        if anchor is None:
            continue
        delta = np.asarray(anchor[:2], dtype=np.float64) - (check_xy[0] if i == 0 else check_xy[-1])
        for pts, s in ((check, s_check), (data, s_data)):
            d = s if i == 0 else s_check[-1] - s
            pts += np.clip(1.0 - d / PIN_BLEND_M, 0.0, 1.0)[:, None] * delta[None, :]
    return data, check


def fit_road(seed: RoadSeed, ev: Evidence, rules: RuleBook, *, strip_factor: float = 1.0, anchors=(None, None)) -> RoadFit:
    """``anchors``: per end, None or (x, y, heading or None) the alignment is pinned to (a junction node, FR-AF-3)."""
    flags: list[Flag] = []
    line = seed.line
    track = track_line(line, ev)
    prof = None
    data_xy = check_xy = None
    rejected, bridged = 0, False
    for _ in range(REGISTER_PASSES):
        lp = near_points(line, ev)
        prof = cross_profiles(lp, track.z_at, line.length)
        reg, _st, mid, weight = register_line(line, prof)
        has = weight > 0
        guarded = _guard(reg, seed.line)
        bridged = guarded is not reg
        if has.sum() >= 3:
            data_xy = line.offset_point(prof.s[has], mid[has])
            data_w = weight[has]
            if guarded is not reg:
                near = _within(seed.line, data_xy)
                rejected = max(rejected, int((~near).sum()))
                data_xy, data_w = data_xy[near], data_w[near]
                if len(data_xy) < 3:
                    data_xy = None
        reg = guarded
        check_xy = reg.xy
        line = reg
        track = track_line(line, ev)
    if bridged:
        flags.append(
            Flag("FR-VC-3:off_vector", seed.id, "FR-VC-3", f"registered line strayed over {REGISTER_MAX_M:.0f} m off the input vector; bridged, {rejected} measured centers ignored")
        )
        if data_xy is not None:
            # The bridged line stands in, weakly, where no measured center is left (a vector hint).
            hint = check_xy[:: int(HINT_STEP_M / 0.5)]
            d, _i = cKDTree(data_xy).query(hint)
            fill = d > fa.SUPPORT_M
            data_xy = np.vstack([data_xy, hint[fill]])
            data_w = np.concatenate([data_w, np.full(int(fill.sum()), HINT_WEIGHT)])
    if data_xy is None:
        flags.append(Flag("FR-VC-2:unsupported", seed.id, "FR-VC-2", "no pavement edges measured; alignment follows the input vector"))
        data_xy = seed.line.xy[::10]
        data_w = np.ones(len(data_xy))
        check_xy = seed.line.xy

    if any(a is not None for a in anchors):
        data_xy, check_xy = _pin(np.asarray(data_xy, dtype=np.float64), np.asarray(check_xy, dtype=np.float64), anchors)
    chain, arep = fa.fit_alignment(data_xy, data_w, check_xy, rules=rules, road_class=seed.road_class, pins=tuple(anchors))
    if arep.flags:
        flags.append(Flag("FR-AF-1", seed.id, "FR-AF-1", "; ".join(arep.flags)))
    alignment = fa.to_alignment(chain)
    aline = Polyline.from_alignment(alignment)
    length = alignment.length
    track = track_line(aline, ev)
    lp = near_points(aline, ev)
    prof = cross_profiles(lp, track.z_at, length)
    gaps = [(iv.s0, iv.s1) for iv in track.intervals(length) if iv.kind != "observed"]

    edges = fit_edges(prof, length, gaps)
    if edges is None:
        half, _ = rules.fill("lane.width", road_class=seed.road_class)
        edges = EdgeFit(np.array([0.0]), np.array([half]), np.array([0.0]), np.array([half]), math.nan, 0, 0.0)
        flags.append(Flag("FR-XS-1:no_edges", seed.id, "FR-XS-1", "pavement edges not measured; width by rule"))
    wl = np.interp(lp.s, edges.knots_left, edges.left)
    wr = np.interp(lp.s, edges.knots_right, edges.right)

    # Joint profile and cross slope, two passes (FR-AF-2, FR-XS-2).
    zt = track.z_at(lp.s)
    inside = (lp.u <= wl - EDGE_MARGIN_M) & (lp.u >= -wr + EDGE_MARGIN_M)
    crown, _ = rules.fill("cross_slope.crown", road_class=seed.road_class)
    sel = inside & (np.abs(lp.u) <= 1.5) & (np.abs(lp.z - zt) <= 0.3)
    vert, _pfit, _bins = fp.fit_profile(lp.s[sel], lp.z[sel], length, rules=rules, road_class=seed.road_class)
    slopes_l = (np.array([0.0]), np.array([-crown]))
    slopes_r = (np.array([0.0]), np.array([crown]))
    for _ in range(2):
        pz = vert.z(lp.s)
        slope = np.where(lp.u >= 0, np.interp(lp.s, *slopes_l), np.interp(lp.s, *slopes_r))
        band = SURFACE_BAND_M + 0.02 * np.abs(lp.u)
        sel = inside & (np.abs(lp.z - (pz + slope * lp.u)) <= band)
        slopes_l, slopes_r, slope_rms = fit_cross_slopes(lp.s[sel], lp.u[sel], lp.z[sel] - pz[sel], length, gaps, crown)
        slope = np.where(lp.u >= 0, np.interp(lp.s, *slopes_l), np.interp(lp.s, *slopes_r))
        zc = lp.z - slope * lp.u
        sel = inside & (np.abs(zc - pz) <= band)
        vert, _pfit, _bins = fp.fit_profile(lp.s[sel], zc[sel], length, rules=rules, road_class=seed.road_class)
    s_pts, z_pts = lp.s[sel], zc[sel]
    surface = (lp.s[sel].copy(), lp.u[sel].copy(), lp.z[sel].copy())

    # ---------------------------------------------------------------- provenance
    paved = float(np.median(np.interp(prof.s, edges.knots_left, edges.left) + np.interp(prof.s, edges.knots_right, edges.right)))
    vprovs, pflags = profile_provenance(seed.id, seed.road_class, vert, length, s_pts, z_pts, _bins[0], gaps, paved, rules, strip_factor)
    profile = fp.to_profile(vert, length, vprovs)

    aprovs = []
    _s_d, u_d, _ok = aline.project(data_xy)
    s_d = _s_d
    valid_s = prof.s[prof.valid]
    for el in fa.to_alignment(chain).elements:
        a, b = el.s0, el.s0 + el.length
        m = (s_d >= a) & (s_d <= b) & np.isfinite(u_d)
        rms, p95 = _stats(u_d[m])
        n_st = max(1, int(np.sum((prof.s >= a) & (prof.s <= b))))
        obs = min(1.0, float(np.sum((valid_s >= a) & (valid_s <= b))) / n_st)
        src = Source.MEASURED if obs >= 0.2 else Source.VECTOR
        prov = Provenance(src, confidence(obs, rms, strip_factor) if src == Source.MEASURED else 0.3, rms_m=rms, p95_m=p95, n_points=int(m.sum()), observed_fraction=obs)
        if el.kind != "line":
            for k in (el.k0, el.k1):
                if abs(k) > 1e-5:
                    ok, rule = rules.evaluate("curve.radius", 1.0 / abs(k), road_class=seed.road_class)
                    if not ok and f"{OVERRIDE}:curve.radius" not in prov.flags:
                        prov.flags.append(f"{OVERRIDE}:curve.radius")
                        flags.append(Flag(OVERRIDE, seed.id, "FR-AF-4", f"measured radius {1 / abs(k):.0f} m outside rule {rule.id} [{rule.low}, {rule.high}]"))
        aprovs.append(prov)
    alignment = fa.to_alignment(chain, aprovs)

    wprov = Provenance(
        Source.MEASURED if edges.n else Source.RULE,
        confidence(edges.observed_fraction, edges.rms_m, strip_factor) if edges.n else 0.3,
        rms_m=None if not math.isfinite(edges.rms_m) else edges.rms_m,
        n_points=edges.n,
        observed_fraction=edges.observed_fraction,
        rule_id=None if edges.n else "lane.width",
    )
    sprov = Provenance(Source.MEASURED, confidence(float(prof.valid.mean()), slope_rms, strip_factor), rms_m=slope_rms if math.isfinite(slope_rms) else None, n_points=int(sel.sum()))
    for side, (kn, vals) in (("left", slopes_l), ("right", slopes_r)):
        for v in vals:
            ok, rule = rules.evaluate("superelevation.rate", abs(float(v)), road_class=seed.road_class)
            if not ok:
                flags.append(Flag(OVERRIDE, seed.id, "FR-AF-4", f"measured {side} cross slope {v:.4f} outside rule {rule.id}"))
                if f"{OVERRIDE}:superelevation.rate" not in sprov.flags:
                    sprov.flags.append(f"{OVERRIDE}:superelevation.rate")
    thick, tprov = rules.fill("pavement.thickness.asphalt", road_class=seed.road_class)
    xs = CrossSection(
        width_left=piecewise_from(edges.knots_left, edges.left, wprov),
        width_right=piecewise_from(edges.knots_right, edges.right, wprov),
        slope_left=piecewise_from(*slopes_l, sprov),
        slope_right=piecewise_from(*slopes_r, sprov),
        pavement_thickness=Piecewise([Knot(0.0, thick, tprov)]),
        lanes=infer_lanes(float(np.median(edges.left)), float(np.median(edges.right)), seed.road_class, seed.oneway, rules, length),
    )
    sheets = sorted({int(x) for x in track.sheet if x >= 0})
    road = Road(
        id=seed.id,
        name=seed.name,
        road_class=seed.road_class,
        kind=seed.kind,
        oneway=seed.oneway,
        alignment=alignment,
        profile=profile,
        cross_section=xs,
        vector_refs=seed.vector_refs,
        support=track.intervals(length),
        sheet_ids=[f"sheet_{i}" for i in sheets],
    )
    for iv in road.support:
        if iv.kind == "unsupported":
            flags.append(Flag("FR-VC-2:unsupported", seed.id, "FR-VC-2", f"no supporting sheet over s={iv.s0:.1f}-{iv.s1:.1f}"))
    return RoadFit(seed, road, aline, track, prof, lp, edges, vert, flags, gaps, surface, strip_factor, pflags)


def profile_provenance(
    road_id: str,
    road_class: str,
    vert,
    length: float,
    s_pts: np.ndarray,
    z_pts: np.ndarray,
    bin_s: np.ndarray,
    gaps: list[tuple[float, float]],
    paved_width: float,
    rules: RuleBook,
    strip_factor: float = 1.0,
) -> tuple[list[Provenance], list[Flag]]:
    """Per L2 element: residual statistics, observed fraction, rule overrides (FR-AF-4, FR-AF-5, FR-OC-1).

    ``z_pts`` are cross-slope-corrected heights at stations ``s_pts``; ``bin_s`` the
    occupied 1 m profile bins.
    """
    flags: list[Flag] = []
    vprovs = []
    for a, b, kind in fp.element_ranges(vert, length):
        m = (s_pts >= a) & (s_pts <= b)
        res = z_pts[m] - vert.z(s_pts[m])
        res = res[np.abs(res) < 1.0]
        rms, p95 = _stats(res)
        obs = fp._observed_fraction(a, b, bin_s)
        prov = Provenance(
            Source.MEASURED,
            confidence(obs, rms, strip_factor),
            rms_m=rms,
            p95_m=p95,
            n_points=int(res.size),
            density_pts_m2=float(res.size / max((b - a) * paved_width, 1e-6)) if res.size else None,
            observed_fraction=obs,
        )
        if obs < 0.999 and any(g0 < b and g1 > a for g0, g1 in gaps):
            prov.flags.append("FR-OC-1:carried")
        g0, g1 = float(vert.grade(np.array([a + 1e-6]))[0]), float(vert.grade(np.array([b - 1e-6]))[0])
        for g in (g0, g1):
            ok, rule = rules.evaluate("grade", abs(g), road_class=road_class)
            if not ok and f"{OVERRIDE}:grade" not in prov.flags:
                prov.flags.append(f"{OVERRIDE}:grade")
                flags.append(Flag(OVERRIDE, road_id, "FR-AF-4", f"measured grade {g:.4f} outside rule {rule.id} [{rule.low}, {rule.high}] at s={a:.1f}-{b:.1f}"))
        if kind == "curve" and abs(g1 - g0) > 1e-6:
            kval = (b - a) / (100.0 * abs(g1 - g0))
            rid = "vcurve.k_crest" if g1 < g0 else "vcurve.k_sag"
            ok, rule = rules.evaluate(rid, kval, road_class=road_class)
            if not ok:
                prov.flags.append(f"{OVERRIDE}:{rid}")
                flags.append(Flag(OVERRIDE, road_id, "FR-AF-4", f"measured K {kval:.1f} outside rule {rid} [{rule.low}, {rule.high}] at s={a:.1f}-{b:.1f}"))
        vprovs.append(prov)
    return vprovs, flags

