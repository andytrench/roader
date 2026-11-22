"""L3 cross-section estimation (FR-XS-1, FR-XS-2, FR-XS-3, FR-RS-4 barriers).

Edges come from the cross-profiles (``roader.conflate.register``) measured on the
fitted alignment and are fitted as edge lines along the road. Cross slope is
fitted per side as a piecewise-linear function of station from the residuals of
the surface returns against the profile, so banking changes only through
transitions. Lane counts and shoulders are inferred from the paved width and
the rule library and tagged as rule-sourced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roader.conflate.polyline import Polyline
from roader.conflate.register import (
    MAX_WIDTH_M,
    WIDTH_DEV_M,
    WIDTH_DEV_REL,
    WIDTH_WINDOW_M,
    Profiles,
)
from roader.evidence import Evidence
from roader.fit.piecewise import fit_piecewise, hat_basis, knot_stations, simplify
from roader.model.geometry import Knot, Piecewise
from roader.model.provenance import Provenance, Source
from roader.model.schema import Barrier, LaneGroup
from roader.model.serialize import stable_id
from roader.rules import RuleBook

WIDTH_KNOT_M = 50.0
WIDTH_TOL_M = 0.15
KNOT_SUPPORT = 3  # edge stations a width knot needs within half a knot spacing (D-063)
MIN_SIDE_M = 0.5  # the reference line lies on its pavement, so each edge is at least this far to its side
SLOPE_KNOT_M = 40.0
SLOPE_TOL = 0.003
SLOPE_PRIOR_WEIGHT = 5.0  # pulls thinly observed knots toward the crown rule, in units of points
BARRIER_REACH_M = 3.0  # raised features up to this far beyond a pavement edge belong to it
BARRIER_INSIDE_M = 1.0
MIN_BARRIER_M = 6.0
CURB_MAX_M = 0.35  # raised features no taller than this are curbs or raised walkways


@dataclass
class EdgeFit:
    knots_left: np.ndarray
    left: np.ndarray
    knots_right: np.ndarray
    right: np.ndarray  # positive distances
    rms_m: float
    n: int
    observed_fraction: float


def side_valid(prof: Profiles, values: np.ndarray) -> np.ndarray:
    """Stations where one side's edge distance is usable on its own.

    The edge must be found, on its own side of the centerline by ``MIN_SIDE_M``, and
    close to that side's local median, so an edge that ran into a driveway, a gore or
    a neighbouring pavement is left out while the other side still counts.
    """
    ok = np.isfinite(values) & (values >= MIN_SIDE_M) & (values <= MAX_WIDTH_M)
    if ok.sum() < 3:
        return ok
    vs, ss = values[ok], prof.s[ok]
    lo = np.searchsorted(ss, prof.s - WIDTH_WINDOW_M, side="left")
    hi = np.searchsorted(ss, prof.s + WIDTH_WINDOW_M, side="right")
    local = np.array([np.median(vs[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)])
    return ok & (np.abs(values - local) <= np.maximum(WIDTH_DEV_M, WIDTH_DEV_REL * local))


def supported_knots(knots: np.ndarray, s: np.ndarray, reach: float, least: int = KNOT_SUPPORT) -> np.ndarray:
    """Knots with at least ``least`` data stations within ``reach``: a knot without its own
    data is set by its neighbour's data alone, and two knots resting on one isolated
    edge split it between them, so either swings far beyond any measured value
    (FR-OC-1 tapers instead)."""
    if len(s) == 0:
        return knots[:1]
    s = np.sort(s)
    count = np.searchsorted(s, knots + reach, side="right") - np.searchsorted(s, knots - reach, side="left")
    near = count >= min(least, len(s))
    return knots[near] if near.any() else knots[:1]


def fit_edges(prof: Profiles, length: float, gaps: list[tuple[float, float]]) -> EdgeFit | None:
    """Edge lines per side (FR-XS-1), each from the stations where that side's edge is usable."""
    sides = []
    for values in (prof.left, -prof.right):
        ok = side_valid(prof, values)
        if ok.sum() < 3:
            return None
        sides.append((prof.s[ok], values[ok]))
    knots = knot_stations(length, WIDTH_KNOT_M, gaps)
    out = []
    residuals = []
    for s, values in sides:
        kn = supported_knots(knots, s, 0.5 * WIDTH_KNOT_M)
        v, r = fit_piecewise(s, values, kn, min_sigma=0.05)
        residuals.append(r)
        k2, v2 = simplify(kn, v, WIDTH_TOL_M)
        out.append((k2, v2))
    res = np.concatenate(residuals)
    ok = np.abs(res) < 2.0
    n = max(len(sides[0][0]), len(sides[1][0]))
    return EdgeFit(
        out[0][0],
        out[0][1],
        out[1][0],
        out[1][1],
        float(np.sqrt(np.mean(res[ok] ** 2))) if ok.any() else math.nan,
        n,
        float(min(len(sides[0][0]), len(sides[1][0])) / max(len(prof.s), 1)),
    )


def piecewise_from(knots: np.ndarray, values: np.ndarray, prov: Provenance) -> Piecewise:
    return Piecewise([Knot(float(k), float(v), prov) for k, v in zip(knots, values, strict=True)])


def fit_cross_slopes(
    s: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    length: float,
    gaps: list[tuple[float, float]],
    crown: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], float]:
    """Piecewise-linear slope_left (u > 0) and slope_right (u < 0) from residuals r = z - profile."""
    knots = knot_stations(length, SLOPE_KNOT_M, gaps)
    k = len(knots)
    hat = hat_basis(s, knots)
    left, right = u > 0.0, u < 0.0
    # Joint model r = c(s) + slope_left(s) u+ + slope_right(s) u-; the offset c(s)
    # absorbs any bias of the current profile so the slopes are not pulled by it.
    design = np.hstack([hat, hat * (u * left)[:, None], hat * (u * right)[:, None]])
    w = np.ones(len(s))
    prior = np.concatenate([np.zeros(k), np.full(k, -crown), np.full(k, crown)])
    pw = np.concatenate([np.full(k, 1e-6), np.full(k, SLOPE_PRIOR_WEIGHT), np.full(k, SLOPE_PRIOR_WEIGHT)])
    diff = np.diff(np.eye(k), axis=0)
    smooth = np.zeros((3 * (k - 1), 3 * k))
    for b in range(3):
        smooth[b * (k - 1) : (b + 1) * (k - 1), b * k : (b + 1) * k] = diff * 1e-3
    values = prior.copy()
    for _ in range(6):
        aw = design * w[:, None]
        lhs = design.T @ aw + np.diag(pw) + smooth.T @ smooth
        rhs = aw.T @ r + pw * prior
        values = np.linalg.solve(lhs, rhs)
        res = r - design @ values
        sigma = max(1.4826 * float(np.median(np.abs(res))), 0.02)
        a = np.abs(res) / (1.5 * sigma)
        w = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-12))
    out = []
    for b, side in ((1, left), (2, right)):
        v = values[b * k : (b + 1) * k]
        if side.sum() < 20:
            v = prior[b * k : (b + 1) * k]
        out.append(simplify(knots, v, SLOPE_TOL))
    rms = float(np.sqrt(np.mean(res[np.abs(res) < 0.5] ** 2))) if len(res) else math.nan
    return out[0], out[1], rms


def infer_lanes(
    width_left: float,
    width_right: float,
    road_class: str,
    oneway: bool,
    rules: RuleBook,
    length: float,
) -> list[LaneGroup]:
    """FR-XS-3: lane count from paved width and class; every value is rule-sourced."""
    lane_w, lane_prov = rules.fill("lane.width", road_class=road_class)
    sh_r, _ = rules.fill("shoulder.right.width", road_class=road_class)
    sh_l, _ = rules.fill("shoulder.left.width", road_class=road_class)
    total = width_left + width_right
    if oneway:
        n = max(1, round((total - sh_r - sh_l) / lane_w))
    else:
        n = max(2, 2 * round((total - 2 * sh_r) / (2 * lane_w)))
    left = n // 2
    right = n - left
    prov = Provenance(Source.RULE, lane_prov.confidence, rule_id="lane.width", flags=["FR-XS-3:inferred", *lane_prov.flags])
    # One-way: the left group's outer shoulder is the median-side (left) shoulder.
    groups = []
    if left:
        groups.append(LaneGroup("left", 0.0, length, left, lane_w, 0.0, sh_l if oneway else sh_r, prov))
    groups.append(LaneGroup("right", 0.0, length, right, lane_w, 0.0, sh_r, prov))
    return groups


def barriers_from_evidence(
    road_id: str,
    alignment_line: Polyline,
    ev: Evidence,
    width_left,
    width_right,
    *,
    structure_ranges: list[tuple[float, float]],
    rules: RuleBook,
    era: str = "*",
) -> list[Barrier]:
    """FR-RS-4: raised linear features at the pavement edges become Barrier elements."""
    out = []
    base_w, base_prov = rules.fill("parapet.base_width", era=era)
    top_w, _ = rules.fill("parapet.top_width", era=era)
    for feat in ev.features:
        if feat.kind != "linear":
            continue
        # Point positions, not cell centers: the inner face is the nearest returns.
        pts = ev.xyz[np.isin(ev.point_layer, feat.layer_idx), :2]
        s, u, ok = alignment_line.project(pts, max_dist=40.0)
        if ok.sum() < 3:
            continue
        s, u = s[ok], u[ok]
        for side, sign in (("left", 1.0), ("right", -1.0)):
            w = np.array([(width_left if sign > 0 else width_right)(float(v)) for v in s])
            d = sign * u - w
            m = (d >= -BARRIER_INSIDE_M) & (d <= BARRIER_REACH_M)
            if m.sum() < 3:
                continue
            s0, s1 = float(np.min(s[m])), float(np.max(s[m]))
            if s1 - s0 < MIN_BARRIER_M:
                continue
            offset = float(np.percentile(sign * u[m], 5.0))
            on_deck = any(a - 2.0 <= 0.5 * (s0 + s1) <= b + 2.0 for a, b in structure_ranges)
            if feat.top_height_m <= CURB_MAX_M:
                kind = "curb"
            elif on_deck:
                kind = "parapet"
            else:
                kind = "guardrail" if feat.top_height_m <= 0.85 else "jersey"
            outer = float(np.percentile(sign * u[m], 95.0))
            widths = (max(outer - offset, 0.15), max(outer - offset, 0.15)) if kind == "curb" else (base_w, top_w)
            prov = Provenance(Source.MEASURED, 0.7, n_points=int(m.sum()), flags=["FR-RS-4"])
            if kind != "curb":
                prov.flags.append(f"FR-RL-3:widths_by_rule:{base_prov.rule_id}")
            out.append(
                Barrier(
                    stable_id("br", road_id, side, round(s0, 0)),
                    side,
                    s0,
                    s1,
                    max(offset, 0.1),
                    float(feat.top_height_m),
                    widths[0],
                    widths[1],
                    kind,
                    prov,
                )
            )
    return _merge_barriers(out)


def _merge_barriers(items: list[Barrier]) -> list[Barrier]:
    """Join barriers on one side that overlap or nearly touch."""
    items = sorted(items, key=lambda b: (b.side, b.s0, b.id))
    out: list[Barrier] = []
    for b in items:
        if out and out[-1].side == b.side and b.s0 <= out[-1].s1 + 3.0 and out[-1].kind == b.kind:
            prev = out[-1]
            n0, n1 = prev.prov.n_points, b.prov.n_points
            prev.s1 = max(prev.s1, b.s1)
            prev.offset_m = (prev.offset_m * n0 + b.offset_m * n1) / max(n0 + n1, 1)
            prev.height_m = (prev.height_m * n0 + b.height_m * n1) / max(n0 + n1, 1)
            prev.prov.n_points = n0 + n1
        else:
            out.append(b)
    return out
