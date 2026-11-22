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
from roader.conflate.register import Profiles
from roader.evidence import Evidence
from roader.fit.piecewise import fit_piecewise, hat_basis, knot_stations, simplify
from roader.model.geometry import Knot, Piecewise
from roader.model.provenance import Provenance, Source
from roader.model.schema import Barrier, LaneGroup
from roader.model.serialize import stable_id
from roader.rules import RuleBook

WIDTH_KNOT_M = 50.0
WIDTH_TOL_M = 0.15
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


def fit_edges(prof: Profiles, length: float, gaps: list[tuple[float, float]]) -> EdgeFit | None:
    valid = prof.valid
    if valid.sum() < 3:
        return None
    s = prof.s[valid]
    knots = knot_stations(length, WIDTH_KNOT_M, gaps)
    out = []
    residuals = []
    for values in (prof.left[valid], -prof.right[valid]):
        v, r = fit_piecewise(s, values, knots, min_sigma=0.05)
        residuals.append(r)
        k2, v2 = simplify(knots, v, WIDTH_TOL_M)
        out.append((k2, v2))
    res = np.concatenate(residuals)
    ok = np.abs(res) < 2.0
    return EdgeFit(
        out[0][0],
        out[0][1],
        out[1][0],
        out[1][1],
        float(np.sqrt(np.mean(res[ok] ** 2))) if ok.any() else math.nan,
        int(valid.sum()),
        float(valid.mean()),
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
