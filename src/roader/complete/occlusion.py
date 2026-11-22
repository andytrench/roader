"""Completing what was not observed (FR-OC-1, FR-OC-2).

The alignment, profile and cross-section of a road are fitted over its whole
length with no breakpoint inside an unobserved gap, so a road passing under a
deck is carried through from both sides by construction (``roader.fit``): no
sag, bump or width change can appear in the gap. This module labels those gaps
in the model (support intervals under a structure become ``occluded`` with
reason ``under_deck``) and checks every completed underpass against the
structure above.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from roader.model.schema import Crossing, Flag, Interval, Road, RoadModel, Structure
from roader.rules import RuleBook

CLEARANCE_STEP_M = 1.0
MIN_PHYSICAL_M = 0.05
COMPLETE_WINDOW_M = 30.0  # observed surface this far either side of a gap anchors its completion
COMPLETION_TARGET_M = 0.25  # AC-A9 occluded-profile target
ONE_SIDED_GRADE = 0.01  # grade uncertainty per meter when only one side was seen


@dataclass
class Completion:
    """FR-OC-1: one occluded gap of a road, carried through from the observed sides.

    ``bound_m`` bounds the completed surface's vertical error: the edge residual
    of the fit plus how far the carried profile departs from the nearer of the
    straight-grade extrapolations from each side. None when no side was observed.
    """

    road_id: str
    s0: float
    s1: float
    reason: str
    above: list[str]  # structures whose deck covers the gap, lowest first is not implied
    sides: int  # observed sides the profile is carried from (0, 1 or 2)
    bound_m: float | None


def structures_over(road: Road, s: float, model: RoadModel) -> list[str]:
    """Ids of structures (other than those this road carries) whose deck footprint covers station s."""
    x, y = road.alignment.offset_point(s, 0.0)
    out = []
    for st in model.structures:
        if st.carried_road == road.id:
            continue
        upper = model.road(st.carried_road)
        su, u = upper.alignment.project(x, y)
        half = max(upper.cross_section.width_left(su), upper.cross_section.width_right(su)) + st.deck.overhang_m + 2.0
        if st.s_start - 2.0 <= su <= st.s_end + 2.0 and abs(u) <= half and upper.profile.z(su) > road.profile.z(s):
            out.append(st.id)
    return sorted(out)


def label_under_deck(road: Road, model: RoadModel) -> None:
    """Rename occluded intervals that lie under a structure footprint to reason 'under_deck'.

    In a stack the deck above may be any level higher than this road, including
    one this road's own structure passes under (FR-RS-2).
    """
    for iv in road.support:
        if iv.kind != "occluded":
            continue
        if structures_over(road, 0.5 * (iv.s0 + iv.s1), model):
            iv.reason = "under_deck"


def complete_occluded(model: RoadModel, tracks: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[Completion]:
    """FR-OC-1 completion records for every occluded interval; ``tracks``: road id -> (s, z) observed surface."""
    out = []
    for road in model.roads:
        s_obs, z_obs = tracks.get(road.id, (np.zeros(0), np.zeros(0)))
        ok = np.isfinite(z_obs)
        s_obs, z_obs = s_obs[ok], z_obs[ok]
        for iv in road.support:
            if iv.kind != "occluded":
                continue
            above = sorted({sid for s in np.linspace(iv.s0, iv.s1, 5) for sid in structures_over(road, float(s), model)})
            sides, bound = _bound(road, s_obs, z_obs, iv.s0, iv.s1)
            out.append(Completion(road.id, iv.s0, iv.s1, iv.reason, above, sides, bound))
    return out


def completion_flags(completions: list[Completion]) -> list[Flag]:
    flags = []
    for c in completions:
        where = f"s={c.s0:.1f}-{c.s1:.1f} ({c.reason}{'; under ' + ', '.join(c.above) if c.above else ''})"
        if c.bound_m is None:
            flags.append(Flag("FR-OC-1:unbounded", c.road_id, "FR-OC-1", f"{where}: no observed surface on either side; completed by the fit alone"))
            continue
        flags.append(Flag("FR-OC-1:completed", c.road_id, "FR-OC-1", f"{where}: carried from {c.sides} observed side(s), vertical bound {c.bound_m:.2f} m"))
        if c.bound_m > COMPLETION_TARGET_M:
            flags.append(Flag("FR-OC-1:uncertain", c.road_id, "FR-OC-1", f"{where}: bound {c.bound_m:.2f} m exceeds the AC-A9 target {COMPLETION_TARGET_M:.2f} m"))
    return flags


def _bound(road: Road, s_obs: np.ndarray, z_obs: np.ndarray, s0: float, s1: float) -> tuple[int, float | None]:
    gap = np.linspace(s0, s1, max(3, int((s1 - s0) / CLEARANCE_STEP_M) + 1))
    zm = np.array([road.profile.z(float(s)) for s in gap])
    resid, extrap, anchors = [], [], []
    for lo, hi, anchor in ((s0 - COMPLETE_WINDOW_M, s0, s0), (s1, s1 + COMPLETE_WINDOW_M, s1)):
        m = (s_obs >= lo) & (s_obs <= hi)
        if m.sum() < 3 or np.ptp(s_obs[m]) < 2.0:
            continue
        a, b = np.polyfit(s_obs[m], z_obs[m], 1)
        resid.append(z_obs[m] - np.array([road.profile.z(float(s)) for s in s_obs[m]]))
        extrap.append(a * gap + b)
        anchors.append(anchor)
    if not extrap:
        return 0, None
    edge = float(np.sqrt(np.mean(np.concatenate(resid) ** 2)))
    dev = np.min(np.abs(np.array(extrap) - zm[None, :]), axis=0)
    if len(extrap) == 1:
        dev = dev + ONE_SIDED_GRADE * np.abs(gap - anchors[0])
    return len(extrap), edge + float(np.max(dev))


def split_at_structures(road: Road, structures: list[Structure]) -> None:
    """Carried roads: support intervals split at structure ends (spec section 5 L5 references)."""
    cuts = sorted({s for st in structures if st.carried_road == road.id for s in (st.s_start, st.s_end)})
    out = []
    for iv in road.support:
        edges = [iv.s0] + [c for c in cuts if iv.s0 < c < iv.s1] + [iv.s1]
        for a, b in itertools.pairwise(edges):
            out.append(Interval(a, b, iv.kind, iv.reason))
    road.support = out


def clearance_profile(model: RoadModel, c: Crossing, st: Structure) -> tuple[float, int]:
    """Minimum lower-surface-to-soffit clearance over the lower road under the deck footprint."""
    from roader.conflate.polyline import Polyline
    from roader.validate.accuracy import _pw

    upper, lower = model.road(c.upper_road), model.road(c.lower_road)
    reach = (st.s_end - st.s_start) + 30.0
    s = np.arange(max(0.0, c.lower_s - reach), min(lower.length, c.lower_s + reach), CLEARANCE_STEP_M)
    lx = lower.cross_section
    rows = []
    for frac in (-1.0, 0.0, 1.0):
        u = np.where(frac < 0, -_pw(lx.width_right, s), _pw(lx.width_left, s) * frac)
        rows.append((s, u))
    ss = np.concatenate([r[0] for r in rows])
    uu = np.concatenate([r[1] for r in rows])
    lower_line = Polyline.from_alignment(lower.alignment)
    xy = lower_line.offset_point(ss, uu)
    upper_line = Polyline.from_alignment(upper.alignment)
    su, us, ok = upper_line.project(xy)
    ux = upper.cross_section
    inside = ok & (su >= st.s_start) & (su <= st.s_end)
    inside &= (us <= _pw(ux.width_left, su) + st.deck.overhang_m) & (us >= -_pw(ux.width_right, su) - st.deck.overhang_m)
    if not inside.any():
        return math.inf, 0
    soffit = np.array([upper.surface_z(float(a), float(b)) for a, b in zip(su[inside], us[inside], strict=True)]) - st.deck.depth_m
    low = np.array([lower.surface_z(float(a), float(b)) for a, b in zip(ss[inside], uu[inside], strict=True)])
    return float(np.min(soffit - low)), int(inside.sum())


def check_clearances(model: RoadModel, rules: RuleBook) -> list[Flag]:
    """FR-OC-2: clearance positive and at least the class minimum; on failure refit the depth, then flag.

    Only the rule-filled superstructure depth is refitted (down to the low end of
    the depth-ratio rule); measured surfaces are never moved.
    """
    flags: list[Flag] = []
    for c in model.crossings:
        if not c.structure_id:
            continue
        st = model.structure(c.structure_id)
        lower = model.road(c.lower_road)
        need, _prov = rules.fill("clearance.vertical.rural", road_class=lower.road_class)
        clear, n = clearance_profile(model, c, st)
        if n == 0:
            clear = c.dz_m - st.deck.depth_m
        if clear < need and st.deck.prov.rule_id:
            rule = rules.get(st.deck.prov.rule_id)
            spans = np.diff([sp.station for sp in sorted(st.supports, key=lambda p: p.station)])
            max_span = float(spans.max()) if len(spans) else st.s_end - st.s_start
            thinner = max(rule.low * max_span, st.deck.slab_thickness_m + 0.1)
            if thinner < st.deck.depth_m:
                st.deck.depth_m = thinner
                st.deck.prov.flags.append("FR-OC-2:depth_refit_to_rule_low")
                clear, n = clearance_profile(model, c, st)
                if n == 0:
                    clear = c.dz_m - st.deck.depth_m
        c.clearance_m = float(clear)
        if clear <= MIN_PHYSICAL_M:
            flags.append(Flag("FR-OC-2:impossible", c.id, "FR-OC-2", f"clearance {clear:.2f} m is not physically possible"))
        elif clear < need:
            c.prov.flags.append("FR-OC-2:below_minimum_measured")
            flags.append(Flag("FR-OC-2:below_minimum", c.id, "FR-OC-2", f"clearance {clear:.2f} m below class minimum {need:.2f} m (measured surfaces; depth by rule)"))
    return flags
