"""Decks shared between roads (FR-CR-3): twin decks meeting at a joint, ramps riding a deck.

Twin carriageways often cross on two structures side by side whose rule overhangs
would overlap in a narrow median. Decks of twin structures are disjoint: on the
sides facing each other each deck stops at the joint line half way across the gap
between the two pavements (or the barriers measured there), less an open joint
of ``JOINT_GAP_M``, and no rule parapet
is filled on a joint.

A ramp diverging from (or merging into) a road on that road's deck rides the
parent's deck (``roader.conflate.structures.absorb_diverging``). Here the parent
structure records the stretch of the ramp it carries, up to where the ramp's whole
section has cleared the parent's skewed deck end, so the ramp's own pavement,
barriers and earthwork start there.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from roader.conflate.structures import Share
from roader.model.provenance import Source
from roader.model.schema import Barrier, Crossing, Flag, Road, SharedSpan, Structure

TWIN_STEP_M = 1.0
TWIN_MIN_SAMPLES = 5
TWIN_MAX_DZ_M = 1.0  # deck tops this close in height at the joint
TWIN_PARALLEL = math.cos(math.radians(35.0))
TWIN_REACH_M = 40.0
JOINT_GAP_M = 0.02  # open longitudinal joint between twin decks; touching decks from two road grids cannot conform
CLEAR_STEP_M = 0.5
CLEAR_TOL_M = 1e-3
FOOTPRINT_STEP_M = 1.0


def _side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def _width(road: Road, side: str):
    return road.cross_section.width_left if side == "left" else road.cross_section.width_right


def _barrier_reach(road: Road, st: Structure, side: str) -> float:
    """Outer face of the barriers standing on one side of the deck, 0 when there are none."""
    return max(
        (b.offset_m + b.base_width_m for b in road.cross_section.barriers if b.side == side and b.s0 < st.s_end and b.s1 > st.s_start),
        default=0.0,
    )


def _end_slope(st: Structure, near: float) -> float:
    if not st.supports:
        return 0.0
    return math.tan(min(st.supports, key=lambda sp: (abs(sp.station - near), sp.id)).skew_rad)


def _beside(a: Structure, b: Structure, ra: Road, rb: Road) -> list[tuple[float, float, float]]:
    """Stations of ``a`` with ``b``'s road alongside its deck: (s on a, s on b, u of a's line on b)."""
    margin = max(abs(_end_slope(b, b.s_start)), abs(_end_slope(b, b.s_end))) * (max(k.value for k in rb.cross_section.width_left.knots) + b.deck.overhang_m) + 1.0
    out = []
    for s in np.arange(a.s_start, a.s_end + 1e-9, TWIN_STEP_M):
        s = float(s)
        x, y, ha, _k = ra.alignment.evaluate(s)
        sb, ub = rb.alignment.project(x, y)
        if not b.s_start - margin <= sb <= b.s_end + margin or abs(ub) > TWIN_REACH_M:
            continue
        hb = rb.alignment.evaluate(sb)[2]
        if abs(math.cos(ha - hb)) < TWIN_PARALLEL or abs(ra.profile.z(s) - rb.profile.z(sb)) > TWIN_MAX_DZ_M:
            continue
        out.append((s, sb, ub))
    return out


def _gaps(a: Structure, ra: Road, side_a: str, b: Structure, rb: Road, side_b: str, reach_a: float, reach_b: float, samples) -> list[float]:
    """Plan gap between the two facing edges (pavement, or a barrier standing beyond it) at each sample."""
    sign_a, sign_b = _side_sign(side_a), _side_sign(side_b)
    out = []
    for s, _sb, _ub in samples:
        edge_a = max(_width(ra, side_a)(s), reach_a)
        x, y = ra.alignment.offset_point(s, sign_a * edge_a)
        sb, ub = rb.alignment.project(x, y)
        out.append(sign_b * ub - max(_width(rb, side_b)(sb), reach_b))
    return out


def twin_joints(structures: list[Structure], roads: dict[str, Road], crossings: list[Crossing]) -> list[Flag]:
    """Make the decks of twin structures disjoint (FR-CR-3, AC-H3, FR-SV-7).

    Two structures on different roads are twins when the roads run alongside each
    other over the decks at about the same height and do not cross, and their
    facing deck edges (barriers and overhang) would meet or overlap. On the facing
    sides, rule parapets are dropped and the overhang becomes half the smallest gap
    between the facing edges, so the decks meet at most on a joint line.
    """
    crossing_pairs = {frozenset((c.upper_road, c.lower_road)) for c in crossings}
    flags: list[Flag] = []
    ordered = sorted(structures, key=lambda st: st.id)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if a.carried_road == b.carried_road or frozenset((a.carried_road, b.carried_road)) in crossing_pairs:
                continue
            ra, rb = roads[a.carried_road], roads[b.carried_road]
            samples = _beside(a, b, ra, rb)
            if len(samples) < TWIN_MIN_SAMPLES:
                continue
            side_b = "left" if float(np.median([ub for _s, _sb, ub in samples])) > 0.0 else "right"
            ua = [ra.alignment.project(*rb.alignment.offset_point(sb, 0.0))[1] for _s, sb, _ub in samples]
            side_a = "left" if float(np.median(ua)) > 0.0 else "right"
            pavement = min(_gaps(a, ra, side_a, b, rb, side_b, 0.0, 0.0, samples))
            used = (
                max(_barrier_reach(ra, a, side_a) - max(_width(ra, side_a)(s) for s, _sb, _ub in samples), 0.0)
                + a.deck.overhang(side_a)
                + max(_barrier_reach(rb, b, side_b) - max(_width(rb, side_b)(sb) for _s, sb, _ub in samples), 0.0)
                + b.deck.overhang(side_b)
            )
            if pavement >= used:
                continue  # the decks never meet
            dropped = 0
            for road, st, side in ((ra, a, side_a), (rb, b, side_b)):
                keep = [
                    bar
                    for bar in road.cross_section.barriers
                    if not (bar.side == side and bar.kind == "parapet" and bar.prov.source == Source.RULE and bar.s0 < st.s_end and bar.s1 > st.s_start)
                ]
                dropped += len(road.cross_section.barriers) - len(keep)
                road.cross_section.barriers = keep
            gap = min(_gaps(a, ra, side_a, b, rb, side_b, _barrier_reach(ra, a, side_a), _barrier_reach(rb, b, side_b), samples))
            # Negative when the fitted pavements overlap: the decks stop short of their pavement edges.
            half = (gap - JOINT_GAP_M) / 2.0
            for st, side in ((a, side_a), (b, side_b)):
                value = min(st.deck.overhang(side), half)
                if side == "left":
                    st.deck.overhang_left_m = value
                else:
                    st.deck.overhang_right_m = value
                st.deck.prov.flags.append(f"FR-CR-3:twin_joint:{side}")
            for st, other in ((a, b), (b, a)):
                message = f"twin of {other.id}: facing pavements {pavement:.2f} m apart, edges {gap:.2f} m; joint overhang {half:.2f} m, {dropped} rule parapets dropped"
                flags.append(Flag("FR-CR-3:twin_joint", st.id, "FR-CR-3", message))
            if gap < JOINT_GAP_M:
                for st in (a, b):
                    flags.append(Flag("FR-CR-3:twin_overlap", st.id, "FR-CR-3", f"facing edges {gap:.2f} m apart in plan, under the {JOINT_GAP_M} m joint; each deck stops {-half:.2f} m inside its pavement edge"))
    return flags


def _clear_of_deck(st: Structure, parent: Road, child: Road, s_seam: float, toward: float, reach: tuple[float, float]) -> float:
    """First station of ``child`` from its seam where its whole section is past the parent's skewed deck end."""
    probe = min(max(s_seam + toward * 20.0, 0.0), child.length)
    p0 = parent.alignment.project(*child.alignment.offset_point(s_seam, 0.0))[0]
    p1 = parent.alignment.project(*child.alignment.offset_point(probe, 0.0))[0]
    to_end = p1 >= p0
    end = st.s_end if to_end else st.s_start
    t = _end_slope(st, end)

    def clear(s: float) -> bool:
        for u in (reach[0], 0.0, -reach[1]):
            ps, pu = parent.alignment.project(*child.alignment.offset_point(s, u))
            q = ps - t * pu
            if (q < end - CLEAR_TOL_M) if to_end else (q > end + CLEAR_TOL_M):
                return False
        return True

    limit = child.length if toward > 0 else 0.0
    lo = s_seam
    s = s_seam
    while not clear(s):
        lo = s
        s = s + toward * CLEAR_STEP_M
        if (s - limit) * toward >= 0.0:
            return limit
    hi = s
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if clear(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _inside_parent(parent: Road, child: Road, s_from: float, toward: float) -> float:
    """Last station of ``child`` from ``s_from`` before either pavement edge leaves the parent's pavement."""
    xs, pxs = child.cross_section, parent.cross_section

    def inside(s: float) -> bool:
        for u in (xs.width_left(s), -xs.width_right(s)):
            ps, pu = parent.alignment.project(*child.alignment.offset_point(s, u))
            if not (0.0 < ps < parent.length and -pxs.width_right(ps) - CLEAR_TOL_M <= pu <= pxs.width_left(ps) + CLEAR_TOL_M):
                return False
        return True

    limit = child.length if toward > 0 else 0.0
    if not inside(s_from):
        return s_from
    lo, s = s_from, s_from
    while inside(s):
        lo = s
        s = s + toward * CLEAR_STEP_M
        if (s - limit) * toward >= 0.0:
            return limit
    hi = s
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _clip_barriers(barriers: list[Barrier], a: float, b: float) -> list[Barrier]:
    out = []
    for bar in barriers:
        if bar.s1 <= a or bar.s0 >= b:
            out.append(bar)
            continue
        if bar.s0 < a:
            out.append(_cut(bar, bar.s0, a))
        if bar.s1 > b:
            out.append(_cut(bar, b, bar.s1))
    return out


def _cut(bar: Barrier, s0: float, s1: float) -> Barrier:
    return dataclasses.replace(bar, s0=s0, s1=s1, prov=dataclasses.replace(bar.prov, flags=[*bar.prov.flags, "FR-CR-3:clipped_at_shared_deck"]))


def attach_shared(
    structures: list[Structure], shares: list[Share], roads: dict[str, Road], barriers: dict[str, list[Barrier]], crossings: list[Crossing]
) -> list[Flag]:
    """Record on each parent structure the stretch of every road riding its deck (FR-CR-3).

    The stretch runs from the road's seam to where its section, barriers included,
    has cleared the parent's deck end along the skewed end line (D-025), and on
    while both its pavement edges stay on the parent's pavement. The road's
    barriers are cut back to outside it and crossings it carries there take the
    parent's structure. A stretch whose pavement leaves the parent's deck sideways
    is flagged: the parent deck does not cover it.
    """
    flags: list[Flag] = []
    for sh in shares:
        st = next((s for s in structures if s.carried_road == sh.parent_road and s.s_start - 1.0 <= sh.parent_s <= s.s_end + 1.0), None)
        if st is None:
            flags.append(Flag("FR-CR-3:no_parent_structure", sh.road_id, "FR-CR-3", f"seam on {sh.parent_road} at s={sh.parent_s:.1f} has no structure; road rides on nothing"))
            continue
        parent, child = roads[sh.parent_road], roads[sh.road_id]
        mid = sh.s_seam + sh.toward * 1.0
        reach = tuple(
            max([_width(child, side)(mid), *(b.offset_m + b.base_width_m for b in barriers.get(child.id, []) if b.side == side)]) for side in ("left", "right")
        )
        deck_end = _clear_of_deck(st, parent, child, sh.s_seam, sh.toward, reach)
        end = _inside_parent(parent, child, deck_end, sh.toward)
        a, b = sorted((sh.s_seam, end))
        a, b = max(a, 0.0), min(b, child.length)
        st.shared.append(SharedSpan(child.id, a, b))
        st.shared.sort(key=lambda x: (x.road_id, x.s_start))
        barriers[child.id] = _clip_barriers(barriers.get(child.id, []), a, b)
        da, db = sorted((sh.s_seam, deck_end))
        for c in crossings:
            if c.upper_road == child.id and da <= c.upper_s <= db:
                c.structure_id = st.id
        outside = _outside_deck(st, parent, child, a, b)
        flags.append(Flag("FR-CR-3:shares_deck", st.id, "FR-CR-3", f"{child.id} ({child.name}) rides this deck and its road's pavement over s={a:.1f}-{b:.1f} from its seam at {sh.parent_road} s={sh.parent_s:.1f}"))
        if outside:
            flags.append(
                Flag("FR-CR-3:shared_deck_too_narrow", st.id, "FR-CR-3", f"{child.id} pavement leaves the deck sideways by up to {outside:.2f} m on its shared stretch s={a:.1f}-{b:.1f}")
            )
    return flags


def _outside_deck(st: Structure, parent: Road, child: Road, a: float, b: float) -> float:
    """How far the child's pavement edges reach past the parent's deck edges over [a, b] (0 when inside)."""
    worst = 0.0
    t0, t1 = _end_slope(st, st.s_start), _end_slope(st, st.s_end)
    for s in np.arange(a, b + 1e-9, FOOTPRINT_STEP_M):
        for u in (child.cross_section.width_left(float(s)), -child.cross_section.width_right(float(s))):
            ps, pu = parent.alignment.project(*child.alignment.offset_point(float(s), u))
            if not st.s_start + t0 * pu <= ps <= st.s_end + t1 * pu:
                continue  # past the deck end: the road's own pavement there
            left = parent.cross_section.width_left(ps) + st.deck.overhang("left")
            right = parent.cross_section.width_right(ps) + st.deck.overhang("right")
            worst = max(worst, pu - left, -right - pu)
    return worst
