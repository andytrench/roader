"""L6 junction areas from a planar partition of the road edge lines (FR-SG-3, AC-H4, D-046).

Where a ramp leaves or joins another road, or a road ends on another road's
pavement, independently swept roads overlap. Each such place becomes one
junction area that replaces the overlapping sweeps:

1. Participants: the roads meeting at a node, each with an anchor station (a road
   end, or the station a branch end projects to). Nodes come from diverge and
   merge links and from road ends that lie on another road's pavement at grade.
2. Each participant's station interval grows away from the node until its cross
   section line is at least ``GORE_GAP_M`` clear of every other participant's
   pavement, plus ``SEAM_MARGIN_M``. Interval ends inside a road are seams.
3. Outline: the union of the participants' pavement over their intervals (the
   planar partition of their edge lines), with every gore pocket between two
   neighbouring seams closed by a straight line across it.
4. Lifting: each seam stores a patch of its road's surface, and lifted sections
   of every participant sit inside the area; ``JunctionArea.surface_z`` blends
   them (D-046), so elevation and slope match each road at its seam.

Areas that overlap each other, or that a third road's pavement enters at the same
level, are merged into one area and regrown (D-069). An area may sit next to a
deck: a participant's interval stays off its own decks and their skewed station
ramps, no support may stand in the area, and another road's deck over the area
must clear it by ``LEVEL_DZ_M``; the lifted surface must stay within ``LEVEL_DZ_M``
of every participant it covers. Otherwise the node is flagged and left out.

Shared pavement (D-070): two roads at the same level whose pavements still overlap
outside every junction area (adjacent ramps, twin carriageways, a ramp that never
separates from its mainline) each get one ``shared`` area over their overlapping
interval. The overlap is split midway between the two roads' edges (the midpoint
of each overlap chord across the narrower road), or given wholly to the narrower
road when it lies inside the other; each road keeps its own surface on its side,
and both outlines carry the same split vertices so their faces conform (D-021).

Outline vertices closer than ``MIN_EDGE_M`` are merged (FR-SV-5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import split as split_polygon

from roader.model.provenance import Provenance, Source
from roader.model.schema import (
    Flag,
    JunctionArea,
    JunctionSeam,
    Road,
    RoadModel,
    Structure,
    Support,
)
from roader.model.serialize import FLOAT_DIGITS, stable_id
from roader.solids.stations import RAMP_FACTOR, RAMP_MIN_M, support_block

GORE_GAP_M = 1.0  # a seam's cross-section line stays this clear of other participants' pavement
SEAM_MARGIN_M = 1.0  # added beyond the first clear station
SEARCH_STEP_M = 0.5
MAX_REACH_M = 200.0  # roads still overlapping this far from the node are a fitting problem, not a junction
LOCAL_REACH_M = MAX_REACH_M + 100.0  # footprint of another participant considered around its anchor
TOUCH_LATERAL_M = 1.0  # a road end this far outside another road's pavement edge still touches it
END_SNAP_M = 0.5  # anchors this close to a road end are that end
LEVEL_DZ_M = 1.0  # a road end within this of the other surface is at grade (else it passes over or under)
NODE_MERGE_M = 1.0
SECTION_SPACING_M = 5.0  # lifted interior sections along each participant
SECTION_CLEAR_M = 2.0  # no interior section this close to a seam
OUTLINE_STEP_M = 2.0
OUTLINE_CHORD_M = 0.01
STRUCTURE_CLEAR_M = 1.0  # seams stay this far off decks and their skewed station ramps
# The solids ramp a structure's skew to zero over RAMP_FACTOR * |tan skew| * umax (solids.stations);
# umax is estimated from the widths, the deck overhang and this allowance for embankment toes.
TOE_ALLOWANCE_M = 30.0
INTERSECTION_ANGLE = math.radians(45.0)
STEP_FLAG_M = 0.05  # participants disagreeing by more at the node are flagged (FR-AF-3)
DERIV_STEP_M = 0.5
GORE_OUTSIDE_FRACTION = 0.99
# A gore closure joins two seam rows that cleared each other by about GORE_GAP_M; a longer
# chord would cut across the outside of a curve, not a gore.
GORE_CLOSURE_MAX_M = 6.0
_TRUST = {Source.MEASURED: 0, Source.INVENTORY: 1, Source.VECTOR: 2, Source.RULE: 3}
LIFT_RULE_ID = "junction.lift"
MAX_ROUNDS = 40  # merge rounds of overlapping areas
MIN_EDGE_M = 1e-3  # FR-SV-5: outline vertices closer than this are one vertex
CONFORM_TOL_M = 2e-6  # a vertex this close to another outline's edge is put on it (D-021)
SHARED_MIN_M2 = 0.05  # overlaps smaller than this are rounding, not shared pavement
SHARED_STEP_M = 1.0  # overlap chords along the narrower road
SHARED_REACH_M = 100.0  # the other road's footprint considered beyond the overlap
EDGE_TOL_M = 0.05  # a chord end this close to a road's edge lies on that edge
CONTAINED_FRACTION = 0.8  # chords wholly inside the narrower road: it keeps the overlap
SPLIT_EXTEND_M = 2.0
MIN_CHORD_M = 0.05  # shorter chords near an overlap tip are left out of the split line (FR-SV-5)
SHARED_RULE_ID = "junction.shared_split"


@dataclass
class _Node:
    x: float
    y: float
    anchors: dict[str, float]  # road id -> anchor station


# ---------------------------------------------------------------- road geometry helpers


_SAMPLES: dict[int, tuple[object, np.ndarray, cKDTree]] = {}


def _project(road: Road, x: float, y: float) -> tuple[float, float]:
    """``Alignment.project`` seeded from a cached 1 m sample tree (same Newton steps, same result)."""
    al = road.alignment
    hit = _SAMPLES.get(id(al))
    if hit is None or hit[0] is not al:
        length = al.length
        count = max(2, math.ceil(length) + 1)
        st = np.array([length * i / (count - 1) for i in range(count)])
        hit = (al, st, cKDTree(np.array([al.evaluate(float(v))[:2] for v in st])))
        _SAMPLES[id(al)] = hit
    _al, st, tree = hit
    s = float(st[int(tree.query((x, y))[1])])
    length = al.length
    for _ in range(8):
        px, py, heading, k = al.evaluate(s)
        tx, ty = math.cos(heading), math.sin(heading)
        dx, dy = x - px, y - py
        along = dx * tx + dy * ty
        lateral = -dx * ty + dy * tx
        denom = 1.0 - k * lateral
        if abs(denom) < 1e-6:
            break
        ds = along / denom
        s = min(max(s + ds, 0.0), length)
        if abs(ds) < 1e-9:
            break
    px, py, heading, _k = al.evaluate(s)
    return s, -(x - px) * math.sin(heading) + (y - py) * math.cos(heading)


def _edge_stations(road: Road, a: float, b: float) -> list[float]:
    """Stations from a to b dense enough for a chord error under OUTLINE_CHORD_M, with width knots."""
    out = [a]
    s = a
    while s < b - 1e-9:
        k = abs(road.alignment.evaluate(s)[3])
        step = OUTLINE_STEP_M if k < 1e-9 else min(OUTLINE_STEP_M, math.sqrt(8.0 * OUTLINE_CHORD_M / k))
        s = min(b, s + step)
        out.append(s)
    xs = road.cross_section
    knots = {kn.s for pw in (xs.width_left, xs.width_right) for kn in pw.knots if a + 1e-3 < kn.s < b - 1e-3}
    return sorted(set(out) | knots)


def _row(road: Road, s: float) -> list[tuple[float, float]]:
    """Right edge, reference line, left edge at station s."""
    xs = road.cross_section
    return [
        road.alignment.offset_point(s, -xs.width_right(s)),
        road.alignment.offset_point(s, 0.0),
        road.alignment.offset_point(s, xs.width_left(s)),
    ]


def _footprint(road: Road, a: float, b: float) -> Polygon:
    st = _edge_stations(road, a, b)
    xs = road.cross_section
    right = [road.alignment.offset_point(s, -xs.width_right(s)) for s in st]
    left = [road.alignment.offset_point(s, xs.width_left(s)) for s in reversed(st)]
    ring = right + [road.alignment.offset_point(b, 0.0)] + left + [road.alignment.offset_point(a, 0.0)]
    return shapely.make_valid(Polygon(ring)) if not Polygon(ring).is_valid else Polygon(ring)


def _row_line(road: Road, s: float) -> LineString:
    r = _row(road, s)
    return LineString([r[0], r[2]])


def _source(road: Road) -> tuple[Source, float, str | None]:
    if road.prov is not None:
        return road.prov.source, float(road.prov.confidence), road.prov.rule_id
    xs = road.cross_section
    provs = [e.prov for e in road.alignment.elements] + [e.prov for e in road.profile.elements]
    for pw in (xs.width_left, xs.width_right, xs.slope_left, xs.slope_right, xs.pavement_thickness):
        provs += [k.prov for k in pw.knots]
    worst = max(provs, key=lambda p: _TRUST[Source(p.source)])
    return Source(worst.source), min(float(p.confidence) for p in provs), worst.rule_id


# ---------------------------------------------------------------- nodes


def _snap_end(road: Road, s: float) -> float:
    if s <= END_SNAP_M:
        return 0.0
    if s >= road.length - END_SNAP_M:
        return road.length
    return s


def _touches(model: RoadModel) -> tuple[list[_Node], list[Flag]]:
    """Road ends on another road's pavement at grade, and diverge/merge links at grade."""
    roads = {r.id: r for r in model.roads}
    nodes: list[_Node] = []
    flags: list[Flag] = []
    for link in sorted(model.links, key=lambda lk: lk.id):
        if link.kind not in ("diverge", "merge") or link.from_road not in roads or link.to_road not in roads:
            continue
        if link.kind == "diverge":
            branch, end, trunk, s_t = roads[link.to_road], link.to_end, roads[link.from_road], link.from_s
        else:
            branch, end, trunk, s_t = roads[link.from_road], link.from_end, roads[link.to_road], link.to_s
        s_b = 0.0 if end == "start" else branch.length
        x, y = branch.alignment.offset_point(s_b, 0.0)
        u_t = _project(trunk, x, y)[1]
        if s_t is None:
            s_t = _project(trunk, x, y)[0]
        dz = branch.surface_z(s_b, 0.0) - trunk.surface_z(float(s_t), u_t)
        if abs(dz) > LEVEL_DZ_M:
            flags.append(Flag("FR-SG-3:not_level", link.id, "FR-SG-3",
                              f"{branch.id} meets {trunk.id} {dz:+.2f} m off its surface; no junction area"))
            continue
        nodes.append(_Node(x, y, {branch.id: s_b, trunk.id: _snap_end(trunk, float(s_t))}))
    # Coarse reference lines, to skip projecting onto roads that are nowhere near.
    coarse = {}
    for road in model.roads:
        n = max(2, math.ceil(road.length / 10.0) + 1)
        line = LineString([road.alignment.offset_point(road.length * i / (n - 1), 0.0) for i in range(n)])
        xs = road.cross_section
        reach = max(k.value for pw in (xs.width_left, xs.width_right) for k in pw.knots) + TOUCH_LATERAL_M + 10.0
        coarse[road.id] = (line, reach)
    for road in model.roads:
        for s_e in (0.0, road.length):
            x, y = road.alignment.offset_point(s_e, 0.0)
            z = road.surface_z(s_e, 0.0)
            anchors = {road.id: s_e}
            interior = False
            for other in model.roads:
                if other.id == road.id:
                    continue
                line, reach = coarse[other.id]
                if line.distance(Point(x, y)) > reach:
                    continue
                s, u = _project(other, x, y)
                px, py = other.alignment.offset_point(s, u)
                if math.hypot(px - x, py - y) > END_SNAP_M:
                    continue  # beyond the other road's end
                xs = other.cross_section
                if not -xs.width_right(s) - TOUCH_LATERAL_M <= u <= xs.width_left(s) + TOUCH_LATERAL_M:
                    continue
                if abs(other.surface_z(s, u) - z) > LEVEL_DZ_M:
                    continue
                s = _snap_end(other, s)
                anchors[other.id] = s
                interior = interior or 0.0 < s < other.length
            if interior or len(anchors) >= 3:
                nodes.append(_Node(x, y, anchors))
    return nodes, flags


# ---------------------------------------------------------------- intervals


def _local_footprint(road: Road, anchor: float) -> Polygon:
    return _footprint(road, max(0.0, anchor - LOCAL_REACH_M), min(road.length, anchor + LOCAL_REACH_M))


def _first_clear(road: Road, start: float, direction: float, others: list[Polygon]) -> float | None:
    """First station from start (moving in direction) whose cross-section line clears every other footprint.

    None when the road ends first; NaN when the roads still overlap MAX_REACH_M from the node.
    """
    s = start
    while 0.0 <= s <= road.length:
        if abs(s - start) > MAX_REACH_M:
            return math.nan
        line = _row_line(road, s)
        if all(line.distance(p) >= GORE_GAP_M for p in others):
            return s
        s += direction * SEARCH_STEP_M
    return None


def _interval(road: Road, anchor: float, others: list[Polygon]) -> tuple[float, float] | None:
    """Station interval of one participant, or None if it never clears the others."""
    length = road.length
    lo, hi = anchor, anchor
    if anchor > 0.0:
        c = _first_clear(road, anchor, -1.0, others)
        if (c is None and anchor >= length) or (c is not None and math.isnan(c)):
            return None
        lo = 0.0 if c is None else max(0.0, c - SEAM_MARGIN_M)
    if anchor < length:
        c = _first_clear(road, anchor, 1.0, others)
        if (c is None and anchor <= 0.0) or (c is not None and math.isnan(c)):
            return None
        hi = length if c is None else min(length, c + SEAM_MARGIN_M)
    if lo < END_SNAP_M:
        lo = 0.0
    if hi > length - END_SNAP_M:
        hi = length
    if lo <= 0.0 and hi >= length:
        return None  # the whole road would be absorbed
    return lo, hi


def _grow(road: Road, lo: float, hi: float, others: list[Polygon]) -> tuple[float, float] | None:
    """Move interval ends whose cross-section line is not clear of ``others`` outward until it is."""
    length = road.length
    if lo > 0.0 and not all(_row_line(road, lo).distance(p) >= GORE_GAP_M for p in others):
        c = _first_clear(road, lo, -1.0, others)
        if c is not None and math.isnan(c):
            return None
        lo = 0.0 if c is None else max(0.0, c - SEAM_MARGIN_M)
    if hi < length and not all(_row_line(road, hi).distance(p) >= GORE_GAP_M for p in others):
        c = _first_clear(road, hi, 1.0, others)
        if c is not None and math.isnan(c):
            return None
        hi = length if c is None else min(length, c + SEAM_MARGIN_M)
    if lo < END_SNAP_M:
        lo = 0.0
    if hi > length - END_SNAP_M:
        hi = length
    return lo, hi


# ---------------------------------------------------------------- build


def build_junctions(model: RoadModel) -> tuple[list[JunctionArea], list[Flag]]:
    """L6 junction and shared areas of the model (FR-SG-3, FR-SV-7). Deterministic; ids from content (D-011)."""
    roads = {r.id: r for r in model.roads}
    _SAMPLES.clear()
    nodes, flags = _touches(model)
    groups: list[tuple[list[_Node], dict[str, tuple[float, float]]]] = []
    stuck: list[_Node] = []
    for node in nodes:
        fps = {rid: _local_footprint(roads[rid], s) for rid, s in node.anchors.items()}
        intervals: dict[str, tuple[float, float]] = {}
        ok = True
        for rid, s in sorted(node.anchors.items()):
            iv = _interval(roads[rid], s, [fp for other, fp in fps.items() if other != rid])
            if iv is None:
                ok = False
                break
            intervals[rid] = iv
        if not ok:
            stuck.append(node)
            continue
        groups.append(([node], intervals))
    full = {rid: _footprint(r, 0.0, r.length) for rid, r in roads.items()}
    areas, group_of, settle_flags = _settle(model, roads, _merge(groups))
    flags.extend(settle_flags)
    zone_map: dict[str, list[tuple[float, float]]] = {}
    for area in areas:
        for rid, iv in group_of[area.id][1].items():
            zone_map.setdefault(rid, []).append(iv)
    for area in areas:
        flags.extend(_step_flags(area, roads, group_of[area.id][0]))
    shared, shared_flags = _shared(model, roads, full, areas, zone_map)
    flags.extend(shared_flags)
    placed = [Polygon(a.outline, a.holes) for a in [*areas, *shared]]
    for node in stuck:
        if any(p.buffer(TOUCH_LATERAL_M).contains(Point(node.x, node.y)) for p in placed):
            continue
        flags.append(Flag("FR-SG-3:no_clear_seam", min(node.anchors), "FR-SG-3",
                          f"roads {sorted(node.anchors)} meeting at ({node.x:.1f}, {node.y:.1f}) do not separate within "
                          f"{MAX_REACH_M:.0f} m (or one lies wholly on the other); no junction area"))
    out = sorted([*areas, *shared], key=lambda a: a.id)
    flags.extend(_overlaps_left(model, roads, out))
    return out, flags


def _merge(groups):
    """Merge node groups that share a road with overlapping intervals, or sit at the same point."""
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                (na, ia), (nb, ib) = groups[i], groups[j]
                close = any(math.hypot(a.x - b.x, a.y - b.y) <= NODE_MERGE_M for a in na for b in nb)
                shared = any(r in ib and ia[r][0] < ib[r][1] and ib[r][0] < ia[r][1] for r in ia)
                if close or shared:
                    groups[i] = (na + nb, _union_intervals(ia, ib))
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return groups


def _union_intervals(ia: dict[str, tuple[float, float]], ib: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    merged = dict(ia)
    for r, (lo, hi) in ib.items():
        merged[r] = (min(lo, merged[r][0]), max(hi, merged[r][1])) if r in merged else (lo, hi)
    return merged


def _regrow(roads: dict[str, Road], intervals: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]] | None:
    """Grow every participant's interval until its ends clear all other participants (D-069)."""
    for _ in range(MAX_ROUNDS):
        fps = {
            r: _footprint(roads[r], max(0.0, lo - SHARED_REACH_M), min(roads[r].length, hi + SHARED_REACH_M))
            for r, (lo, hi) in intervals.items()
        }
        changed = False
        out = {}
        for r, (lo, hi) in sorted(intervals.items()):
            iv = _grow(roads[r], lo, hi, [fp for o, fp in fps.items() if o != r])
            if iv is None or (iv[0] <= 0.0 and iv[1] >= roads[r].length):
                return None
            changed = changed or iv != (lo, hi)
            out[r] = iv
        intervals = out
        if not changed:
            return intervals
    return None


def _settle(model: RoadModel, roads: dict[str, Road], groups):
    """Build the areas of the node groups, merging areas that overlap (D-069).

    Returns the areas, each area's (nodes, intervals) by area id, and flags.
    """
    dropped: list[Flag] = []
    for _round in range(MAX_ROUNDS):
        built: list[tuple[int, JunctionArea, Polygon]] = []
        problems: list[Flag] = []
        for k, (group_nodes, intervals) in enumerate(groups):
            area, problem = _junction(model, roads, group_nodes, intervals)
            if area is None:
                problems.append(Flag("FR-SG-3:skipped", min(intervals), "FR-SG-3", f"junction of {sorted(intervals)} not built: {problem}"))
                continue
            built.append((k, area, Polygon(area.outline, area.holes)))
        action = _overlapping_pair(built)
        if action is None:
            return [a for _k, a, _p in built], {a.id: groups[k] for k, a, _p in built}, dropped + problems
        i, j = action[1], action[2]
        (na, ia), (nb, ib) = groups[i], groups[j]
        merged = _regrow(roads, _union_intervals(ia, ib))
        trial = None if merged is None else _junction(model, roads, na + nb, merged)[0]
        if trial is None:
            drop = max(i, j)
            dropped.append(Flag("FR-SG-3:overlap", min(groups[drop][1]), "FR-SG-3",
                                f"junction of {sorted(groups[drop][1])} overlaps the junction of {sorted(groups[min(i, j)][1])} "
                                "and the merged area cannot be built; not built"))
            del groups[drop]
        else:
            groups[min(i, j)] = (na + nb, merged)
            del groups[max(i, j)]
        groups = _merge(groups)
    raise RuntimeError("junction areas did not settle")  # pragma: no cover - MAX_ROUNDS merges always settle


def _overlapping_pair(built) -> tuple[str, int, int] | None:
    for x in range(len(built)):
        for y in range(x + 1, len(built)):
            if built[x][2].intersection(built[y][2]).area > 1e-6:
                return ("merge", built[x][0], built[y][0])
    return None


# ---------------------------------------------------------------- structures and levels


def _structure_reach(model: RoadModel, road: Road, st: Structure) -> tuple[float, float]:
    """Stations of the carried road a seam must stay out of: the span plus its skewed station ramps."""
    supports = sorted(st.supports, key=lambda sp: (sp.station, sp.id))
    xs = road.cross_section
    half = max(k.value for pw in (xs.width_left, xs.width_right) for k in pw.knots)
    zone = max((b.offset_m + b.base_width_m for b in xs.barriers), default=0.0)
    toe = TOE_ALLOWANCE_M if any(ss.road_id == road.id and ss.kind == "fill" for ss in model.side_slopes) else 0.0
    umax = 1.25 * (max(half, zone) + st.deck.overhang_m + toe) + 1.0
    out = []
    for end in (st.s_start, st.s_end):
        t = math.tan(min(supports, key=lambda sp: abs(sp.station - end)).skew_rad) if supports else 0.0
        out.append(RAMP_FACTOR * abs(t) * umax + RAMP_MIN_M if t != 0.0 else 0.0)
    return st.s_start - out[0] - STRUCTURE_CLEAR_M, st.s_end + out[1] + STRUCTURE_CLEAR_M


def _support_outline(road: Road, st: Structure, sp: Support) -> Polygon:
    a, b = support_block(st, sp)
    t = math.tan(sp.skew_rad)
    xs = road.cross_section
    ur, ul = -xs.width_right(sp.station) - st.deck.overhang_m, xs.width_left(sp.station) + st.deck.overhang_m
    corners = [(a + t * ur, ur), (b + t * ur, ur), (b + t * ul, ul), (a + t * ul, ul)]
    return Polygon([road.alignment.offset_point(min(max(s, 0.0), road.length), u) for s, u in corners]).buffer(0.05)


def _deck_outline(road: Road, st: Structure) -> Polygon:
    t = max((abs(math.tan(sp.skew_rad)) for sp in st.supports), default=0.0)
    xs = road.cross_section
    reach = t * (max(k.value for pw in (xs.width_left, xs.width_right) for k in pw.knots) + st.deck.overhang_m)
    return _footprint(road, max(0.0, st.s_start - reach), min(road.length, st.s_end + reach)).buffer(st.deck.overhang_m)


def _structure_conflict(model: RoadModel, roads: dict[str, Road], intervals: dict[str, tuple[float, float]], area: JunctionArea) -> str:
    """Why an area cannot stand here: a participant's deck or skewed stations, a support, or another deck at its level."""
    poly = Polygon(area.outline, area.holes)
    for st in sorted(model.structures, key=lambda s: s.id):
        road = roads.get(st.carried_road)
        if road is None:
            continue
        if st.carried_road in intervals:
            a, b = _structure_reach(model, road, st)
            lo, hi = intervals[st.carried_road]
            if lo < b and hi > a:
                return f"interval on {st.carried_road} reaches structure {st.id} or its skewed stations ({a:.1f}-{b:.1f})"
        for sp in st.supports:
            if _support_outline(road, st, sp).intersects(poly):
                return f"support {sp.id} stands in the area"
        if st.carried_road in intervals:
            continue
        deck = _deck_outline(road, st)
        if not deck.intersects(poly):
            continue
        for g in shapely.get_parts(deck.intersection(poly)):
            if g.is_empty or g.area <= 0.0:
                continue
            for x, y in [*g.exterior.coords[:-1][:: max(1, len(g.exterior.coords) // 20)], tuple(g.representative_point().coords[0])]:
                s, u = _project(road, x, y)
                if not st.s_start <= s <= st.s_end:
                    continue
                soffit = road.surface_z(s, u) - st.deck.depth_m
                if area.lifted_z(x, y) > soffit - LEVEL_DZ_M:
                    return f"deck of {st.id} is at the level of the area"
    return ""


def _level_problem(area: JunctionArea, roads: dict[str, Road], intervals: dict[str, tuple[float, float]]) -> str:
    """The lifted surface must stay at the level of every road it covers, never above or below it (FR-SG-3)."""
    for rid, (lo, hi) in sorted(intervals.items()):
        road = roads[rid]
        xs = road.cross_section
        n = max(1, math.ceil((hi - lo) / SECTION_SPACING_M))
        for j in range(n + 1):
            s = lo + (hi - lo) * j / n
            for u in (-0.9 * xs.width_right(s), 0.0, 0.9 * xs.width_left(s)):
                x, y = road.alignment.offset_point(s, u)
                if not area.contains(x, y):
                    continue
                dz = area.lifted_z(x, y) - road.surface_z(s, u)
                if abs(dz) > LEVEL_DZ_M:
                    return f"lifted surface is {dz:+.2f} m off {rid} at s={s:.1f}"
    return ""


# ---------------------------------------------------------------- one area


def _junction(model, roads, nodes, intervals) -> tuple[JunctionArea | None, str]:
    fps = [_footprint(roads[r], lo, hi) for r, (lo, hi) in sorted(intervals.items())]
    union = shapely.union_all(fps)
    if union.geom_type != "Polygon":
        return None, "participant pavements do not form one polygon"
    union = orient(union, 1.0)

    seam_defs: list[tuple[str, float, int]] = []
    for rid, (lo, hi) in sorted(intervals.items()):
        if lo > 0.0:
            seam_defs.append((rid, lo, 1))
        if hi < roads[rid].length:
            seam_defs.append((rid, hi, -1))

    ring = [tuple(map(float, p)) for p in union.exterior.coords[:-1]]
    marks: list[tuple[int, int, int]] = []  # (index of first end along the ring, index of second end, seam index)
    for k, (rid, s, _side) in enumerate(seam_defs):
        a, _c, b = _row(roads[rid], s)
        ia, ib = _nearest(ring, a), _nearest(ring, b)
        if ia is None or ib is None:
            return None, f"seam {rid}@{s:.2f} is not on the area boundary"
        n = len(ring)
        if (ib - ia) % n <= (ia - ib) % n:
            marks.append((ia, ib, k))
        else:
            marks.append((ib, ia, k))
    marks.sort()
    if not marks:
        return None, "no seams"
    n = len(ring)
    if any((a1 - a0) % n > 2 for a0, a1, _ in marks):
        return None, "seam rows are not simple boundary edges"

    outline: list[tuple[float, float]] = []
    rows: set[tuple[float, float]] = set()
    for idx, (a0, a1, k) in enumerate(marks):
        rid, s, _side = seam_defs[k]
        row = _row(roads[rid], s)
        if _close(ring[a0], row[2]):
            row = row[::-1]
        outline.extend(row)
        rows.update(_round_xy(p) for p in row)
        b0 = marks[(idx + 1) % len(marks)][0]
        path = [ring[(a1 + j) % n] for j in range(1, (b0 - a1) % n)]
        start, end = row[-1], ring[b0]
        outline.extend(_gore_closed(path, start, end, union))
    holes = [list(h.coords[:-1]) for h in union.interiors]
    outline, *holes = _snap_rings([outline, *holes], rows)
    poly = Polygon(outline, holes)
    if not poly.is_valid or not poly.exterior.is_ccw or poly.area < union.area - 1e-4:
        return None, "outline is not a valid polygon"

    seams = [_patch(roads[rid], s, side) for rid, s, side in seam_defs]
    sections = []
    for rid, (lo, hi) in sorted(intervals.items()):
        road = roads[rid]
        n_sec = max(1, math.ceil((hi - lo) / SECTION_SPACING_M))
        for j in range(n_sec + 1):
            s = lo + (hi - lo) * j / n_sec
            if any(r == rid and abs(s - ss) < SECTION_CLEAR_M for r, ss, _ in seam_defs):
                continue
            sections.append(_patch(road, s, 0))

    srcs = [_source(roads[r]) for r in sorted(intervals)]
    worst = max(srcs, key=lambda item: _TRUST[item[0]])
    prov = Provenance(
        worst[0],
        min(c for _s, c, _r in srcs),
        rule_id=(worst[2] or LIFT_RULE_ID) if worst[0] == Source.RULE else None,
        flags=["FR-SG-3:lifted_from_roads"],
    )
    key = sorted((rid, round(lo, 1), round(hi, 1)) for rid, (lo, hi) in intervals.items())
    area = JunctionArea(
        id=stable_id("jn", key),
        road_ids=sorted(intervals),
        outline=[(float(x), float(y)) for x, y in outline],
        prov=prov,
        kind=_kind(roads, nodes),
        seams=seams,
        holes=[[(float(x), float(y)) for x, y in h] for h in holes],
        sections=sections,
    )
    problem = _structure_conflict(model, roads, intervals, area) or _level_problem(area, roads, intervals)
    if problem:
        return None, problem
    return area, ""


# ---------------------------------------------------------------- shared pavement (D-070)


@dataclass
class _Stretch:
    """Overlap pieces of two roads at the same level, with each road's interval covering them."""

    a: str
    b: str
    pieces: list[Polygon]
    iv: dict[str, tuple[float, float]]


def _stations_of(road: Road, geom: Polygon) -> tuple[float, float]:
    coords = list(geom.exterior.coords)
    step = max(1, len(coords) // 60)
    st = [_project(road, x, y)[0] for x, y in coords[::step]]
    return min(st), max(st)


def _shared(model: RoadModel, roads: dict[str, Road], full: dict[str, Polygon], areas: list[JunctionArea], zone_map: dict[str, list[tuple[float, float]]]):
    """Shared areas for same-level pavement overlaps outside every junction area (D-070)."""
    flags: list[Flag] = []
    covered = shapely.union_all([Polygon(a.outline, a.holes) for a in areas]) if areas else Polygon()
    stretches: list[_Stretch] = []
    ids = sorted(roads)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if not full[a].intersects(full[b]):
                continue
            inter = full[a].intersection(full[b])
            if inter.area < SHARED_MIN_M2:
                continue
            if not covered.is_empty:
                inter = inter.difference(covered)
            parts = [g for g in shapely.get_parts(inter) if g.geom_type == "Polygon" and g.area >= SHARED_MIN_M2]
            for g in sorted(parts, key=lambda g: (round(g.centroid.x, 3), round(g.centroid.y, 3))):
                st = _stretch(roads, full, a, b, g, flags)
                if st is not None:
                    stretches.append(st)
    out: list[JunctionArea] = []
    for group in _stretch_groups(stretches):
        names = sorted({r for st in group for r in (st.a, st.b)})
        built, problem = _shared_group(model, roads, group, zone_map, covered, midway=True)
        if problem and "outline" in problem:
            # Near-touching split vertices: give each overlap wholly to the narrower road instead.
            built, fallback = _shared_group(model, roads, group, zone_map, covered, midway=False)
            if not fallback:
                flags.append(Flag("FR-SG-3:shared_to_narrower", names[0], "FR-SG-3",
                                  f"shared pavement of {names} could not be split midway ({problem}); the narrower road keeps it"))
            problem = fallback
        if problem:
            flags.append(Flag("FR-SG-3:shared_skipped", names[0], "FR-SG-3", f"shared pavement of {names} not split: {problem}"))
            continue
        out.extend(built)
        for st in group:
            area = sum(p.area for p in st.pieces)
            flags.append(Flag("FR-SG-3:shared_split", st.a, "FR-SG-3",
                              f"{st.a} and {st.b} share {area:.1f} m2 of pavement ({st.a} {st.iv[st.a][0]:.1f}-{st.iv[st.a][1]:.1f}, "
                              f"{st.b} {st.iv[st.b][0]:.1f}-{st.iv[st.b][1]:.1f}); split between them"))
            flags.extend(_shared_step(roads, st))
    return out, flags


def _stretch(roads, full, a: str, b: str, piece: Polygon, flags: list[Flag]) -> _Stretch | None:
    ra, rb = roads[a], roads[b]
    p = piece.representative_point()
    sa, ua = _project(ra, p.x, p.y)
    sb, ub = _project(rb, p.x, p.y)
    if abs(ra.surface_z(sa, ua) - rb.surface_z(sb, ub)) > LEVEL_DZ_M:
        return None  # one passes over the other
    if abs(math.sin(ra.alignment.evaluate(sa)[2] - rb.alignment.evaluate(sb)[2])) > math.sin(INTERSECTION_ANGLE):
        flags.append(Flag("FR-SG-3:crossing_overlap", a, "FR-SG-3",
                          f"{a} and {b} cross at the same level at ({p.x:.1f}, {p.y:.1f}) with no junction area"))
        return None
    span = {a: _stations_of(ra, piece), b: _stations_of(rb, piece)}
    iv = {}
    for r, o in ((a, b), (b, a)):
        lo_o, hi_o = span[o]
        other = full[o] if roads[o].length <= 2 * SHARED_REACH_M else _footprint(
            roads[o], max(0.0, lo_o - SHARED_REACH_M), min(roads[o].length, hi_o + SHARED_REACH_M))
        grown = _grow(roads[r], *span[r], [other])
        if grown is None:
            flags.append(Flag("FR-SG-3:shared_skipped", r, "FR-SG-3",
                              f"{a} and {b} overlap at ({p.x:.1f}, {p.y:.1f}) and do not separate within {MAX_REACH_M:.0f} m"))
            return None
        iv[r] = grown
    return _Stretch(a, b, [piece], iv)


def _stretch_groups(stretches: list[_Stretch]) -> list[list[_Stretch]]:
    """Stretches that share a road with overlapping intervals are split together."""
    parent = list(range(len(stretches)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, si in enumerate(stretches):
        for j in range(i + 1, len(stretches)):
            sj = stretches[j]
            if any(r in sj.iv and si.iv[r][0] <= sj.iv[r][1] and sj.iv[r][0] <= si.iv[r][1] for r in si.iv):
                parent[find(j)] = find(i)
    groups: dict[int, list[_Stretch]] = {}
    for i, st in enumerate(stretches):
        groups.setdefault(find(i), []).append(st)
    return [groups[k] for k in sorted(groups)]


def _merged_intervals(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for lo, hi in sorted(ivs):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def _shared_group(model, roads, group: list[_Stretch], zone_map, covered, *, midway: bool) -> tuple[list[JunctionArea], str]:
    names = sorted({r for st in group for r in (st.a, st.b)})
    per_road: dict[str, list[tuple[float, float]]] = {r: _merged_intervals([st.iv[r] for st in group if r in st.iv]) for r in names}
    for r, ivs in per_road.items():
        for lo, hi in ivs:
            if any(lo < zb and za < hi for za, zb in zone_map.get(r, [])):
                return [], f"interval {lo:.1f}-{hi:.1f} on {r} meets a junction area"
            for st in model.structures:
                if st.carried_road == r:
                    a, b = _structure_reach(model, roads[r], st)
                    if lo < b and hi > a:
                        return [], f"interval on {r} reaches structure {st.id} or its skewed stations ({a:.1f}-{b:.1f})"
    # Pieces come from the very footprints the regions are cut from, so the split leaves no sliver.
    fps = {(r, iv): _footprint(roads[r], *iv) for r in names for iv in per_road[r]}
    keys = sorted(fps)
    given: dict[tuple[str, tuple[float, float]], list[Polygon]] = {k: [] for k in keys}  # overlap given to others
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            if ka[0] == kb[0] or not fps[ka].intersects(fps[kb]):
                continue
            inter = fps[ka].intersection(fps[kb])
            if not covered.is_empty:
                inter = inter.difference(covered)
            ra, rb = roads[ka[0]], roads[kb[0]]
            for piece in shapely.get_parts(inter):
                if piece.geom_type != "Polygon" or piece.area <= 0.0:
                    continue
                q = piece.representative_point()
                if abs(ra.surface_z(*_project(ra, q.x, q.y)) - rb.surface_z(*_project(rb, q.x, q.y))) > LEVEL_DZ_M:
                    continue
                qa, qb = _split_piece(ra, rb, piece, midway)
                given[ka].append(qb)
                given[kb].append(qa)
    regions: list[tuple[str, float, float, Polygon]] = []
    for key in keys:
        r, (lo, hi) = key
        parts = [g for g in given[key] if not g.is_empty]
        region = fps[key].difference(shapely.union_all(parts)) if parts else fps[key]
        region = _largest(region)
        if region is None:
            return [], f"{r} {lo:.1f}-{hi:.1f} is cut into pieces"
        if not covered.is_empty and region.intersection(covered).area > 1e-4:
            return [], f"{r} {lo:.1f}-{hi:.1f} runs into a junction area"
        regions.append((r, lo, hi, orient(region, 1.0)))
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if regions[i][3].intersection(regions[j][3]).area > 1e-4:
                return [], f"split pieces of {regions[i][0]} and {regions[j][0]} overlap (three roads share pavement)"
    rows: set[tuple[float, float]] = set()
    for r, lo, hi, _poly in regions:
        for s in (lo, hi):
            if 0.0 < s < roads[r].length:
                rows.update(_round_xy(p) for p in _row(roads[r], s))
    rings: list[list[tuple[float, float]]] = []
    index: list[tuple[int, int]] = []  # (region, hole index or -1) per ring
    for k, (_r, _lo, _hi, poly) in enumerate(regions):
        rings.append([tuple(map(float, p)) for p in poly.exterior.coords[:-1]])
        index.append((k, -1))
        for h, hole in enumerate(poly.interiors):
            rings.append([tuple(map(float, p)) for p in hole.coords[:-1]])
            index.append((k, h))
    rings = _conform(_snap_rings(rings, rows))
    out = []
    for k, (r, lo, hi, _poly) in enumerate(regions):
        outline = next(rg for rg, (kk, h) in zip(rings, index, strict=True) if kk == k and h == -1)
        holes = [rg for rg, (kk, h) in zip(rings, index, strict=True) if kk == k and h >= 0]
        area, problem = _shared_area(roads[r], lo, hi, outline, holes, sorted(set(names) - {r}))
        if area is None:
            return [], problem
        out.append(area)
    return out, ""


def _largest(geom) -> Polygon | None:
    parts = [g for g in shapely.get_parts(geom) if g.geom_type == "Polygon" and g.area > SHARED_MIN_M2]
    if len(parts) != 1:
        return None
    return parts[0]


def _split_piece(ra: Road, rb: Road, piece: Polygon, midway: bool = True) -> tuple[Polygon, Polygon]:
    """(part kept by ra, part kept by rb) of their overlap: midway between their edges, or all to the narrower road if inside the other."""
    p = piece.representative_point()

    def width(road: Road) -> float:
        s = _project(road, p.x, p.y)[0]
        return road.cross_section.width_left(s) + road.cross_section.width_right(s)

    swap = (width(rb), rb.id) < (width(ra), ra.id)
    narrow, wide = (rb, ra) if swap else (ra, rb)
    keep_narrow, keep_wide = _split_by_chords(narrow, wide, piece) if midway else (piece, Polygon())
    return (keep_wide, keep_narrow) if swap else (keep_narrow, keep_wide)


def _split_by_chords(narrow: Road, wide: Road, piece: Polygon) -> tuple[Polygon, Polygon]:
    xs = narrow.cross_section
    lo, hi = _stations_of(narrow, piece)
    lo, hi = max(0.0, lo - SHARED_STEP_M), min(narrow.length, hi + SHARED_STEP_M)
    n = max(1, math.ceil((hi - lo) / SHARED_STEP_M))
    reach = 2.0 * (xs.width_left(lo) + xs.width_right(lo)) + 50.0
    mids: list[tuple[float, float, tuple[float, float]]] = []
    toward: list[float] = []  # +1 when the narrow road's side of the midline lies at larger offsets
    n_chords = n_inside = 0
    for j in range(n + 1):
        s = lo + (hi - lo) * j / n
        wr, wl = xs.width_right(s), xs.width_left(s)
        line = LineString([narrow.alignment.offset_point(s, -wr - reach), narrow.alignment.offset_point(s, wl + reach)])
        cut = line.intersection(piece)
        segs = [g for g in shapely.get_parts(cut) if g.geom_type == "LineString" and g.length >= MIN_CHORD_M]
        if not segs:
            continue
        seg = max(segs, key=lambda g: g.length)
        (x0, y0), (x1, y1) = seg.coords[0], seg.coords[-1]
        u0 = _project(narrow, x0, y0)[1]
        u1 = _project(narrow, x1, y1)[1]
        if u1 < u0:
            u0, u1 = u1, u0
            x0, y0, x1, y1 = x1, y1, x0, y0
        own0 = abs(u0 + wr) <= EDGE_TOL_M or abs(u0 - wl) <= EDGE_TOL_M
        own1 = abs(u1 + wr) <= EDGE_TOL_M or abs(u1 - wl) <= EDGE_TOL_M
        n_chords += 1
        n_inside += own0 and own1
        um = 0.5 * (u0 + u1)
        mids.append((s, um, (0.5 * (x0 + x1), 0.5 * (y0 + y1))))
        if own0 != own1:
            # The chord end on the wide road's edge lies deep inside the narrow road: that half is the narrow road's.
            toward.append(1.0 if own0 else -1.0)
    if n_chords == 0 or n_inside >= CONTAINED_FRACTION * n_chords or len(mids) < 2:
        return piece, Polygon()
    side = 1.0 if sum(toward) >= 0.0 else -1.0
    pts = [m[2] for m in mids]
    (ax, ay), (bx, by) = pts[0], pts[1]
    (cx, cy), (dx, dy) = pts[-2], pts[-1]
    la, lb = math.hypot(ax - bx, ay - by) or 1.0, math.hypot(dx - cx, dy - cy) or 1.0
    ext = SPLIT_EXTEND_M + 2.0 * SHARED_STEP_M
    head = (ax + (ax - bx) / la * ext, ay + (ay - by) / la * ext)
    tail = (dx + (dx - cx) / lb * ext, dy + (dy - cy) / lb * ext)
    blade = LineString([head, *pts, tail])
    if not blade.is_simple:
        return piece, Polygon()
    parts = [g for g in shapely.get_parts(split_polygon(piece, blade)) if g.area > 0.0]
    if len(parts) < 2:
        return piece, Polygon()
    ss = np.array([m[0] for m in mids])
    us = np.array([m[1] for m in mids])
    keep_n, keep_w = [], []
    for g in parts:
        q = g.representative_point()
        s, u = _project(narrow, q.x, q.y)
        um = float(np.interp(s, ss, us))
        (keep_n if (u - um) * side > 0.0 else keep_w).append(g)
    if not keep_n or not keep_w:
        return piece, Polygon()
    return shapely.union_all(keep_n), shapely.union_all(keep_w)


def _shared_area(road: Road, lo: float, hi: float, outline, holes, partners: list[str]) -> tuple[JunctionArea | None, str]:
    seam_defs = [(s, side) for s, side in ((lo, 1), (hi, -1)) if 0.0 < s < road.length]
    for s, _side in seam_defs:
        row = [_round_xy(p) for p in _row(road, s)]
        idx = [outline.index(p) if p in outline else None for p in row]
        if None in idx:
            return None, f"seam {road.id}@{s:.2f} is not on the split outline"
        n = len(outline)
        a, b, c = idx  # type: ignore[misc]
        if not ((b - a) % n == 1 and (c - b) % n == 1) and not ((a - b) % n == 1 and (b - c) % n == 1):
            return None, f"seam row {road.id}@{s:.2f} is not a boundary edge of the split outline"
    poly = Polygon(outline, holes)
    if not poly.is_valid or not poly.exterior.is_ccw:
        return None, f"split outline of {road.id} is not a valid polygon"
    seams = [_patch(road, s, side) for s, side in seam_defs]
    n_sec = max(1, math.ceil((hi - lo) / SECTION_SPACING_M))
    sections = []
    for j in range(n_sec + 1):
        s = lo + (hi - lo) * j / n_sec
        if any(abs(s - ss) < SECTION_CLEAR_M for ss, _ in seam_defs):
            continue
        sections.append(_patch(road, s, 0))
    src, conf, rule = _source(road)
    prov = Provenance(src, conf, rule_id=(rule or SHARED_RULE_ID) if src == Source.RULE else None,
                      flags=["FR-SG-3:shared_split", "FR-SG-3:lifted_from_roads", *(f"FR-SG-3:shared_with:{p}" for p in partners)])
    area = JunctionArea(
        id=stable_id("jn", "shared", road.id, round(lo, 1), round(hi, 1)),
        road_ids=[road.id],
        outline=[(float(x), float(y)) for x, y in outline],
        prov=prov,
        kind="shared",
        seams=seams,
        holes=[[(float(x), float(y)) for x, y in h] for h in holes],
        sections=sections,
    )
    return area, ""


def _shared_step(roads: dict[str, Road], st: _Stretch) -> list[Flag]:
    """FR-AF-3: the two surfaces meet at the split; report where they disagree."""
    ra, rb = roads[st.a], roads[st.b]
    worst = 0.0
    for piece in st.pieces:
        for x, y in piece.exterior.coords[:: max(1, len(piece.exterior.coords) // 30)]:
            sa, ua = _project(ra, x, y)
            sb, ub = _project(rb, x, y)
            dz = ra.surface_z(sa, ua) - rb.surface_z(sb, ub)
            worst = dz if abs(dz) > abs(worst) else worst
    if abs(worst) <= STEP_FLAG_M:
        return []
    return [Flag("FR-AF-3:shared_step", st.a, "FR-AF-3", f"{st.a} is up to {worst:+.3f} m off {st.b} where their shared pavement is split")]


def _overlaps_left(model: RoadModel, roads: dict[str, Road], areas: list[JunctionArea]) -> list[Flag]:
    """FR-SV-7: same-level pavement overlaps that no area resolves (each road's swept part only)."""
    zones: dict[str, list[tuple[float, float]]] = {}
    for area in areas:
        for rid in area.road_ids:
            lo = [sm.s for sm in area.seams if sm.road_id == rid and sm.side > 0]
            hi = [sm.s for sm in area.seams if sm.road_id == rid and sm.side < 0]
            zones.setdefault(rid, []).append((min(lo) if lo else 0.0, max(hi) if hi else roads[rid].length))
    swept: dict[str, list[Polygon]] = {}
    for rid, road in roads.items():
        pieces = [(0.0, road.length)]
        for za, zb in sorted(zones.get(rid, [])):
            nxt = []
            for a, b in pieces:
                if zb <= a or za >= b:
                    nxt.append((a, b))
                    continue
                if za > a:
                    nxt.append((a, za))
                if zb < b:
                    nxt.append((zb, b))
            pieces = nxt
        swept[rid] = [_footprint(road, a, b) for a, b in pieces if b - a > 0.05]
    polys = [(f"road {rid}", rid, p) for rid in sorted(swept) for p in swept[rid]]
    polys += [(f"area {a.id}", a.id, Polygon(a.outline, a.holes)) for a in areas]
    out = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            (na, ida, pa), (nb, idb, pb) = polys[i], polys[j]
            if ida == idb or not pa.intersects(pb):
                continue
            inter = pa.intersection(pb)
            if inter.area < SHARED_MIN_M2:
                continue
            q = inter.representative_point()
            za = _top_at(roads, areas, ida, q.x, q.y)
            zb = _top_at(roads, areas, idb, q.x, q.y)
            if abs(za - zb) > LEVEL_DZ_M:
                continue
            out.append(Flag("FR-SV-7:pavement_overlap", min(ida, idb), "FR-SV-7",
                            f"{na} and {nb} overlap by {inter.area:.1f} m2 at ({q.x:.1f}, {q.y:.1f}) at the same level"))
    return out


def _top_at(roads: dict[str, Road], areas: list[JunctionArea], eid: str, x: float, y: float) -> float:
    if eid in roads:
        s, u = _project(roads[eid], x, y)
        return roads[eid].surface_z(s, u)
    return next(a for a in areas if a.id == eid).lifted_z(x, y)


# ---------------------------------------------------------------- outline hygiene


def _round_xy(p) -> tuple[float, float]:
    return (round(float(p[0]), FLOAT_DIGITS), round(float(p[1]), FLOAT_DIGITS))


def _snap_rings(rings: list[list[tuple[float, float]]], protected: set[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Round to the model's precision and merge vertices closer than MIN_EDGE_M (protected ones win); drop repeats and spikes."""
    rings = [[_round_xy(p) for p in r] for r in rings]
    cell = MIN_EDGE_M
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    rep: dict[tuple[float, float], tuple[float, float]] = {}
    for p in sorted({p for r in rings for p in r}, key=lambda q: (q not in protected, q)):
        kx, ky = math.floor(p[0] / cell), math.floor(p[1] / cell)
        hit = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in grid.get((kx + dx, ky + dy), []):
                    if math.hypot(p[0] - q[0], p[1] - q[1]) < MIN_EDGE_M:
                        hit = q
                        break
                if hit:
                    break
            if hit:
                break
        if hit is None:
            grid.setdefault((kx, ky), []).append(p)
            rep[p] = p
        else:
            rep[p] = hit
    out = []
    for r in rings:
        nr: list[tuple[float, float]] = []
        for p in r:
            q = rep[p]
            if not nr or nr[-1] != q:
                nr.append(q)
        while len(nr) > 1 and nr[0] == nr[-1]:
            nr.pop()
        changed = True
        while changed and len(nr) >= 3:
            changed = False
            m = len(nr)
            for i in range(m):
                if nr[i - 1] == nr[(i + 1) % m]:  # spike a-b-a
                    for k in sorted({i, (i + 1) % m}, reverse=True):
                        del nr[k]
                    changed = True
                    break
                if nr[i] not in protected and _needle(nr[i - 1], nr[i], nr[(i + 1) % m]):
                    del nr[i]
                    changed = True
                    break
        out.append(nr)
    return out


def _needle(a, b, c) -> bool:
    """b is the tip of a zero-width spike: off the line ac by under MIN_EDGE_M and beyond its ends."""
    dx, dy = c[0] - a[0], c[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 < 1e-18:
        return True
    t = ((b[0] - a[0]) * dx + (b[1] - a[1]) * dy) / length2
    height = abs(dx * (b[1] - a[1]) - dy * (b[0] - a[0])) / math.sqrt(length2)
    return height < MIN_EDGE_M and not 0.0 <= t <= 1.0


def _conform(rings: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Put every vertex lying on another ring's edge into that edge, so touching outlines share vertices (D-021)."""
    pts = sorted({p for r in rings for p in r})
    if not pts:
        return rings
    arr = np.array(pts)
    tree = shapely.STRtree(shapely.points(arr))
    out = []
    for r in rings:
        own = set(r)
        nr: list[tuple[float, float]] = []
        for i, p in enumerate(r):
            q = r[(i + 1) % len(r)]
            nr.append(p)
            seg = LineString([p, q])
            near = tree.query(seg.buffer(CONFORM_TOL_M))
            on = []
            for k in near:
                v = pts[int(k)]
                if v in own:
                    continue
                if seg.distance(Point(v)) <= CONFORM_TOL_M:
                    on.append((seg.project(Point(v)), v))
            nr.extend(v for _t, v in sorted(on))
        out.append(nr)
    return out


def _nearest(ring, p, tol: float = 1e-6) -> int | None:
    best, bi = math.inf, None
    for i, q in enumerate(ring):
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        if d < best:
            best, bi = d, i
    return bi if best <= tol else None


def _close(a, b, tol: float = 1e-6) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


def _gore_closed(path, start, end, union: Polygon) -> list[tuple[float, float]]:
    """The boundary between two seams, or nothing (a straight closure) if it bounds a gore pocket outside the union."""
    if not path or math.hypot(end[0] - start[0], end[1] - start[1]) > GORE_CLOSURE_MAX_M:
        return list(path)
    pocket = Polygon([start, *path, end])
    if not pocket.is_valid:
        pocket = shapely.make_valid(pocket)
    if pocket.area > 1e-6 and pocket.difference(union).area >= GORE_OUTSIDE_FRACTION * pocket.area:
        return []
    return list(path)


def _patch(road: Road, s: float, side: int) -> JunctionSeam:
    """Road surface at s lifted into a seam patch; derivatives taken on the junction side (D-046)."""
    xs = road.cross_section
    toward = 1.0 if side >= 0 else -1.0
    h = DERIV_STEP_M
    s2 = min(max(s + toward * h, 0.0), road.length)
    if abs(s2 - s) < 1e-6:
        s2 = s - toward * h
    ds = s2 - s
    x, y, heading, _k = road.alignment.evaluate(s)
    k = road.alignment.evaluate(min(max(s + toward * 1e-3, 0.0), road.length))[3]
    return JunctionSeam(
        road_id=road.id,
        s=s,
        u0=-xs.width_right(s),
        u1=xs.width_left(s),
        side=side,
        x=x,
        y=y,
        heading=heading,
        curvature=k,
        z=road.profile.z(s),
        grade=road.profile.grade(s),
        grade_rate=(road.profile.grade(s2) - road.profile.grade(s)) / ds,
        slope_left=xs.slope_left(s),
        slope_right=xs.slope_right(s),
        slope_left_rate=(xs.slope_left(s2) - xs.slope_left(s)) / ds,
        slope_right_rate=(xs.slope_right(s2) - xs.slope_right(s)) / ds,
        thickness=xs.pavement_thickness(s),
    )


def _kind(roads: dict[str, Road], nodes: list[_Node]) -> str:
    kinds = set()
    for node in nodes:
        headings = []
        n_in = n_out = 0
        for rid, s in node.anchors.items():
            road = roads[rid]
            headings.append(road.alignment.evaluate(s)[2])
            if s <= 0.0:
                n_out += 1
            elif s >= road.length:
                n_in += 1
            else:
                n_in += 1
                n_out += 1
        # Angle between the direction lines, ignoring which way each road is drawn.
        wide = any(abs(math.sin(a - b)) > math.sin(INTERSECTION_ANGLE) for i, a in enumerate(headings) for b in headings[i + 1 :])
        if wide:
            kinds.add("intersection")
        elif n_in == 1 and n_out >= 2:
            kinds.add("diverge")
        elif n_out == 1 and n_in >= 2:
            kinds.add("merge")
        else:
            kinds.add("gore")
    return kinds.pop() if len(kinds) == 1 else ("intersection" if "intersection" in kinds else "gore")


def _step_flags(area: JunctionArea, roads: dict[str, Road], nodes: list[_Node]) -> list[Flag]:
    """FR-AF-3: participants whose surfaces disagree at their meeting point; the area blends them."""
    out = []
    for node in nodes:
        zs = []
        for rid in sorted(node.anchors):
            road = roads[rid]
            s, u = _project(road, node.x, node.y)
            zs.append((road.surface_z(s, u), rid))
        lo, hi = min(zs), max(zs)
        if hi[0] - lo[0] > STEP_FLAG_M:
            out.append(Flag("FR-AF-3:junction_step", area.id, "FR-AF-3",
                            f"{hi[1]} is {hi[0] - lo[0]:.3f} m above {lo[1]} at ({node.x:.1f}, {node.y:.1f}); the junction surface blends them"))
    return out
