"""L6 junction areas from a planar partition of the road edge lines (FR-SG-3, AC-H4, D-045).

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
   them (D-045), so elevation and slope match each road at its seam.

Junction areas never touch a structure: a node whose intervals reach a deck or a
support footprint is flagged and left to the road sweeps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from roader.model.provenance import Provenance, Source
from roader.model.schema import Flag, JunctionArea, JunctionSeam, Road, RoadModel
from roader.model.serialize import stable_id

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
STRUCTURE_CLEAR_M = 30.0  # seams stay off decks and their skewed station ramps
INTERSECTION_ANGLE = math.radians(45.0)
STEP_FLAG_M = 0.05  # participants disagreeing by more at the node are flagged (FR-AF-3)
DERIV_STEP_M = 0.5
GORE_OUTSIDE_FRACTION = 0.99
# A gore closure joins two seam rows that cleared each other by about GORE_GAP_M; a longer
# chord would cut across the outside of a curve, not a gore.
GORE_CLOSURE_MAX_M = 6.0
_TRUST = {Source.MEASURED: 0, Source.INVENTORY: 1, Source.VECTOR: 2, Source.RULE: 3}
LIFT_RULE_ID = "junction.lift"


@dataclass
class _Node:
    x: float
    y: float
    anchors: dict[str, float]  # road id -> anchor station


# ---------------------------------------------------------------- road geometry helpers


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


def _touches(model: RoadModel) -> list[_Node]:
    """Road ends on another road's pavement at grade, and diverge/merge links."""
    roads = {r.id: r for r in model.roads}
    nodes: list[_Node] = []
    for link in sorted(model.links, key=lambda lk: lk.id):
        if link.kind not in ("diverge", "merge") or link.from_road not in roads or link.to_road not in roads:
            continue
        if link.kind == "diverge":
            branch, end, trunk, s_t = roads[link.to_road], link.to_end, roads[link.from_road], link.from_s
        else:
            branch, end, trunk, s_t = roads[link.from_road], link.from_end, roads[link.to_road], link.to_s
        s_b = 0.0 if end == "start" else branch.length
        x, y = branch.alignment.offset_point(s_b, 0.0)
        if s_t is None:
            s_t, _u = trunk.alignment.project(x, y)
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
                s, u = other.alignment.project(x, y)
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
    return nodes


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


# ---------------------------------------------------------------- build


def build_junctions(model: RoadModel) -> tuple[list[JunctionArea], list[Flag]]:
    """L6 junction areas of the model (FR-SG-3). Deterministic; ids from content (D-011)."""
    roads = {r.id: r for r in model.roads}
    flags: list[Flag] = []
    nodes = _touches(model)
    groups: list[tuple[list[_Node], dict[str, tuple[float, float]]]] = []
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
            flags.append(Flag("FR-SG-3:no_clear_seam", min(node.anchors), "FR-SG-3",
                              f"roads {sorted(node.anchors)} meeting at ({node.x:.1f}, {node.y:.1f}) do not separate within "
                              f"{MAX_REACH_M:.0f} m (or one lies wholly on the other); no junction area"))
            continue
        groups.append(([node], intervals))
    groups = _merge(groups)

    out: list[JunctionArea] = []
    taken: list[Polygon] = []
    for group_nodes, intervals in groups:
        area, problem = _junction(model, roads, group_nodes, intervals)
        if area is None:
            first = min(intervals)
            flags.append(Flag("FR-SG-3:skipped", first, "FR-SG-3", f"junction of {sorted(intervals)} not built: {problem}"))
            continue
        poly = Polygon(area.outline, area.holes)
        if any(poly.intersection(t).area > 1e-6 for t in taken):
            flags.append(Flag("FR-SG-3:overlap", area.id, "FR-SG-3", "junction area overlaps another one; not built"))
            continue
        taken.append(poly)
        flags.extend(_step_flags(area, roads, group_nodes))
        out.append(area)
    out.sort(key=lambda a: a.id)
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
                    merged = dict(ia)
                    for r, (lo, hi) in ib.items():
                        merged[r] = (min(lo, merged[r][0]), max(hi, merged[r][1])) if r in merged else (lo, hi)
                    groups[i] = (na + nb, merged)
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return groups


def _structure_conflict(model: RoadModel, roads: dict[str, Road], intervals: dict[str, tuple[float, float]], poly: Polygon) -> str:
    for st in model.structures:
        if st.carried_road in intervals:
            lo, hi = intervals[st.carried_road]
            if lo < st.s_end + STRUCTURE_CLEAR_M and hi > st.s_start - STRUCTURE_CLEAR_M:
                return f"interval on {st.carried_road} reaches structure {st.id}"
        road = roads.get(st.carried_road)
        if road is None:
            continue
        for sp in st.supports:
            half = 0.5 * max(sp.cap_width_m, sp.thickness_m or 0.0) + 1.0
            a, b = max(0.0, sp.station - half - 2.0), min(road.length, sp.station + half + 2.0)
            if b <= a:
                continue
            fp = _footprint(road, a, b).buffer(st.deck.overhang_m + 1.0)
            if fp.intersects(poly):
                return f"support {sp.id} stands in the area"
    return ""


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
    for idx, (a0, a1, k) in enumerate(marks):
        rid, s, _side = seam_defs[k]
        row = _row(roads[rid], s)
        if _close(ring[a0], row[2]):
            row = row[::-1]
        outline.extend(row)
        b0 = marks[(idx + 1) % len(marks)][0]
        path = [ring[(a1 + j) % n] for j in range(1, (b0 - a1) % n)]
        start, end = row[-1], ring[b0]
        outline.extend(_gore_closed(path, start, end, union))
    holes = [list(h.coords[:-1]) for h in union.interiors]
    poly = Polygon(outline, holes)
    if not poly.is_valid or not poly.exterior.is_ccw or poly.area < union.area - 1e-6:
        return None, "outline is not a valid polygon"
    conflict = _structure_conflict(model, roads, intervals, poly)
    if conflict:
        return None, conflict

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
        holes=[[(float(x), float(y)) for x, y in h.coords[:-1]] for h in union.interiors],
        sections=sections,
    )
    return area, ""


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
    """Road surface at s lifted into a seam patch; derivatives taken on the junction side (D-045)."""
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
            s, u = road.alignment.project(node.x, node.y)
            zs.append((road.surface_z(s, u), rid))
        lo, hi = min(zs), max(zs)
        if hi[0] - lo[0] > STEP_FLAG_M:
            out.append(Flag("FR-AF-3:junction_step", area.id, "FR-AF-3",
                            f"{hi[1]} is {hi[0] - lo[0]:.3f} m above {lo[1]} at ({node.x:.1f}, {node.y:.1f}); the junction surface blends them"))
    return out
