"""L0 links between roads: where one road's end joins another road (FR-AF-3, AC-H4, D-035).

Topology comes from the input vectors: a road end within ``TOPO_M`` of another
road's input centerline is a junction (OSM ways share the node). The road that
ends there is the *child*; the road it leaves or joins is the *parent*, chosen
by kind (mainline before ramp and connector before crossing street), so a ramp
always hangs off its mainline. A ramp or connector ending on a crossing street
is an at-grade terminal: the street is its parent, after a mainline but before
another ramp ending at the same node (D-124). Between roads of one kind the parent is the one
whose interior holds the node, or the longer one when both end there.

Roads are fitted parents first (``fit_network``), and a child's alignment end is
pinned to the node's position on its parent's fitted reference line, so the two
reference lines meet exactly and the child's seam section lies on the parent's
surface on each side of its crown. Where the child leaves within
``PIN_HEADING_MAX`` of the parent's direction it is also pinned to the parent's
heading and curvature there: a ramp leaving a crowned carriageway at an angle,
or on a different curvature, cannot match the parent's surface slope across
the whole seam. Seam stations (``find_links``) are the child's end and its
projection onto the parent.

Link direction follows traffic: a child starting on its parent is a ``diverge``
(parent -> child, ``to_s = 0``); a child ending on its parent is a ``merge``
(child -> parent, ``from_s`` = child length). Where the parent also ends at the
node the kind is ``continue``. A child end that does not lie on the parent's
fitted pavement gets no link and a ``FR-AF-3:no_seam`` flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely

from roader.conflate.graph import RoadSeed
from roader.conflate.register import Neighbour
from roader.fit.median import split_twin_medians
from roader.fit.road import RoadFit, fit_road
from roader.model.provenance import Provenance, Source
from roader.model.schema import Crossing, Flag, Link, Road
from roader.model.serialize import stable_id

TOPO_M = 2.0  # input-vector distance that makes a junction
SEAM_PAVEMENT_PAD_M = 0.5
JUNCTION_DZ_M = 1.0  # linked roads crossing in plan by less than this are the junction overlap, not a grade separation
RANK = {"mainline": 0, "ramp": 1, "connector": 1, "crossing_street": 2}
TERMINAL_RANK = 0.5
PIN_HEADING_MAX = math.radians(25.0)
HEADING_REACH_M = 20.0


@dataclass
class Junction:
    """A child road end on a parent road, from the input vectors."""

    child: str
    end: str  # start | end
    parent: str
    at_parent_end: bool


def _rank(kind: str) -> int:
    return RANK.get(kind, 2)


def junctions(seeds: list[RoadSeed]) -> list[Junction]:
    seeds = sorted(seeds, key=lambda s: s.id)
    lines = {s.id: shapely.LineString(s.line.xy) for s in seeds}
    ends = {s.id: (shapely.Point(s.line.xy[0]), shapely.Point(s.line.xy[-1])) for s in seeds}
    out = []
    for child in seeds:
        for e, point in zip(("start", "end"), ends[child.id], strict=True):
            best = None
            for parent in seeds:
                if parent is child:
                    continue
                d = lines[parent.id].distance(point)
                if d > TOPO_M:
                    continue
                at_end = min(point.distance(p) for p in ends[parent.id]) <= TOPO_M
                rc, rp = _rank(child.kind), _rank(parent.kind)
                if rc == RANK["ramp"] and parent.kind == "crossing_street":
                    rp = TERMINAL_RANK  # an at-grade ramp terminal: the street ranks after mainlines, before ramps (D-124)
                elif rp > rc:
                    continue
                if rp == rc and at_end and (parent.line.length, child.id) < (child.line.length, parent.id):
                    continue
                key = (rp, d, parent.id)
                if best is None or key < best[0]:
                    best = (key, parent.id, at_end)
            if best is not None:
                out.append(Junction(child.id, e, best[1], best[2]))
    return out


def fit_network(seeds: list[RoadSeed], ev, rules, *, strip_factor: float = 1.0) -> tuple[list[RoadFit], list[Junction]]:
    """Every seed fitted, parents before children; returns fits in seed order and the junctions."""
    js = junctions(seeds)
    depth = {s.id: 0 for s in seeds}
    for _ in range(len(js)):
        for j in js:
            depth[j.child] = max(depth[j.child], min(depth[j.parent] + 1, len(seeds)))
    by_id = {s.id: s for s in seeds}
    fitted: dict[str, RoadFit] = {}
    for sid in sorted(by_id, key=lambda i: (depth[i], i)):
        seed = by_id[sid]
        anchors = [None, None]
        for j in js:
            if j.child != sid or j.parent not in fitted:
                continue
            node = seed.line.xy[0] if j.end == "start" else seed.line.xy[-1]
            parent = fitted[j.parent].road
            ps, _pu = parent.alignment.project(float(node[0]), float(node[1]))
            x, y, hp, kp = parent.alignment.evaluate(ps)
            hc = _input_heading(seed, j.end)
            # A shallow junction starts tangent to the parent and on its curvature, so the
            # two surfaces share their slope across the whole seam (AC-H4); a steep one (a
            # ramp terminal) keeps its own heading.
            heading = curvature = None
            for cand, k in ((hp, kp), (hp + math.pi, -kp)) if not parent.oneway else ((hp, kp),):
                if abs(math.remainder(hc - cand, 2.0 * math.pi)) <= PIN_HEADING_MAX:
                    heading, curvature = math.remainder(cand, 2.0 * math.pi), k
            anchors[0 if j.end == "start" else 1] = (x, y, heading, curvature)
        neighbours = [Neighbour(other.line, least_half_width(other, rules)) for other in seeds if other.id != sid]
        fitted[sid] = fit_road(seed, ev, rules, strip_factor=strip_factor, anchors=tuple(anchors), neighbours=neighbours)
    fits = [fitted[s.id] for s in seeds]
    # Twin carriageways stop at their median (L3, FR-XS-4): with every road fitted, so both inner edges are known.
    split_twin_medians(fits, ev.xyz[~np.asarray(ev.transient, dtype=bool)])
    return fits, js


def least_half_width(seed: RoadSeed, rules) -> float:
    """Half the narrowest pavement the road can have: tagged (or least) lane count at the rule's least lane width."""
    counts = []
    for span in seed.spans:
        try:
            counts.append(int(span.tags.get("lanes", "")))
        except ValueError:
            continue
    lanes = min(counts) if counts else (1 if seed.oneway else 2)
    return 0.5 * max(lanes, 1) * rules.get("lane.width", road_class=seed.road_class).low


def _input_heading(seed: RoadSeed, end: str) -> float:
    line = seed.line
    a, b = (0.0, min(HEADING_REACH_M, line.length)) if end == "start" else (max(0.0, line.length - HEADING_REACH_M), line.length)
    p, q = line.point(a)[0], line.point(b)[0]
    return math.atan2(q[1] - p[1], q[0] - p[0])


def _align_conf(road: Road, s: float) -> float:
    for el in road.alignment.elements:
        if el.s0 - 1e-6 <= s <= el.s0 + el.length + 1e-6:
            return el.prov.confidence
    return road.alignment.elements[-1].prov.confidence


def find_links(fits: list[RoadFit], js: list[Junction]) -> tuple[list[Link], list[Flag]]:
    """Links of the unit, ordered by id, and flags for junctions that could not be seamed."""
    roads = {f.road.id: f.road for f in fits}
    links: list[Link] = []
    flags: list[Flag] = []
    for j in js:
        if j.child not in roads or j.parent not in roads:
            continue
        link, flag = _link(roads[j.child], j.end, roads[j.parent], j.at_parent_end)
        if link is not None:
            links.append(link)
        if flag is not None:
            flags.append(flag)
    return sorted(links, key=lambda lk: lk.id), flags


def _link(child: Road, end: str, parent: Road, at_end: bool) -> tuple[Link | None, Flag | None]:
    cs = 0.0 if end == "start" else child.length
    x, y, _h, _k = child.alignment.evaluate(cs)
    ps, pu = parent.alignment.project(x, y)
    xs = parent.cross_section
    if not (-xs.width_right(ps) - SEAM_PAVEMENT_PAD_M <= pu <= xs.width_left(ps) + SEAM_PAVEMENT_PAD_M):
        return None, Flag(
            "FR-AF-3:no_seam",
            child.id,
            "FR-AF-3",
            f"{end} of {child.name} joins {parent.name} in the vectors but lies {pu:+.2f} m off its pavement at s={ps:.1f}",
        )
    prov = Provenance(Source.MEASURED, round(min(_align_conf(child, cs), _align_conf(parent, ps)), 6), flags=["FR-AF-3", "FR-IN-5:topology_from_vectors"])
    ps = float(ps)
    parent_end = "end" if ps > 0.5 * parent.length else "start"
    if end == "start":
        kind = "continue" if at_end else "diverge"
        lid = stable_id("lk", parent.id, child.id, "start")
        return Link(lid, parent.id, parent_end, child.id, "start", kind, from_s=ps, to_s=0.0, prov=prov), None
    kind = "continue" if at_end else "merge"
    lid = stable_id("lk", child.id, parent.id, "end")
    return Link(lid, child.id, "end", parent.id, parent_end, kind, from_s=float(child.length), to_s=ps, prov=prov), None


def drop_junction_crossings(crossings: list[Crossing], measures: list, links: list[Link]) -> tuple[list[Crossing], list, list[Flag]]:
    """Plan intersections of two linked roads with almost no height difference are their junction overlap (FR-CR-1).

    Where a ramp leaves its mainline the two reference lines run on one pavement
    and can intersect; that is the L6 junction, not a grade separation, so it gets
    no crossing (and no structure). Each one dropped is flagged.
    """
    pairs = {frozenset((lk.from_road, lk.to_road)) for lk in links}
    keep, flags = [], []
    for c in crossings:
        if frozenset((c.upper_road, c.lower_road)) in pairs and abs(c.dz_m) < JUNCTION_DZ_M:
            flags.append(Flag("FR-CR-1:junction_overlap", c.id, "FR-CR-1", f"linked roads meet in plan with dz {c.dz_m:.2f} m; part of their junction, not a crossing"))
            continue
        keep.append(c)
    ids = {c.id for c in keep}
    return keep, [m for m in measures if m.crossing_id in ids], flags
