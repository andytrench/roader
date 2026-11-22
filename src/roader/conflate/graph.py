"""Vector graph for one unit (FR-VC-1, spec section 1 scope).

Limited-access edges are the core. A non-core edge is kept when it crosses a core
edge in plan (a grade separation, since limited-access roads have no at-grade
crossings); it is then extended through connected edges of the same street so the
crossing road reaches both approaches. Connected edges of one carriageway are
chained into one road. Every road keeps its source edges and their tags, by
source (FR-IN-7).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import shapely

from roader.conflate.polyline import Polyline
from roader.io.vectors import VectorEdge, is_limited_access
from roader.model.schema import VectorRef
from roader.model.serialize import stable_id

JOIN_M = 1.0  # endpoint coincidence for chaining edges of one road
JOIN_HEADING_MAX = math.radians(45.0)
EXTENT_INSET_M = 10.0  # roads are clipped this far inside the lidar extent
MIN_ROAD_M = 60.0


@dataclass
class EdgeSpan:
    """The station range of one source edge on a road, with its tags (for FR-CR-2 logging)."""

    source: str
    feature_id: str
    s0: float
    s1: float
    tags: dict[str, str]


@dataclass
class RoadSeed:
    id: str
    name: str
    road_class: str
    kind: str  # mainline | ramp | connector | crossing_street
    oneway: bool
    line: Polyline  # input centerline in the direction of travel, clipped to the unit
    vector_refs: list[VectorRef]
    spans: list[EdgeSpan] = field(default_factory=list)

    def tags_at(self, s: float) -> dict[str, str]:
        for span in self.spans:
            if span.s0 - 1e-6 <= s <= span.s1 + 1e-6:
                return span.tags
        return self.spans[0].tags if self.spans else {}


def _oriented(edge: VectorEdge) -> np.ndarray:
    coords = np.asarray(edge.coords, dtype=np.float64)
    return coords[::-1] if edge.tags.get("oneway") == "-1" else coords


def _heading(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _street_key(edge: VectorEdge) -> tuple[str, str]:
    tags = edge.tags
    return (tags.get("name") or tags.get("ref") or "", tags.get("highway", ""))


def _chain(start: int, edges: list[VectorEdge], members: set[int], same) -> list[int]:
    """Edges reachable from start through coincident endpoints with heading continuity."""
    order = [start]
    used = {start}
    for direction in (1, -1):
        current = start
        while True:
            coords = _oriented(edges[current])
            end = coords[-1] if direction == 1 else coords[0]
            end_heading = _heading(coords[-2], coords[-1]) if direction == 1 else _heading(coords[1], coords[0])
            best = None
            for j in sorted(members - used):
                if not same(edges[j], edges[start]):
                    continue
                cj = _oriented(edges[j])
                oneway = edges[j].tags.get("oneway") in ("yes", "-1") or edges[j].tags.get("highway") == "motorway"
                options = [(cj, False)] if oneway else [(cj, False), (cj[::-1], True)]
                for c, _rev in options:
                    head = c[0] if direction == 1 else c[-1]
                    if math.hypot(*(head - end)) > JOIN_M:
                        continue
                    h = _heading(c[0], c[1]) if direction == 1 else _heading(c[-1], c[-2])
                    turn = abs(math.remainder(h - end_heading, 2 * math.pi))
                    if turn <= JOIN_HEADING_MAX and (best is None or turn < best[1]):
                        best = (j, turn)
            if best is None:
                break
            used.add(best[0])
            order = order + [best[0]] if direction == 1 else [best[0], *order]
            current = best[0]
    return order


def _merge_coords(ids: list[int], edges: list[VectorEdge]) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Concatenate chained edges; returns coords and per-edge (edge, first vertex, last vertex)."""
    out: list[np.ndarray] = []
    ranges = []
    for n, i in enumerate(ids):
        c = _oriented(edges[i])
        if out:
            prev_end = out[-1][-1]
            if math.hypot(*(c[0] - prev_end)) > math.hypot(*(c[-1] - prev_end)):
                c = c[::-1]
        start = sum(len(o) for o in out) - (1 if n else 0)
        out.append(c if not out else c[1:])
        ranges.append((i, max(start, 0), start + len(c) - 1))
    return np.vstack(out), ranges


def _bound(heading: float) -> str:
    deg = math.degrees(heading) % 360.0
    if 45.0 <= deg < 135.0:
        return "northbound"
    if 135.0 <= deg < 225.0:
        return "westbound"
    if 225.0 <= deg < 315.0:
        return "southbound"
    return "eastbound"


def _route_label(ref: str) -> str:
    """Interstate refs as written on signs and in crossing lists: "I 190" -> "I-190"."""
    first = ref.split(";")[0].strip()
    m = re.fullmatch(r"I\s*-?\s*(\d+[A-Z]?)", first)
    return f"I-{m.group(1)}" if m else first


def _kind(tags: dict[str, str]) -> str:
    highway = tags.get("highway", "")
    if highway in ("motorway", "trunk") and is_limited_access(tags):
        return "mainline"
    if highway.endswith("_link"):
        return "ramp"
    return "crossing_street"


def select_roads(edges: list[VectorEdge], extent: shapely.Geometry | None = None) -> list[RoadSeed]:
    """Roads of the unit model, ordered by id (FR-PL-3)."""
    edges = sorted(edges, key=lambda e: (e.source, e.feature_id))
    lines = [shapely.LineString(e.coords) for e in edges]
    core = {i for i, e in enumerate(edges) if is_limited_access(e.tags)}
    crossing = {i for i in range(len(edges)) if i not in core and any(lines[i].crosses(lines[c]) for c in core)}

    def same_core(a: VectorEdge, b: VectorEdge) -> bool:
        return (a.tags.get("ref"), a.tags.get("highway"), a.tags.get("oneway")) == (
            b.tags.get("ref"),
            b.tags.get("highway"),
            b.tags.get("oneway"),
        )

    def same_street(a: VectorEdge, b: VectorEdge) -> bool:
        return _street_key(a) == _street_key(b) and _street_key(a)[0] != ""

    groups: list[list[int]] = []
    assigned: set[int] = set()
    non_core = set(range(len(edges))) - core
    for i in sorted(core | crossing):
        if i in assigned:
            continue
        members, same = (core, same_core) if i in core else (non_core, same_street)
        chain = _chain(i, edges, members - assigned, same)
        assigned.update(chain)
        groups.append(chain)

    inner = extent.buffer(-EXTENT_INSET_M) if extent is not None else None
    seeds = []
    for chain in groups:
        coords, ranges = _merge_coords(chain, edges)
        line = shapely.LineString(coords)
        piece = line
        if inner is not None:
            clipped = line.intersection(inner)
            parts = [g for g in getattr(clipped, "geoms", [clipped]) if isinstance(g, shapely.LineString) and not g.is_empty]
            if not parts:
                continue
            piece = max(parts, key=lambda g: g.length)
        if piece.length < MIN_ROAD_M:
            continue
        pc = np.asarray(piece.coords)
        # Keep the direction of travel of the merged chain.
        if line.project(shapely.Point(pc[0])) > line.project(shapely.Point(pc[-1])):
            pc = pc[::-1]
        poly = Polyline(pc)
        first = edges[chain[0]]
        tags = first.tags
        oneway = tags.get("oneway") in ("yes", "-1") or tags.get("highway") == "motorway"
        name = tags.get("name") or tags.get("ref") or f"{tags.get('highway', 'road')} {first.feature_id}"
        if oneway and _kind(tags) == "mainline":
            h = _heading(pc[0], pc[-1])
            name = f"{_route_label(tags.get('ref') or name)} {_bound(h)}"
        spans = []
        offset = line.project(shapely.Point(pc[0]))
        for i, a, b in ranges:
            s0 = line.project(shapely.Point(coords[a])) - offset
            s1 = line.project(shapely.Point(coords[b])) - offset
            lo, hi = max(min(s0, s1), 0.0), min(max(s0, s1), poly.length)
            if hi > lo:
                spans.append(EdgeSpan(edges[i].source, edges[i].feature_id, lo, hi, dict(edges[i].tags)))
        refs = [VectorRef(edges[i].source, edges[i].feature_id, dict(edges[i].tags)) for i in chain]
        rid = stable_id("rd", *sorted(f"{edges[i].source}:{edges[i].feature_id}" for i in chain))
        seeds.append(
            RoadSeed(
                id=rid,
                name=name,
                road_class=tags.get("highway", "road"),
                kind=_kind(tags),
                oneway=oneway,
                line=poly,
                vector_refs=refs,
                spans=sorted(spans, key=lambda sp: sp.s0),
            )
        )
    return sorted(seeds, key=lambda r: r.id)


def data_extent(xy: np.ndarray, cell_m: float = 5.0) -> shapely.Geometry:
    """Plan outline of the point cloud: union of occupied cells with interior holes filled.

    Holes (water, the shadow under a deck) are not the edge of the data, so roads
    are not clipped there.
    """
    cells = np.unique(np.floor(xy / cell_m).astype(np.int64), axis=0)
    boxes = shapely.box(cells[:, 0] * cell_m, cells[:, 1] * cell_m, (cells[:, 0] + 1) * cell_m, (cells[:, 1] + 1) * cell_m)
    union = shapely.union_all(boxes)
    polys = [g for g in getattr(union, "geoms", [union]) if isinstance(g, shapely.Polygon)]
    return shapely.union_all([shapely.Polygon(p.exterior) for p in polys])
