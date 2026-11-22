"""Snap OSM ways onto a LiDAR corridor prism and group at-grade interchanges."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge

from roader.legacy.decks import profile_z_at, ribbons_from_clouds, voids_from_profiles
from roader.legacy.lidar_layers import LidarLayers, elevated_candidate_points
from roader.legacy.model import Carriageway, Deck, Interchange, Skip, SolidFace, VoidUnder
from roader.legacy.surface import (
    Cloud,
    NdviFn,
    class_half_width,
    dead_end_reason,
    tag_skip_reason,
    tag_str,
)

# Ends closer than this, with the same class and heading, are one road.
_JOIN_M = 8.0
_HEADING_MIN = 0.57
# A ramp end this far from the parent centerline is extended to the pavement edge.
_RAMP_REACH_M = 20.0
# A copy sits on another road when most of the smaller footprint overlaps at the same height.
_COPY_OVERLAP = 0.6
_COPY_Z_M = 1.5


@dataclass
class _Way:
    line: LineString
    highway: str
    osm_ids: list[int]
    attrs: dict[str, Any]
    row_index: int
    part_index: int


@dataclass
class AssignResult:
    carriageways: list[Carriageway] = field(default_factory=list)
    decks: list[Deck] = field(default_factory=list)
    voids: list[VoidUnder] = field(default_factory=list)
    skips: list[Skip] = field(default_factory=list)


def _as_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    try:
        merged = linemerge(geom)
        return _as_lines(merged)
    except Exception:
        return []


def _highway_str(val: Any) -> str:
    if isinstance(val, list):
        return str(val[0]) if val else "road"
    return str(val) if val is not None else "road"


def _osm_ids(row) -> list[int]:
    osm_id = row.get("osmid", row.get("id"))
    if isinstance(osm_id, list):
        return [int(x) for x in osm_id if x is not None and str(x).lstrip("-").isdigit()]
    if osm_id is None:
        return []
    text = str(osm_id)
    if text.lstrip("-").isdigit():
        return [int(text)]
    return []


def _is_bridge_tag(val: Any) -> bool:
    text = tag_str(val).lower()
    return text not in {"", "no", "false", "0"}


def _join_ways(ways: list[_Way]) -> list[_Way]:
    """Chain edges that continue one road. A bridge or layer change stays a separate piece."""
    if len(ways) < 2:
        return ways
    ends = [_end_frame(way.line) for way in ways]
    cells: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for index, (start, end, _start_h, _end_h) in enumerate(ends):
        if start is None:
            continue
        cells[_cell(start)].append((index, 0))
        cells[_cell(end)].append((index, 1))

    def nearby(point: np.ndarray) -> list[tuple[int, int]]:
        cx, cy = _cell(point)
        found: list[tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                found.extend(cells.get((cx + dx, cy + dy), []))
        return found

    proposals: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for index, way in enumerate(ways):
        if ends[index][0] is None:
            continue
        for leave in (0, 1):
            point, heading = _port(ends[index], leave)
            hits: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for other, enter in nearby(point):
                if other == index or (other, enter) in seen or not _same_piece(way, ways[other]):
                    continue
                seen.add((other, enter))
                if not _port_continues(ends[other], enter, point, heading):
                    continue
                if _overlaps_too_much(way.line, ways[other].line):
                    continue
                hits.append((other, enter))
            if len(hits) == 1:
                proposals.append(((index, leave), hits[0]))

    enter_ports = Counter(dst for _src, dst in proposals)
    links = {src: dst for src, dst in proposals if enter_ports[dst] == 1}
    entered = {dst[0] for dst in links.values()}
    visited: set[int] = set()
    joined: list[_Way] = []

    def walk(start: int, enter: int) -> None:
        chain: list[tuple[int, bool]] = []
        index: int | None = start
        while index is not None and index not in visited:
            visited.add(index)
            chain.append((index, enter == 1))
            nxt = links.get((index, 1 - enter))
            if nxt is None:
                break
            index, enter = nxt
        joined.append(_merge_chain(ways, chain))

    for start in [index for index in range(len(ways)) if index not in entered] + list(range(len(ways))):
        if start in visited or ends[start][0] is None:
            continue
        # Enter at the end only when the chain leaves through the original start.
        enter = 1 if (start, 0) in links and (start, 1) not in links else 0
        walk(start, enter)
    return joined


def _port(
    frame: tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray],
    leave: int,
) -> tuple[np.ndarray, np.ndarray]:
    start, end, start_h, end_h = frame
    if leave == 1:
        return end, end_h  # type: ignore[return-value]
    return start, -start_h  # type: ignore[return-value]


def _port_continues(
    frame: tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray],
    enter: int,
    point: np.ndarray,
    heading: np.ndarray,
) -> bool:
    start, end, start_h, end_h = frame
    if start is None or end is None:
        return False
    if enter == 0:
        return float(np.linalg.norm(point - start)) <= _JOIN_M and float(np.dot(heading, start_h)) >= _HEADING_MIN
    return float(np.linalg.norm(point - end)) <= _JOIN_M and float(np.dot(heading, -end_h)) >= _HEADING_MIN


def _end_frame(line: LineString) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    coords = [coord[:2] for coord in line.coords]
    if len(coords) < 2:
        return None, None, np.zeros(2), np.zeros(2)
    start = np.asarray(coords[0], dtype=float)
    end = np.asarray(coords[-1], dtype=float)
    return start, end, _unit(np.asarray(coords[1], dtype=float) - start), _unit(end - np.asarray(coords[-2], dtype=float))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros(2)
    return vector / norm


def _cell(point: np.ndarray) -> tuple[int, int]:
    return int(np.floor(point[0] / _JOIN_M)), int(np.floor(point[1] / _JOIN_M))


def _same_piece(left: _Way, right: _Way) -> bool:
    if left.highway != right.highway:
        return False
    if _is_bridge_tag(left.attrs.get("bridge")) != _is_bridge_tag(right.attrs.get("bridge")):
        return False
    return tag_str(left.attrs.get("layer_tag")) == tag_str(right.attrs.get("layer_tag"))


def _overlaps_too_much(left: LineString, right: LineString) -> bool:
    shorter = min(float(left.length), float(right.length))
    if shorter < 1.0:
        return False
    shared = left.intersection(right.buffer(3.0))
    return float(getattr(shared, "length", 0.0)) > 0.5 * shorter


def _merge_chain(ways: list[_Way], chain: list[tuple[int, bool]]) -> _Way:
    coords: list[tuple[float, float]] = []
    osm_ids: list[int] = []
    for index, flip in chain:
        pts = [tuple(coord[:2]) for coord in ways[index].line.coords]
        if flip:
            pts.reverse()
        if coords and pts and _close(coords[-1], pts[0]):
            pts = pts[1:]
        coords.extend(pts)
        for osm_id in ways[index].osm_ids:
            if osm_id not in osm_ids:
                osm_ids.append(osm_id)
    first = ways[chain[0][0]]
    line = LineString(coords) if len(coords) >= 2 else first.line
    return _Way(
        line=line,
        highway=first.highway,
        osm_ids=osm_ids,
        attrs=dict(first.attrs),
        row_index=min(ways[index].row_index for index, _flip in chain),
        part_index=0,
    )


def _close(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 < 1.0


def _extend_ramps(ways: list[_Way]) -> None:
    """Carry a link end out to the pavement edge of the road it joins."""
    parents = [way for way in ways if not way.highway.endswith("_link")]
    for way in ways:
        if not way.highway.endswith("_link"):
            continue
        parent_class = way.highway.removesuffix("_link")
        targets = [parent.line for parent in parents if parent.highway == parent_class]
        if not targets:
            continue
        half = class_half_width(parent_class)
        extended = _extend_line(way.line, targets, half)
        if extended is not None:
            way.line = extended


def _extend_line(line: LineString, parents: list[LineString], parent_half: float) -> LineString | None:
    coords = [tuple(coord[:2]) for coord in line.coords]
    if len(coords) < 2:
        return None
    changed = False
    for at_start in (True, False):
        end = np.asarray(coords[0] if at_start else coords[-1], dtype=float)
        prev = np.asarray(coords[1] if at_start else coords[-2], dtype=float)
        travel = _unit(prev - end) if at_start else _unit(end - prev)
        nearest, dist = _nearest_parent(end, parents)
        if nearest is None or dist <= parent_half + 0.5 or dist > _RAMP_REACH_M:
            continue
        toward = nearest - end
        # The parent is ahead of the last point, or behind the first point.
        aligned = float(np.dot(travel, toward))
        if at_start and aligned >= 0.0:
            continue
        if not at_start and aligned <= 0.0:
            continue
        new = nearest + (end - nearest) * (parent_half / dist)
        point = (float(new[0]), float(new[1]))
        if at_start:
            coords.insert(0, point)
        else:
            coords.append(point)
        changed = True
    if not changed:
        return None
    return LineString(coords)


def _nearest_parent(end: np.ndarray, parents: list[LineString]) -> tuple[np.ndarray | None, float]:
    point = Point(float(end[0]), float(end[1]))
    best: np.ndarray | None = None
    best_dist = float("inf")
    for parent in parents:
        dist = float(parent.distance(point))
        if dist < best_dist:
            nearest = parent.interpolate(parent.project(point))
            best = np.array([nearest.x, nearest.y], dtype=float)
            best_dist = dist
    return best, best_dist


_CLASS_RANK = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "residential": 5,
    "unclassified": 6,
    "living_street": 7,
    "service": 8,
    "road": 9,
}


def _class_rank(highway: str) -> float:
    base = highway.removesuffix("_link")
    rank = float(_CLASS_RANK.get(base, 9))
    if highway.endswith("_link"):
        rank += 0.5
    return rank


def _drop_copies(
    carriageways: list[Carriageway],
    decks: list[Deck],
    skips: list[Skip],
) -> tuple[list[Carriageway], list[Deck]]:
    """Drop a way that already lies on another road at the same height. The higher class stays."""
    dropped: set[str] = set()
    for left_i, left in enumerate(carriageways):
        if left.polygon is None:
            continue
        for right in carriageways[left_i + 1 :]:
            if right.polygon is None or left.id in dropped or right.id in dropped:
                continue
            if left.z_mean_m is None or right.z_mean_m is None:
                continue
            if abs(float(left.z_mean_m) - float(right.z_mean_m)) > _COPY_Z_M:
                continue
            overlap = left.polygon.intersection(right.polygon)
            smaller = min(float(left.polygon.area), float(right.polygon.area))
            if smaller < 1.0 or float(overlap.area) / smaller < _COPY_OVERLAP:
                continue
            loser = _loser(left, right)
            dropped.add(loser.id)
            skips.append(
                Skip(
                    reason="duplicate",
                    highway=loser.highway,
                    osm_ids=list(loser.osm_ids),
                    xy=_line_xy(loser.centerline),
                )
            )
    if not dropped:
        return carriageways, decks
    kept = [item for item in carriageways if item.id not in dropped]
    kept_decks = [
        deck for deck in decks if str(deck.attrs.get("carriageway_id", "")) not in dropped
    ]
    return kept, kept_decks


def _loser(left: Carriageway, right: Carriageway) -> Carriageway:
    left_rank = _class_rank(left.highway)
    right_rank = _class_rank(right.highway)
    if left_rank != right_rank:
        return left if left_rank > right_rank else right
    left_len = float(left.centerline.length) if left.centerline is not None else 0.0
    right_len = float(right.centerline.length) if right.centerline is not None else 0.0
    return left if left_len < right_len else right


def _line_xy(line: LineString) -> tuple[float, float] | None:
    if line is None or line.is_empty:
        return None
    coord = line.coords[0]
    return (float(coord[0]), float(coord[1]))


def _face_polygon(coords) -> Polygon:
    a, b, c = coords
    return Polygon([a, b, c, a])


def assign_carriageways(
    edges: gpd.GeoDataFrame,
    layers: LidarLayers,
    *,
    target_crs: str,
    height_above_ground_m: float = 2.5,
    pavement_thickness_m: float = 0.2,
    deck_thickness_m: float = 0.8,
    ndvi: NdviFn | None = None,
) -> AssignResult:
    """Project edges, fit a prism from the corridor, and drop non-roads."""
    if edges is None or edges.empty:
        return AssignResult()
    gdf = edges.to_crs(target_crs)
    ground = Cloud.from_arrays(layers.ground_xy, layers.ground_z, layers.ground_intensity)
    if len(layers.bridge_xy):
        bridge = Cloud.from_arrays(layers.bridge_xy, layers.bridge_z, layers.bridge_intensity)
        deck_source = "las_class_17"
    else:
        elevated_xy, elevated_z = elevated_candidate_points(
            layers,
            height_above_ground_m,
        )
        bridge = Cloud.from_arrays(elevated_xy, elevated_z, None)
        deck_source = "elevated_cluster"

    prepared: list[_Way] = []
    skips: list[Skip] = []
    for row_index, row in gdf.iterrows():
        highway = _highway_str(row.get("highway"))
        attrs = {
            "name": row.get("name"),
            "bridge": row.get("bridge"),
            "layer_tag": row.get("layer"),
            "tunnel": row.get("tunnel"),
            "service": tag_str(row.get("service")),
            "sidewalk": tag_str(row.get("sidewalk")),
        }
        osm_ids = _osm_ids(row)
        for part_index, line in enumerate(_as_lines(row.geometry)):
            prepared.append(
                _Way(
                    line=line,
                    highway=highway,
                    osm_ids=osm_ids,
                    attrs=attrs,
                    row_index=int(row_index) if str(row_index).isdigit() else 0,
                    part_index=part_index,
                )
            )

    survivors: list[_Way] = []
    for way in prepared:
        reason = tag_skip_reason(way.line, way.highway, way.attrs)
        if reason:
            skips.append(
                Skip(
                    reason=reason,
                    highway=way.highway,
                    osm_ids=way.osm_ids,
                    xy=_line_xy(way.line),
                )
            )
        else:
            survivors.append(way)

    survivors = _join_ways(survivors)
    _extend_ramps(survivors)

    kept: list[_Way] = []
    for way in survivors:
        others = [other.line for other in survivors if other is not way]
        reason = dead_end_reason(way.line, way.highway, others)
        if reason:
            skips.append(
                Skip(
                    reason=reason,
                    highway=way.highway,
                    osm_ids=way.osm_ids,
                    xy=_line_xy(way.line),
                )
            )
        else:
            kept.append(way)

    carriageways: list[Carriageway] = []
    decks: list[Deck] = []
    for way in kept:
        fits = ribbons_from_clouds(
            way.line,
            ground,
            bridge,
            highway=way.highway,
            pavement_thickness_m=pavement_thickness_m,
            deck_thickness_m=deck_thickness_m,
            deck_height_m=height_above_ground_m,
            bridge_tag=_is_bridge_tag(way.attrs.get("bridge")),
            ndvi=ndvi,
        )
        if len(fits) == 1 and fits[0].reason:
            skips.append(
                Skip(
                    reason=str(fits[0].reason),
                    highway=way.highway,
                    osm_ids=way.osm_ids,
                    xy=_line_xy(way.line),
                )
            )
            continue
        for run_index, fit in enumerate(fits):
            if fit.reason or fit.polygon is None or fit.centerline is None:
                skips.append(
                    Skip(
                        reason=fit.reason or "lidar_gap",
                        highway=way.highway,
                        osm_ids=way.osm_ids,
                        xy=_line_xy(way.line),
                    )
                )
                continue
            cid = f"cw_{way.row_index}_{way.part_index}_{run_index}"
            faces = [
                SolidFace(solid_id=cid, role=face.role, triangle=_face_polygon(face.coords))
                for face in fit.raw_faces
            ]
            deck_ids: list[str] = []
            largest_id: str | None = None
            largest_area = -1.0
            for part_index, part in enumerate(fit.deck_parts):
                did = f"deck_{cid}_{part_index}"
                deck_ids.append(did)
                area = float(part.polygon.area)
                if area > largest_area:
                    largest_area = area
                    largest_id = did
                decks.append(
                    Deck(
                        id=did,
                        polygon=part.polygon,
                        z_p95_m=part.z_p95,
                        z_max_m=part.z_max,
                        point_count=part.point_count,
                        source=deck_source,
                        attrs={"carriageway_id": cid},
                    )
                )
            attrs = dict(way.attrs)
            attrs["half_width_m"] = fit.half_width_m
            attrs["section"] = fit.section
            attrs["deck_ids"] = deck_ids
            carriageways.append(
                Carriageway(
                    id=cid,
                    centerline=fit.centerline,
                    polygon=fit.polygon,
                    highway=way.highway,
                    layer="deck" if deck_ids else "terrain",
                    osm_ids=list(way.osm_ids),
                    z_mean_m=fit.z_mean,
                    deck_id=largest_id,
                    attrs=attrs,
                    profile_xy=list(fit.profile_xy),
                    profile_z=list(fit.profile_z),
                    profile_ground_z=list(fit.profile_ground_z),
                    profile_deck=list(fit.profile_deck),
                    faces=faces,
                )
            )

    carriageways, decks = _drop_copies(carriageways, decks, skips)
    voids = voids_from_profiles(
        carriageways,
        decks,
        deck_thickness_m=deck_thickness_m,
        z_threshold=height_above_ground_m,
    )
    return AssignResult(
        carriageways=carriageways,
        decks=decks,
        voids=voids,
        skips=skips,
    )


def group_interchanges(
    carriageways: list[Carriageway],
    *,
    z_threshold: float = 2.5,
) -> list[Interchange]:
    """Group link carriageways that meet at grade. A stacked pair stays split."""
    links = [c for c in carriageways if "link" in c.highway]
    if not links:
        return []
    parent = {c.id: c.id for c in links}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in links:
        for right in links:
            if left.id >= right.id:
                continue
            zone = left.centerline.buffer(15.0).intersection(right.centerline.buffer(15.0))
            if zone.is_empty:
                continue
            point = zone.centroid
            z_left = profile_z_at(left, point.x, point.y)
            z_right = profile_z_at(right, point.x, point.y)
            if (
                z_left is not None
                and z_right is not None
                and abs(z_left - z_right) > z_threshold
            ):
                continue
            union(left.id, right.id)

    groups: dict[str, list[str]] = {}
    for carriageway in links:
        groups.setdefault(find(carriageway.id), []).append(carriageway.id)

    interchanges: list[Interchange] = []
    for gi, ids in enumerate(groups.values()):
        iid = f"ix_{gi}"
        interchanges.append(Interchange(id=iid, carriageway_ids=ids))
        for carriageway in carriageways:
            if carriageway.id in ids:
                carriageway.interchange_id = iid
    return interchanges
