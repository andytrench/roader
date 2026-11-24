"""Vector input contract (stage 4 input). Owned by lane `data`."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely

from roader.io.frame import FrameTransformer
from roader.model.schema import Frame


@dataclass
class VectorEdge:
    """One road centerline from one source, in the unit frame.

    Attributes stay keyed by source (FR-IN-7, FR-VC-1). ``coords`` is (N, 2).
    """

    source: str  # "osm" | "tiger" | ...
    feature_id: str
    coords: np.ndarray
    tags: dict[str, str] = field(default_factory=dict)


def read_osm(path: str | Path, frame: Frame) -> list[VectorEdge]:
    """Highway ways from an Overpass JSON extract (``out body`` with nodes, or ``out geom``).

    Nodes missing from the extract are skipped; ways left with fewer than two
    nodes are dropped. Edges are ordered by way id (FR-PL-3).
    """
    doc = json.loads(Path(path).read_text())
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in doc.get("elements", []) if e.get("type") == "node"}
    to_unit = FrameTransformer(frame)
    edges = []
    ways = sorted((e for e in doc.get("elements", []) if e.get("type") == "way"), key=lambda e: e["id"])
    for way in ways:
        tags = {str(k): str(v) for k, v in (way.get("tags") or {}).items()}
        if "highway" not in tags:
            continue
        if "geometry" in way:
            lonlat = [(g["lon"], g["lat"]) for g in way["geometry"] if g]
        else:
            lonlat = [nodes[n] for n in way.get("nodes", []) if n in nodes]
        if len(lonlat) < 2:
            continue
        lon, lat = np.asarray(lonlat, dtype=np.float64).T
        x, y = to_unit.lonlat_to_unit(lon, lat)
        edges.append(VectorEdge("osm", str(way["id"]), np.column_stack([x, y]), tags))
    return edges


READERS: dict[str, Callable[[str | Path, Frame], list[VectorEdge]]] = {"osm": read_osm}


def read_vectors(sources_config: Mapping[str, Mapping], frame: Frame) -> dict[str, list[VectorEdge]]:
    """Read every enabled vector source (FR-IN-5, FR-IN-7).

    ``sources_config`` maps source name to a mapping with ``enabled`` and
    ``path`` (a UnitConfig's ``vectors`` works). Disabled sources are absent
    from the result, so nothing downstream can read their attributes.
    """
    out: dict[str, list[VectorEdge]] = {}
    for name in sorted(sources_config):
        cfg = sources_config[name]
        enabled = cfg["enabled"] if isinstance(cfg, Mapping) else cfg.enabled
        if not enabled:
            continue
        if name not in READERS:
            raise ValueError(f"no reader for vector source {name!r}")
        path = cfg["path"] if isinstance(cfg, Mapping) else cfg.path
        out[name] = READERS[name](path, frame)
    return out


def is_limited_access(tags: Mapping[str, str]) -> bool:
    """Spec section 1 core classes: motorways, their links, and trunk tagged motorroad/expressway."""
    highway = tags.get("highway", "")
    if highway in ("motorway", "motorway_link", "trunk_link"):
        return True
    return highway == "trunk" and (tags.get("motorroad") == "yes" or tags.get("expressway") == "yes")


PATH_CLASSES = frozenset({"footway", "cycleway", "path", "pedestrian", "steps", "bridleway", "corridor"})


def is_path(tags: Mapping[str, str]) -> bool:
    """Footways, cycleways and the other non-motorised ``highway`` classes (D-160)."""
    return tags.get("highway", "") in PATH_CLASSES


def path_in_scope(tags: Mapping[str, str]) -> bool:
    """A path on its own bridge: ``bridge`` other than ``no``, not a sidewalk or crossing (D-160)."""
    bridged = tags.get("bridge", "no") not in ("", "no")
    return bridged and tags.get("footway") not in ("sidewalk", "crossing")


def in_scope(edges: list[VectorEdge], buffer_m: float = 150.0) -> list[VectorEdge]:
    """Limited-access edges plus every edge crossing or within ``buffer_m`` of one (spec section 1).

    Paths count only on their own bridges (D-160).
    """
    lines = [shapely.LineString(e.coords) for e in edges]
    core = [i for i, e in enumerate(edges) if is_limited_access(e.tags)]
    if not core:
        return []
    tree = shapely.STRtree([lines[i] for i in core])
    keep = set(core)
    for i, line in enumerate(lines):
        if i in keep or (is_path(edges[i].tags) and not path_in_scope(edges[i].tags)):
            continue
        if tree.query(line, predicate="dwithin", distance=buffer_m).size:
            keep.add(i)
    return [e for i, e in enumerate(edges) if i in keep]
