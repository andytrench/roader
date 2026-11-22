"""Canonical R2 entities (in-memory)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


@dataclass
class SolidFace:
    """One triangle of a closed carriageway prism."""

    solid_id: str
    role: str  # "top" | "bottom" | "wall"
    triangle: Polygon


@dataclass
class Skip:
    """An OSM way that did not become a carriageway, with the reason."""

    reason: str
    highway: str
    osm_ids: list[int] = field(default_factory=list)
    xy: tuple[float, float] | None = None


@dataclass
class Carriageway:
    id: str
    centerline: LineString
    polygon: Polygon | MultiPolygon | None
    highway: str
    layer: str  # "terrain" | "deck" | "unknown"
    osm_ids: list[int] = field(default_factory=list)
    z_mean_m: float | None = None
    deck_id: str | None = None
    interchange_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    profile_xy: list[tuple[float, float]] = field(default_factory=list)
    profile_z: list[float] = field(default_factory=list)
    profile_ground_z: list[float] = field(default_factory=list)
    profile_deck: list[bool] = field(default_factory=list)
    faces: list[SolidFace] = field(default_factory=list)


@dataclass
class Deck:
    id: str
    polygon: Polygon | MultiPolygon
    z_p95_m: float
    z_max_m: float
    point_count: int
    source: str  # "las_class_17" | "elevated_cluster"
    clearance_below_m: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoidUnder:
    id: str
    polygon: Polygon | MultiPolygon
    deck_id: str
    z_floor_m: float | None
    z_ceiling_m: float | None
    clearance_m: float | None
    kind: str  # "terrain" | "road" | "water" | "unknown"
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sidewalk:
    id: str
    geometry: BaseGeometry
    carriageway_id: str | None
    source: str  # "osm" | "inferred_naip" | "inferred_buffer"
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Interchange:
    id: str
    carriageway_ids: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadScene:
    """One AOI worth of reconstructed objects."""

    crs: str
    carriageways: list[Carriageway] = field(default_factory=list)
    decks: list[Deck] = field(default_factory=list)
    voids: list[VoidUnder] = field(default_factory=list)
    sidewalks: list[Sidewalk] = field(default_factory=list)
    interchanges: list[Interchange] = field(default_factory=list)
    skips: list[Skip] = field(default_factory=list)
