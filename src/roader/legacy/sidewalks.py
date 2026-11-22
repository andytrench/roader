"""Sidewalks from OSM tags and mapped footways.

The pilot does not invent a sidewalk for a way that has no OSM evidence.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.errors import GEOSException

from roader.legacy.model import Carriageway, Sidewalk
from roader.legacy.surface import class_half_width, tag_str

URBAN = {
    "residential",
    "living_street",
    "primary",
    "secondary",
    "tertiary",
}

_SIDEWALK_TAGS = {"left", "right", "both", "yes"}


def sidewalks_from_osm(
    osm_sidewalks: gpd.GeoDataFrame,
    target_crs: str,
) -> list[Sidewalk]:
    if osm_sidewalks is None or osm_sidewalks.empty:
        return []
    gdf = osm_sidewalks.to_crs(target_crs)
    out: list[Sidewalk] = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        out.append(
            Sidewalk(
                id=f"sw_osm_{i}",
                geometry=geom,
                carriageway_id=None,
                source="osm",
                attrs={
                    "highway": row.get("highway"),
                    "footway": row.get("footway"),
                    "name": row.get("name"),
                },
            )
        )
    return out


def sidewalks_from_tags(
    carriageways: list[Carriageway],
    *,
    enabled: bool,
    width_m: float = 1.5,
) -> list[Sidewalk]:
    """Strips beside ways that carry OSM sidewalk=left|right|both|yes."""
    if not enabled:
        return []
    out: list[Sidewalk] = []
    for carriageway in carriageways:
        tag = tag_str(carriageway.attrs.get("sidewalk")).lower()
        if tag not in _SIDEWALK_TAGS:
            continue
        half = carriageway.attrs.get("half_width_m")
        if not isinstance(half, (int, float)):
            half = class_half_width(carriageway.highway)
        geom = _sidewalk_geom(carriageway.centerline, float(half), width_m, tag)
        if geom is None or geom.is_empty:
            continue
        out.append(
            Sidewalk(
                id=f"sw_tag_{carriageway.id}",
                geometry=geom,
                carriageway_id=carriageway.id,
                source="osm",
                attrs={"sidewalk": tag, "from": "sidewalk_tag"},
            )
        )
    return out


def _sidewalk_geom(line, half: float, width: float, tag: str):
    if tag in {"both", "yes"}:
        outer = line.buffer(half + width, cap_style=2)
        inner = line.buffer(half, cap_style=2)
        geom = outer.difference(inner)
        return None if geom.is_empty else geom
    side = "left" if tag == "left" else "right"
    try:
        offset = line.parallel_offset(half + width * 0.5, side, join_style=2)
    except (ValueError, GEOSException):
        return None
    if offset is None or offset.is_empty:
        return None
    geom = offset.buffer(width * 0.5, cap_style=2)
    return None if geom.is_empty else geom


def infer_sidewalk_buffers(
    carriageways: list[Carriageway],
    *,
    enabled: bool,
    offset_m: float = 1.5,
    width_m: float = 1.5,
) -> list[Sidewalk]:
    """Unused by the pilot. A buffer without an OSM tag invents a sidewalk.

    Kept so older callers can still ask for a low-confidence ring. Motorway
    and unclassified ways are not in ``URBAN``.
    """
    if not enabled:
        return []
    out: list[Sidewalk] = []
    for c in carriageways:
        if c.highway not in URBAN:
            continue
        if c.layer == "deck":
            continue
        # Ring between carriageway buffer and larger buffer
        outer = c.centerline.buffer(offset_m + width_m, cap_style=2)
        inner = c.centerline.buffer(offset_m, cap_style=2)
        ring = outer.difference(inner)
        if ring.is_empty:
            continue
        out.append(
            Sidewalk(
                id=f"sw_inf_{c.id}",
                geometry=ring,
                carriageway_id=c.id,
                source="inferred_buffer",
                attrs={"confidence": "low"},
            )
        )
    return out
