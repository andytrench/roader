"""Seed road centerlines and sidewalk tags from OpenStreetMap via OSMnx."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import LineString


# Overpass mirrors were refusing or returning an empty planet during the Boston run.
# The main map call answers for a single 1 km tile, which is the fallback.
_OVERPASS_URLS = (
    "https://overpass.osm.ch/api",
    "https://overpass-api.de/api",
)
_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"
_MAP_DOCS: dict[tuple[float, float, float, float], bytes] = {}
_UA = {"Accept": "*/*", "User-Agent": "roader/0.1 (local road reconstruction)"}


def _overpass_xml(query: str) -> bytes:
    """POST a query ourselves. OSMnx pauses a minute when a mirror returns HTML."""
    last: Exception | None = None
    for index, base in enumerate(_OVERPASS_URLS):
        if index:
            time.sleep(5)
        url = base.rstrip("/") + "/interpreter"
        try:
            response = requests.post(
                url,
                data=query,
                timeout=15,
                headers=_UA,
            )
            response.raise_for_status()
            body = response.content
            if b"<osm" not in body[:800] or b"<way" not in body:
                raise RuntimeError("Overpass returned no road ways")
            return body
        except Exception as exc:  # noqa: BLE001 - the next endpoint may answer
            last = exc
    if last is None:
        raise RuntimeError("no Overpass endpoint")
    raise last


def _map_document(bbox_wgs84: tuple[float, float, float, float]) -> bytes:
    """Full OSM document for a small bbox. The map call allows up to 0.25 square degrees."""
    key = tuple(round(v, 5) for v in bbox_wgs84)
    cached = _MAP_DOCS.get(key)
    if cached is not None:
        return cached
    west, south, east, north = bbox_wgs84
    area = abs(east - west) * abs(north - south)
    if area <= 0 or area > 0.25:
        raise RuntimeError(f"bbox area {area:.4f} deg^2 is outside the map call limit")
    response = requests.get(
        _MAP_URL,
        params={"bbox": f"{west},{south},{east},{north}"},
        timeout=120,
        headers=_UA,
    )
    response.raise_for_status()
    body = response.content
    if b"<osm" not in body[:800] or b"<way" not in body:
        raise RuntimeError("OpenStreetMap map call returned no ways")
    _MAP_DOCS[key] = body
    return body


def _tags(way: ET.Element) -> dict[str, str]:
    return {
        tag.get("k", ""): tag.get("v", "")
        for tag in way.findall("tag")
        if tag.get("k")
    }


def _highway_kept(tags: dict[str, str]) -> bool:
    return tags.get("highway") in HIGHWAY_KEEP


def _sidewalk_kept(tags: dict[str, str]) -> bool:
    if tags.get("highway") in {"footway", "path", "pedestrian", "cycleway"}:
        return True
    if tags.get("footway") in {"sidewalk", "crossing"}:
        return True
    return "sidewalk" in tags


def _subset_xml(document: bytes, keep) -> bytes:
    """Ways that pass keep, plus the nodes those ways use."""
    root = ET.fromstring(document)
    nodes = {node.get("id"): node for node in root.findall("node")}
    ways: list[ET.Element] = []
    needed: set[str] = set()
    for way in root.findall("way"):
        tags = _tags(way)
        if not keep(tags):
            continue
        refs = [nd.get("ref") for nd in way.findall("nd")]
        if not refs or any(ref not in nodes for ref in refs):
            continue
        ways.append(way)
        needed.update(ref for ref in refs if ref)
    if not ways:
        raise RuntimeError("OpenStreetMap returned no road ways")
    out = ET.Element("osm", {"version": "0.6"})
    for ref in needed:
        out.append(nodes[ref])
    for way in ways:
        out.append(way)
    return ET.tostring(out, xml_declaration=True, encoding="UTF-8")


def _lines_from_document(document: bytes, keep) -> gpd.GeoDataFrame:
    root = ET.fromstring(document)
    coords: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        if node_id and node.get("lon") and node.get("lat"):
            coords[node_id] = (float(node.get("lon", "0")), float(node.get("lat", "0")))
    rows: list[dict[str, Any]] = []
    for way in root.findall("way"):
        tags = _tags(way)
        if not keep(tags):
            continue
        line = [coords[ref] for nd in way.findall("nd") if (ref := nd.get("ref")) in coords]
        if len(line) < 2:
            continue
        rows.append(
            {
                "highway": tags.get("highway"),
                "footway": tags.get("footway"),
                "name": tags.get("name"),
                "geometry": LineString(line),
            }
        )
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


HIGHWAY_KEEP = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
    "unclassified",
    "living_street",
    "service",
    "road",
}


def fetch_road_edges(
    bbox_wgs84: tuple[float, float, float, float],
    network_type: str = "all",
) -> gpd.GeoDataFrame:
    """Return projected-capable edges GeoDataFrame in WGS84.

    bbox: (west, south, east, north)
    """
    west, south, east, north = bbox_wgs84
    # network_type is kept for callers. The query is the drive network this pilot uses.
    del network_type
    query = (
        "[out:xml][timeout:25];"
        'way["highway"~"motorway|trunk|primary|secondary|tertiary|residential'
        '|unclassified|living_street|service|road"]'
        f"({south},{west},{north},{east});"
        "(._;>;);"
        "out body;"
    )
    try:
        xml = _overpass_xml(query)
    except Exception:
        xml = _subset_xml(_map_document(bbox_wgs84), _highway_kept)
    with NamedTemporaryFile(suffix=".osm", delete=False) as handle:
        handle.write(xml)
        path = Path(handle.name)
    try:
        graph = ox.graph_from_xml(path, simplify=True, retain_all=True)
    finally:
        path.unlink(missing_ok=True)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    edges = edges.reset_index(drop=False)
    if "highway" in edges.columns:
        def _keep(h: Any) -> bool:
            if isinstance(h, list):
                return any(x in HIGHWAY_KEEP for x in h)
            return h in HIGHWAY_KEEP

        edges = edges[edges["highway"].map(_keep)].copy()
    edges = edges.set_crs("EPSG:4326", allow_override=True)
    return edges


def fetch_sidewalk_geometries(
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """OSM footways / sidewalks / paths inside the bbox."""
    try:
        # Reuses the map document when the road fetch already downloaded it.
        return _lines_from_document(_map_document(bbox_wgs84), _sidewalk_kept)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
