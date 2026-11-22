"""Real milestone 2 stack fixture, when fetched (scripts/fetch_fixtures.py): FR-IN-1, FR-PL-3, AC-H2.

The AOI must hold the I-635 / US 75 High Five: both gores of every general-purpose
direct connector between the two mainlines, every elevated level, and every plan
crossing among in-scope edges inside it with a margin to the edge.
"""

import heapq
import json
import math
from pathlib import Path

import numpy as np
import pytest
import shapely
import yaml

from roader.io import (
    FrameTransformer,
    in_scope,
    load_unit_config,
    read_points,
    read_vectors,
    verify_inputs,
)
from roader.io.vectors import is_limited_access

ROOT = Path(__file__).resolve().parents[2]
SITE_ID = "dallas_high_five_i635_us75"
CONFIG = ROOT / "configs" / "units" / f"{SITE_ID}.toml"
cfg = load_unit_config(CONFIG)
SITE = next(s for s in yaml.safe_load((ROOT / "tests" / "sites.yaml").read_text())["sites"] if s["id"] == SITE_ID)
CROSSINGS = yaml.safe_load((ROOT / SITE["crossings"]).read_text())

pytestmark = pytest.mark.skipif(
    not (cfg.lidar[0].path.exists() and cfg.vectors["osm"].path.exists()),
    reason=f"fixture not fetched; run scripts/fetch_fixtures.py --site {SITE_ID}",
)


def _aoi_polygon(frame) -> shapely.Polygon:
    w, s, e, n = cfg.aoi_lonlat
    x, y = FrameTransformer(frame).lonlat_to_unit([w, e, e, w], [s, s, n, n])
    return shapely.Polygon(np.column_stack([x, y]))


def _unit_points(frame, lonlat) -> np.ndarray:
    lon, lat = np.asarray(lonlat, dtype=np.float64).reshape(-1, 2).T
    x, y = FrameTransformer(frame).lonlat_to_unit(lon, lat)
    return shapely.points(np.column_stack([x, y]))


@pytest.fixture(scope="module")
def loaded():
    frame = cfg.frame()
    src = cfg.lidar[0]
    pts, rec = read_points(src.path, cfg.aoi_lonlat, frame, name=src.name, year=src.year, quality_level=src.quality_level)
    return frame, pts, rec


@pytest.fixture(scope="module")
def edges():
    frame = cfg.frame()
    return frame, read_vectors(cfg.vectors, frame)["osm"]


def test_fr_pl_3_fixture_matches_pins():
    verify_inputs(cfg)


def test_fr_in_1_fixture_density_and_classes(loaded):
    frame, pts, rec = loaded
    density = len(pts) / _aoi_polygon(frame).area
    assert density >= 2.0  # QL2
    assert density == pytest.approx(SITE["lidar"]["density_pts_m2"], abs=0.1)
    assert rec.point_count == len(pts)
    assert rec.horizontal_crs == "EPSG:3857" and rec.vertical_crs.startswith("NAVD88")
    classes, counts = np.unique(pts.classification, return_counts=True)
    seen = dict(zip(classes.tolist(), counts.tolist(), strict=True))
    assert set(seen) == set(SITE["lidar"]["classes_seen"]) - {7, 18}
    assert seen.get(17, 0) == SITE["lidar"]["class_17_points"] > 0


def _connector_gores(doc: dict) -> set[tuple[int, int]]:
    """(diverge node, merge node) of the shortest motorway_link path from each I-635 node to US 75 and back."""
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in doc["elements"] if e["type"] == "node"}
    ways = [e for e in doc["elements"] if e["type"] == "way"]

    def mainline(match):
        return {n for w in ways if w["tags"].get("highway") == "motorway" and match(w["tags"].get("ref", "")) for n in w["nodes"]}

    i635 = mainline(lambda r: r == "I 635")
    us75 = mainline(lambda r: "US 75" in r.split(";"))
    adj: dict[int, list[int]] = {}
    for w in ways:
        if w["tags"].get("highway") == "motorway_link":
            for a, b in zip(w["nodes"], w["nodes"][1:], strict=False):
                adj.setdefault(a, []).append(b)

    def length(a, b):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        return math.hypot((x2 - x1) * math.cos(math.radians(y1)), y2 - y1) * 111_320.0

    gores = set()
    for src, dst in ((i635, us75), (us75, i635)):
        for start in sorted(src & set(adj)):
            dist, queue = {start: 0.0}, [(0.0, start)]
            while queue:
                d, u = heapq.heappop(queue)
                if d > dist[u] or d > 4000.0:
                    continue
                if u in dst and u != start:
                    gores.add((start, u))
                    continue
                for v in adj.get(u, []):
                    if v in nodes and d + length(u, v) < dist.get(v, math.inf):
                        dist[v] = d + length(u, v)
                        heapq.heappush(queue, (dist[v], v))
    return {g: (nodes[g[0]], nodes[g[1]]) for g in gores}


def test_aoi_covers_every_direct_connector_and_level(edges):
    frame, _ = edges
    doc = json.loads(cfg.vectors["osm"].path.read_text())
    gores = _connector_gores(doc)
    assert len(gores) == 8  # four movements each way between the two mainlines
    aoi = _aoi_polygon(frame)
    ends = _unit_points(frame, [p for pair in gores.values() for p in pair])
    assert shapely.contains(aoi, ends).all()
    assert shapely.distance(aoi.exterior, ends).min() >= 90.0
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in doc["elements"] if e["type"] == "node"}
    upper = [nodes[n] for w in doc["elements"] if w["type"] == "way" and w["tags"].get("layer", "0") in ("3", "4", "5") for n in w["nodes"]]
    assert len(upper) > 0
    # The layer-5 connector leaves the US 75 southbound collector about 33 m north of the AOI.
    assert shapely.contains(aoi.buffer(50.0), _unit_points(frame, upper)).all()


def _plan_crossings(scoped, aoi) -> set[tuple[str, str]]:
    """Way pairs crossing inside the AOI away from a shared vertex, one of them limited-access."""
    lines = [shapely.LineString(e.coords) for e in scoped]
    out = set()
    for i, a in enumerate(scoped):
        for j in range(i + 1, len(scoped)):
            b = scoped[j]
            if not (is_limited_access(a.tags) or is_limited_access(b.tags)) or not lines[i].crosses(lines[j]):
                continue
            shared = shapely.MultiPoint(a.coords).intersection(shapely.MultiPoint(b.coords))
            hits = shapely.get_parts(lines[i].intersection(lines[j]))
            if any(h.geom_type == "Point" and aoi.contains(h) and (shared.is_empty or h.distance(shared) > 1.0) for h in hits):
                out.add(tuple(sorted((a.feature_id, b.feature_id))))
    return out


def test_ac_h2_crossing_list_covers_every_plan_crossing(edges):
    frame, all_edges = edges
    aoi = _aoi_polygon(frame)
    scoped = [e for e in in_scope(all_edges) if shapely.LineString(e.coords).intersects(aoi)]
    listed = {tuple(sorted((str(c["upper"]["osm_way"]), str(c["lower"]["osm_way"])))) for c in CROSSINGS["crossings"]}
    unresolved = {tuple(sorted((str(c["a"]["osm_way"]), str(c["b"]["osm_way"])))) for c in CROSSINGS["unresolved"]}
    assert not listed & unresolved
    assert _plan_crossings(scoped, aoi) == listed | unresolved
    for c in CROSSINGS["crossings"] + CROSSINGS["unresolved"]:
        assert aoi.exterior.distance(_unit_points(frame, c["lonlat"])[0]) >= 100.0
    assert all(c["reason"] for c in CROSSINGS["unresolved"])
