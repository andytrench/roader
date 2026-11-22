"""Real milestone 2 fixture, when fetched (scripts/fetch_fixtures.py): FR-IN-1, FR-PL-3, AC-H2.

The AOI must hold the whole I-93 / I-95 cloverleaf: every ramp joining the two
mainlines, every plan crossing among in-scope edges, and a margin around them.
"""

import json
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
SITE_ID = "woburn_i93_i95"
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
    assert density >= 8.0  # QL1
    assert density == pytest.approx(SITE["lidar"]["density_pts_m2"], abs=0.1)
    assert rec.point_count == len(pts)
    assert rec.horizontal_crs == "EPSG:3857" and rec.vertical_crs.startswith("NAVD88")
    classes, counts = np.unique(pts.classification, return_counts=True)
    seen = dict(zip(classes.tolist(), counts.tolist(), strict=True))
    assert set(seen) == set(SITE["lidar"]["classes_seen"]) - {7, 18}
    assert seen.get(17, 0) == SITE["lidar"]["class_17_points"] > 0


def _interchange_ramps(osm_path: Path) -> list[list[tuple[float, float]]]:
    """Node lon/lat of each connected group of motorway_link ways touching both I-93 and I-95."""
    doc = json.loads(osm_path.read_text())
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in doc["elements"] if e["type"] == "node"}
    ways = [e for e in doc["elements"] if e["type"] == "way"]

    def mainline(ref):
        return {n for w in ways if w["tags"].get("highway") == "motorway" and ref in w["tags"].get("ref", "").split(";") for n in w["nodes"]}

    i93, i95 = mainline("I 93"), mainline("I 95")
    groups: list[set[int]] = []
    for w in ways:
        if w["tags"].get("highway") != "motorway_link":
            continue
        merged = set(w["nodes"])
        for g in [g for g in groups if g & merged]:
            groups.remove(g)
            merged |= g
        groups.append(merged)
    return [[nodes[n] for n in g if n in nodes] for g in groups if g & i93 and g & i95]


def test_aoi_covers_every_cloverleaf_ramp(edges):
    frame, _ = edges
    ramps = _interchange_ramps(cfg.vectors["osm"].path)
    assert len(ramps) == 8  # four loops, four outer ramps
    boundary = _aoi_polygon(frame).exterior
    t = FrameTransformer(frame)
    for lonlat in ramps:
        lon, lat = np.asarray(lonlat).T
        x, y = t.lonlat_to_unit(lon, lat)
        pts = shapely.points(np.column_stack([x, y]))
        assert shapely.contains(_aoi_polygon(frame), pts).all()
        assert shapely.distance(boundary, pts).min() >= 90.0


def _plan_crossings(scoped) -> set[tuple[str, str]]:
    """Way pairs whose centerlines cross away from a shared vertex, one of them limited-access."""
    lines = [shapely.LineString(e.coords) for e in scoped]
    out = set()
    for i, a in enumerate(scoped):
        for j in range(i + 1, len(scoped)):
            b = scoped[j]
            if not (is_limited_access(a.tags) or is_limited_access(b.tags)) or not lines[i].crosses(lines[j]):
                continue
            shared = shapely.MultiPoint(a.coords).intersection(shapely.MultiPoint(b.coords))
            hits = shapely.get_parts(lines[i].intersection(lines[j]))
            if any(h.geom_type == "Point" and h.distance(shared) > 1.0 for h in hits) or shared.is_empty:
                out.add(tuple(sorted((a.feature_id, b.feature_id))))
    return out


def test_ac_h2_crossing_list_covers_every_plan_crossing(edges):
    frame, all_edges = edges
    aoi = _aoi_polygon(frame)
    scoped = [e for e in in_scope(all_edges) if shapely.LineString(e.coords).intersects(aoi)]
    listed = {tuple(sorted((str(c["upper"]["osm_way"]), str(c["lower"]["osm_way"])))) for c in CROSSINGS["crossings"]}
    assert _plan_crossings(scoped) == listed
    t = FrameTransformer(frame)
    for c in CROSSINGS["crossings"]:
        x, y = t.lonlat_to_unit([c["lonlat"][0]], [c["lonlat"][1]])
        assert aoi.exterior.distance(shapely.Point(float(x[0]), float(y[0]))) >= 100.0
