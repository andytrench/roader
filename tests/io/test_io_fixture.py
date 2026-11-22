"""Real milestone 1 fixture, when fetched (scripts/fetch_fixtures.py): FR-IN-1, FR-IN-2, FR-PL-3."""

from pathlib import Path

import numpy as np
import pytest
import shapely

from roader.io import (
    FrameTransformer,
    in_scope,
    load_unit_config,
    read_points,
    read_vectors,
    verify_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "units" / "wboylston_laurel_i190.toml"
cfg = load_unit_config(CONFIG)

pytestmark = pytest.mark.skipif(
    not (cfg.lidar[0].path.exists() and cfg.vectors["osm"].path.exists()),
    reason="fixture not fetched; run scripts/fetch_fixtures.py --site wboylston_laurel_i190",
)


@pytest.fixture(scope="module")
def loaded():
    frame = cfg.frame()
    src = cfg.lidar[0]
    pts, rec = read_points(src.path, cfg.aoi_lonlat, frame, name=src.name, year=src.year, quality_level=src.quality_level)
    return frame, pts, rec


def test_fr_pl_3_fixture_matches_pins():
    verify_inputs(cfg)


def test_fr_in_1_fixture_density_and_classes(loaded):
    frame, pts, rec = loaded
    w, s, e, n = cfg.aoi_lonlat
    t = FrameTransformer(frame)
    x, y = t.lonlat_to_unit([w, e, e, w], [s, s, n, n])
    area = shapely.Polygon(np.column_stack([x, y])).area
    assert len(pts) / area >= 8.0
    assert rec.point_count == len(pts)
    assert rec.horizontal_crs == "EPSG:3857" and rec.vertical_crs.startswith("NAVD88")
    classes = set(np.unique(pts.classification).tolist())
    assert {2, 17} <= classes and not classes & {7, 18}
    assert np.isfinite(pts.gps_time).all() and np.isfinite(pts.scan_angle).all()


def test_scope_fixture_vectors(loaded):
    frame, _, _ = loaded
    edges = read_vectors(cfg.vectors, frame)["osm"]
    scoped = in_scope(edges)
    highways = {e.tags["highway"] for e in scoped}
    assert "motorway" in highways
    assert any(e.feature_id == "183371679" for e in scoped)  # Laurel Street bridge
