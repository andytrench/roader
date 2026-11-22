"""Unit config and record: FR-IN-2, FR-IN-7, FR-PL-3, D-008, D-009; tests/sites.yaml layout."""

from pathlib import Path

import pytest
import yaml

import roader
from roader.io import build_unit_record, load_unit_config, sha256_file
from roader.model.schema import LidarCollection

ROOT = Path(__file__).resolve().parents[2]
M1 = "wboylston_laurel_i190"

CONFIG = """
[unit]
site_id = "t"
aoi_lonlat = [-71.82, 42.38, -71.81, 42.39]

[lidar]
path = "lidar.laz"
name = "P"
year = 2021
quality_level = "QL1"

[vectors.osm]
enabled = true
path = "osm.json"
license = "ODbL-1.0"
sha256 = "{osm_sha}"

[vectors.tiger]
enabled = false
path = "tiger.json"
license = "public domain"
"""


def _config(tmp_path, osm_sha):
    (tmp_path / "osm.json").write_text('{"elements": []}')
    path = tmp_path / "unit.toml"
    path.write_text(CONFIG.format(osm_sha=osm_sha))
    return load_unit_config(path, root=tmp_path)


def test_fr_in_2_unit_record(tmp_path):
    _config(tmp_path, "")
    sha = sha256_file(tmp_path / "osm.json")
    cfg = _config(tmp_path, sha)
    assert cfg.imagery_enabled is False  # D-009
    assert cfg.lidar[0].path == tmp_path / "lidar.laz"
    assert cfg.frame().anchor_lon == pytest.approx(-71.815)
    rec = build_unit_record(cfg, [LidarCollection("P", 2021, "QL1", None, "EPSG:3857", "NAVD88 (assumed)")])
    assert rec.unit_id == "t"
    assert rec.software_version == roader.__version__
    assert [(v.name, v.enabled, v.sha256) for v in rec.vectors] == [("osm", True, sha), ("tiger", False, "")]
    assert rec.lidar[0].name == "P"


def test_fr_pl_3_pin_mismatch_raises(tmp_path):
    cfg = _config(tmp_path, "0" * 64)
    with pytest.raises(ValueError, match="pinned"):
        build_unit_record(cfg, [])


def test_m1_unit_config_matches_sites_yaml():
    cfg = load_unit_config(ROOT / "configs" / "units" / f"{M1}.toml")
    sites = {s["id"]: s for s in yaml.safe_load((ROOT / "tests" / "sites.yaml").read_text())["sites"]}
    site = sites[M1]
    assert cfg.site_id == M1
    assert list(cfg.aoi_lonlat) == site["aoi_lonlat"]
    assert cfg.lidar[0].sha256 == site["lidar_sha256"]
    assert cfg.vectors["osm"].sha256 == site["vector_sha256"]
    assert cfg.vectors["osm"].enabled and not cfg.imagery_enabled
    assert cfg.lidar[0].path == ROOT / site["lidar_path"]


def test_sites_yaml_lists_all_spec_classes():
    doc = yaml.safe_load((ROOT / "tests" / "sites.yaml").read_text())
    classes = {c["class"]: c for c in doc["classes"]}
    assert len(classes) == 12
    site_ids = {s["id"] for s in doc["sites"]}
    for c in classes.values():
        assert c["status"] in ("assigned", "unassigned")
        assert (c["status"] == "assigned") == bool(c.get("sites"))
        assert set(c.get("sites", [])) <= site_ids
    for site in doc["sites"]:
        assert site["class"] in classes
        assert len(site["aoi_lonlat"]) == 4
        for key in ("project", "ept", "year", "quality_level", "horizontal_crs", "vertical_crs"):
            assert key in site["lidar"]
        assert site["vectors"]["license"]
    m1 = [s for s in doc["sites"] if s["milestone"] == 1]
    assert [s["class"] for s in m1] == ["simple_overpass_rural_freeway"]
    assert len(m1[0]["lidar_sha256"]) == 64 and len(m1[0]["vector_sha256"]) == 64


def test_ac_h2_crossing_list_present():
    doc = yaml.safe_load((ROOT / "tests" / "sites" / M1 / "crossings.yaml").read_text())
    assert doc["site_id"] == M1
    assert doc["status"].startswith("draft")
    assert doc["crossings"]
    for c in doc["crossings"]:
        assert c["upper"] and c["lower"] and c["dz_m"] > 0
        assert len(c["lonlat"]) == 2
