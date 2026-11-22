"""Unit config and record: FR-IN-2, FR-IN-7, FR-PL-3, D-008, D-009; tests/sites.yaml layout."""

from pathlib import Path

import pytest
import yaml

import roader
from roader.io import build_unit_record, load_unit_config, sha256_file
from roader.model.schema import LidarCollection

ROOT = Path(__file__).resolve().parents[2]
M1 = "wboylston_laurel_i190"
M2 = "woburn_i93_i95"

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


def _sites():
    return {s["id"]: s for s in yaml.safe_load((ROOT / "tests" / "sites.yaml").read_text())["sites"]}


@pytest.mark.parametrize("site_id", [M1, M2])
def test_unit_config_matches_sites_yaml(site_id):
    cfg = load_unit_config(ROOT / "configs" / "units" / f"{site_id}.toml")
    site = _sites()[site_id]
    assert cfg.site_id == site_id
    assert list(cfg.aoi_lonlat) == site["aoi_lonlat"]
    assert cfg.lidar[0].sha256 == site["lidar_sha256"]
    assert cfg.lidar[0].name == site["lidar"]["project"]
    assert cfg.lidar[0].quality_level == site["lidar"]["quality_level"]
    assert cfg.vectors["osm"].sha256 == site["vector_sha256"]
    assert cfg.vectors["osm"].enabled and not cfg.imagery_enabled
    assert cfg.lidar[0].path == ROOT / site["lidar_path"]


def test_sites_yaml_lists_all_spec_classes():
    doc = yaml.safe_load((ROOT / "tests" / "sites.yaml").read_text())
    classes = {c["class"]: c for c in doc["classes"]}
    assert len(classes) == 12
    sites = {s["id"]: s for s in doc["sites"]}
    for c in classes.values():
        assert c["status"] in ("assigned", "shortlisted", "unassigned")
        assert (c["status"] == "unassigned") == (not c.get("sites"))
        assert set(c.get("sites", [])) <= set(sites)
        for sid in c.get("sites", []):
            assert sites[sid]["class"] == c["class"]
            assert (sites[sid].get("status") == "shortlisted") == (c["status"] == "shortlisted")
    for site in doc["sites"]:
        assert site["class"] in classes
        w, s, e, n = site["aoi_lonlat"]
        assert w < e and s < n
        if "crossing_lonlat" in site:
            lon, lat = site["crossing_lonlat"]
            assert w < lon < e and s < lat < n
        for key in ("project", "ept", "year", "quality_level", "horizontal_crs", "vertical_crs", "density_pts_m2"):
            assert key in site["lidar"]
        assert site["vectors"]["license"]
        if site.get("status") == "shortlisted":
            assert site["lidar_sha256"] == "" and site["vector_sha256"] == "" and "crossings" not in site
            assert site["lidar"]["density_pts_m2"] > 0 and 17 in site["lidar"]["classes_seen"]
        else:
            assert len(site["lidar_sha256"]) == 64 and len(site["vector_sha256"]) == 64
            assert (ROOT / site["crossings"]).exists()
    by_milestone = {m: [s["class"] for s in doc["sites"] if s["milestone"] == m] for m in (1, 2)}
    assert by_milestone[1] == ["simple_overpass_rural_freeway"]
    assert by_milestone[2] == ["cloverleaf_interchange", "four_or_five_level_stack"]


@pytest.mark.parametrize("site_id", [M1, M2])
def test_ac_h2_crossing_list_present(site_id):
    site = _sites()[site_id]
    doc = yaml.safe_load((ROOT / site["crossings"]).read_text())
    assert doc["site_id"] == site_id
    assert doc["status"].startswith("draft")
    assert doc["crossings"]
    w, s, e, n = site["aoi_lonlat"]
    structures = {st["id"]: st for st in doc.get("structure_ends", {}).get("structures", [])}
    if structures:
        assert doc["structure_ends"]["status"].startswith("draft")
    ids = [c["id"] for c in doc["crossings"]]
    assert len(ids) == len(set(ids))
    for c in doc["crossings"]:
        assert c["upper"] and c["lower"] and c["dz_m"] > 0
        assert isinstance(c["upper"]["osm_way"], int) and isinstance(c["lower"]["osm_way"], int)
        assert c["dz_m"] == pytest.approx(c["upper_z_m"] - c["lower_z_m"], abs=0.11)
        assert 0 <= c["skew_deg"] <= 90
        lon, lat = c["lonlat"]
        assert w < lon < e and s < lat < n
        if structures:
            st = structures[c["structure"]]
            assert c["upper"]["osm_way"] in st["carries"] and c["lower"]["osm_way"] in st["over"]
    for st in structures.values():
        assert st["ends"]
        for end in st["ends"]:
            assert end["along"] in st["carries"] and end["length_m"] > 0
            for lon, lat in (end["start_lonlat"], end["end_lonlat"]):
                assert w < lon < e and s < lat < n
