"""Point reading: FR-IN-1, FR-IN-2, FR-IN-3, FR-PN-1."""

import shutil
import subprocess

import laspy
import numpy as np
import pytest
from pyproj import CRS, Transformer

from roader.io import (
    NOISE_CLASSES,
    FrameTransformer,
    from_synthetic,
    read_points,
    unit_frame,
)
from roader.model.synthetic import overpass_points

LON, LAT = -71.8161, 42.38906
AOI = (LON - 0.002, LAT - 0.0015, LON + 0.002, LAT + 0.0015)
N_IN, N_OUT = 300, 40


def _scene(seed: int = 3):
    rng = np.random.default_rng(seed)
    lon = np.concatenate([rng.uniform(AOI[0], AOI[2], N_IN), rng.uniform(AOI[2] + 0.001, AOI[2] + 0.01, N_OUT)])
    lat = np.concatenate([rng.uniform(AOI[1], AOI[3], N_IN), rng.uniform(AOI[1], AOI[3], N_OUT)])
    n = lon.size
    cls = np.where(np.arange(n) % 2 == 0, 2, 1).astype(np.uint8)
    cls[10:20] = 7
    cls[20:25] = 18
    cls[25:35] = 17
    withheld = np.zeros(n, bool)
    withheld[:10] = True
    withheld[40:45] = True
    return {
        "lon": lon,
        "lat": lat,
        "z": rng.uniform(160.0, 200.0, n),
        "cls": cls,
        "withheld": withheld,
        "intensity": rng.integers(0, 60000, n).astype(np.uint16),
        "number_of_returns": np.full(n, 2, np.uint8),
        "return_number": (np.arange(n) % 2 + 1).astype(np.uint8),
        "scan_angle_raw": rng.integers(-3000, 3000, n).astype(np.int16),
        "gps_time": rng.uniform(3.0e8, 3.1e8, n),
        "point_source_id": (np.arange(n) % 3 + 101).astype(np.uint16),
    }


def _write(path, scene, point_format=6, crs="EPSG:3857"):
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = to_m.transform(scene["lon"], scene["lat"])
    version = "1.4" if point_format >= 6 else "1.2"
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([np.floor(x.min()), np.floor(y.min()), 0.0])
    if crs:
        header.add_crs(CRS.from_user_input(crs))
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, scene["z"]
    las.classification = scene["cls"]
    las.withheld = scene["withheld"].astype(np.uint8)
    las.intensity = scene["intensity"]
    las.return_number = scene["return_number"]
    las.number_of_returns = scene["number_of_returns"]
    las.point_source_id = scene["point_source_id"]
    if point_format >= 6:
        las.scan_angle = scene["scan_angle_raw"]
    else:
        las.scan_angle_rank = (scene["scan_angle_raw"] // 100).astype(np.int8)
    if "gps_time" in las.point_format.dimension_names:
        las.gps_time = scene["gps_time"]
    las.write(str(path))
    return x, y


def _expected_mask(scene):
    inside = np.arange(scene["cls"].size) < N_IN
    return inside & ~np.isin(scene["cls"], NOISE_CLASSES) & ~scene["withheld"]


@pytest.mark.parametrize("suffix", [".las", ".laz"])
def test_fr_pn_1_drops_withheld_and_noise_keeps_dims(tmp_path, suffix):
    scene = _scene()
    path = tmp_path / f"t{suffix}"
    x, y = _write(path, scene)
    frame = unit_frame(LON, LAT)
    pts, rec = read_points(path, AOI, frame, collection_index=2, name="test", year=2021, quality_level="QL1", chunk_size=37)

    keep = _expected_mask(scene)
    assert len(pts) == int(keep.sum())
    assert not np.isin(pts.classification, NOISE_CLASSES).any()
    # FR-IN-3: classes other than noise are not filtered.
    assert set(np.unique(pts.classification)) == {1, 2, 17}
    ux, uy = FrameTransformer(frame, "EPSG:3857").to_unit(x[keep], y[keep])
    np.testing.assert_allclose(pts.xyz[:, 0], ux, atol=2e-3)
    np.testing.assert_allclose(pts.xyz[:, 1], uy, atol=2e-3)
    np.testing.assert_allclose(pts.xyz[:, 2], scene["z"][keep], atol=1e-3)
    np.testing.assert_array_equal(pts.intensity, scene["intensity"][keep].astype(float))
    np.testing.assert_array_equal(pts.return_number, scene["return_number"][keep])
    np.testing.assert_array_equal(pts.number_of_returns, scene["number_of_returns"][keep])
    np.testing.assert_allclose(pts.scan_angle, scene["scan_angle_raw"][keep] * 0.006)
    np.testing.assert_allclose(pts.gps_time, scene["gps_time"][keep])
    np.testing.assert_array_equal(pts.point_source_id, scene["point_source_id"][keep])
    assert (pts.collection == 2).all()
    assert pts.xyz.dtype == np.float64 and pts.classification.dtype == np.uint8
    assert pts.point_source_id.dtype == np.uint16

    assert rec.name == "test" and rec.year == 2021 and rec.quality_level == "QL1"
    assert rec.horizontal_crs == "EPSG:3857"
    assert rec.vertical_crs.startswith("NAVD88 (assumed")
    assert rec.density_pavement_pts_m2 is None
    assert rec.point_count == len(pts)


def test_fr_pn_1_absent_dims_are_nan(tmp_path):
    scene = _scene()
    path = tmp_path / "pf0.las"
    _write(path, scene, point_format=0)
    pts, _ = read_points(path, AOI, unit_frame(LON, LAT))
    assert len(pts) == int(_expected_mask(scene).sum())
    assert np.isnan(pts.gps_time).all()
    np.testing.assert_allclose(pts.scan_angle, scene["scan_angle_raw"][_expected_mask(scene)] // 100)


def test_fr_in_2_compound_crs_vertical_recorded(tmp_path):
    scene = _scene()
    path2 = tmp_path / "utm.las"
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:6348", always_xy=True)
    ux, uy = to_utm.transform(scene["lon"], scene["lat"])
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001] * 3)
    header.offsets = np.array([np.floor(ux.min()), np.floor(uy.min()), 0.0])
    header.add_crs(CRS.from_user_input("EPSG:6348+5703"))
    las = laspy.LasData(header)
    las.x, las.y, las.z = ux, uy, scene["z"]
    las.classification = scene["cls"]
    las.write(str(path2))
    pts, rec = read_points(path2, AOI, unit_frame(LON, LAT))
    assert rec.horizontal_crs == "EPSG:6348"
    assert "NAVD88" in rec.vertical_crs and "assumed" not in rec.vertical_crs
    expected = (np.arange(scene["cls"].size) < N_IN) & ~np.isin(scene["cls"], NOISE_CLASSES)
    assert len(pts) == int(expected.sum())
    np.testing.assert_allclose(np.sort(pts.xyz[:, 2]), np.sort(scene["z"][expected]), atol=1e-3)


def test_fr_in_2_non_navd88_vertical_refused(tmp_path):
    scene = _scene()
    path = tmp_path / "ellipsoidal.las"
    _write(path, scene, crs="EPSG:3857")
    with pytest.raises(ValueError, match="not NAVD88"):
        read_points(path, AOI, unit_frame(LON, LAT), source_crs="EPSG:6348+5714")


def test_fr_in_2_missing_crs_needs_override(tmp_path):
    scene = _scene()
    path = tmp_path / "nocrs.las"
    _write(path, scene, crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        read_points(path, AOI, unit_frame(LON, LAT))
    pts, _ = read_points(path, AOI, unit_frame(LON, LAT), source_crs="EPSG:3857")
    assert len(pts) == int(_expected_mask(scene).sum())


@pytest.mark.skipif(shutil.which("pdal") is None, reason="pdal CLI not installed")
def test_fr_in_1_copc_spatial_query(tmp_path):
    scene = _scene()
    src = tmp_path / "src.las"
    _write(src, scene)
    copc = tmp_path / "t.copc.laz"
    subprocess.run(["pdal", "translate", str(src), str(copc), "-w", "writers.copc"], check=True, capture_output=True)
    pts, rec = read_points(copc, AOI, unit_frame(LON, LAT))
    keep = _expected_mask(scene)
    assert len(pts) == int(keep.sum())
    # writers.copc stores 0.01 m steps.
    np.testing.assert_allclose(np.sort(pts.xyz[:, 2]), np.sort(scene["z"][keep]), atol=6e-3)
    assert rec.horizontal_crs == "EPSG:3857"


def test_from_synthetic_drops_noise():
    sp = overpass_points(spacing=1.0)
    pts = from_synthetic(sp, collection_index=1)
    keep = ~np.isin(sp.classification, NOISE_CLASSES)
    assert len(pts) == int(keep.sum()) < sp.xyz.shape[0]
    np.testing.assert_array_equal(pts.xyz, sp.xyz[keep])
    assert np.isnan(pts.scan_angle).all() and np.isnan(pts.gps_time).all()
    assert (pts.collection == 1).all()
    assert len(from_synthetic(sp, drop_noise=False)) == sp.xyz.shape[0]
