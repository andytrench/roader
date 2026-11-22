"""Synthetic corridor, prism, driveway, and deck tests. No network and no LAZ."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon

from roader.legacy.assign import assign_carriageways
from roader.legacy.decks import _hold_runs, ribbons_from_clouds
from roader.legacy.export import write_gpkg
from roader.legacy.lidar_layers import LidarLayers, _layers_from_las
from roader.legacy.model import Carriageway, RoadScene
from roader.legacy.sidewalks import infer_sidewalk_buffers, sidewalks_from_tags
from roader.legacy.surface import Cloud, boundary_edge_count, open_ndvi_sampler


def _grid(x0, x1, y0, y1, z, step=2.0) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(x0, x1 + step * 0.5, step)
    ys = np.arange(y0, y1 + step * 0.5, step)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    return xy, np.full(len(xy), z, dtype=float)


def _layers(ground_xy, ground_z, bridge_xy=None, bridge_z=None) -> LidarLayers:
    if bridge_xy is None:
        bridge_xy = np.zeros((0, 2))
        bridge_z = np.zeros((0,))
    return LidarLayers(
        ground_xy=np.asarray(ground_xy, dtype=float),
        ground_z=np.asarray(ground_z, dtype=float),
        bridge_xy=np.asarray(bridge_xy, dtype=float),
        bridge_z=np.asarray(bridge_z, dtype=float),
        all_xy=np.asarray(ground_xy, dtype=float),
        all_z=np.asarray(ground_z, dtype=float),
        all_class=np.full(len(ground_z), 2),
        crs_epsg=26986,
    )


def _ways(rows: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:26986")


def _fit(line, xy, z, highway, bridge_xy=None, bridge_z=None):
    ground = Cloud.from_arrays(xy, z, None)
    if bridge_xy is None:
        bridge = Cloud.from_arrays(np.zeros((0, 2)), np.zeros((0,)), None)
    else:
        bridge = Cloud.from_arrays(bridge_xy, bridge_z, None)
    return ribbons_from_clouds(
        line,
        ground,
        bridge,
        highway=highway,
        pavement_thickness_m=0.2,
        deck_thickness_m=0.8,
        deck_height_m=2.5,
    )


def test_flat_road_crowns_the_edges_and_keeps_the_center() -> None:
    xy, z = _grid(0, 40, -4, 4, 8.0, step=2.0)
    fits = _fit(LineString([(0, 0), (40, 0)]), xy, z, "residential")
    fit = fits[0]
    assert fit.reason is None and fit.section == "standard"
    assert abs(float(np.mean(fit.profile_z)) - 8.0) < 0.05
    half = float(fit.half_width_m)
    center = [
        coord[2]
        for face in fit.raw_faces
        if face.role == "top"
        for coord in face.coords
        if abs(coord[1]) < 0.2
    ]
    edges = [
        coord[2]
        for face in fit.raw_faces
        if face.role == "top"
        for coord in face.coords
        if abs(abs(coord[1]) - half) < 0.2
    ]
    assert center and edges
    drop = float(np.median(center) - np.median(edges))
    assert abs(drop - 0.02 * half) < 0.02


def test_one_station_bump_follows_the_grade() -> None:
    xs = np.arange(0.0, 80.0 + 1e-6, 2.0)
    ys = np.arange(-4.0, 4.0 + 1e-6, 2.0)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    zz = 10.0 + 0.05 * xy[:, 0]
    zz[(xy[:, 0] >= 38.0) & (xy[:, 0] <= 42.0)] += 0.35
    fits = _fit(LineString([(0, 0), (80, 0)]), xy, zz, "secondary")
    profile = np.asarray(fits[0].profile_z)
    xs_station = np.linspace(0.0, 80.0, len(profile))
    trend = 10.0 + 0.05 * xs_station
    assert np.max(np.abs(profile - trend)) < 0.12
    assert profile[-1] - profile[0] > 3.5


def test_straight_cross_slope_stays_on_the_lidar() -> None:
    xs = np.arange(0.0, 40.0 + 1e-6, 2.0)
    ys = np.arange(-6.0, 6.0 + 1e-6, 2.0)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    zz = 10.0 + 0.06 * xy[:, 1]
    fits = _fit(LineString([(0, 0), (40, 0)]), xy, zz, "residential")
    fit = fits[0]
    assert fit.section == "lidar"
    half = float(fit.half_width_m)
    left = [
        coord[2]
        for face in fit.raw_faces
        if face.role == "top"
        for coord in face.coords
        if abs(coord[1] - half) < 0.2
    ]
    right = [
        coord[2]
        for face in fit.raw_faces
        if face.role == "top"
        for coord in face.coords
        if abs(coord[1] + half) < 0.2
    ]
    assert float(np.median(left) - np.median(right)) > 0.4


def test_left_curve_raises_the_outside_edge() -> None:
    radius = 50.0
    angles = np.linspace(-np.pi / 2, 0.0, 16)
    line = LineString([(radius * np.cos(angle), radius + radius * np.sin(angle)) for angle in angles])
    points: list[tuple[float, float]] = []
    heights: list[float] = []
    for distance in np.linspace(0.0, line.length, 36):
        point = line.interpolate(float(distance))
        ahead = line.interpolate(min(line.length, float(distance) + 1.0))
        tangent = np.array([ahead.x - point.x, ahead.y - point.y])
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            continue
        tangent /= norm
        normal = np.array([-tangent[1], tangent[0]])
        for lateral in np.linspace(-5.0, 5.0, 6):
            xy = np.array([point.x, point.y]) + normal * lateral
            points.append((float(xy[0]), float(xy[1])))
            heights.append(20.0 - 0.06 * lateral)
    fits = _fit(line, np.array(points), np.array(heights), "motorway_link")
    fit = fits[0]
    assert fit.reason is None and fit.section == "standard"
    half = float(fit.half_width_m)
    index = len(fit.profile_xy) // 2
    mid = np.array(fit.profile_xy[index])
    step = np.array(fit.profile_xy[index + 1])
    tangent = step - mid
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    left_xy = mid + normal * half
    right_xy = mid - normal * half

    def nearest_z(target: np.ndarray) -> float:
        best = None
        best_d = 1e9
        for face in fit.raw_faces:
            if face.role != "top":
                continue
            for coord in face.coords:
                dist = float(np.hypot(coord[0] - target[0], coord[1] - target[1]))
                if dist < best_d:
                    best_d = dist
                    best = coord[2]
        assert best is not None and best_d < 2.0
        return best

    assert nearest_z(right_xy) > nearest_z(left_xy) + 0.4


def test_car_bumps_stay_off_the_profile() -> None:
    xy, z = _grid(0, 50, -5, 5, 10.0, step=1.0)
    cars = np.array([[10.0, 0.0], [10.0, 0.4], [20.0, 0.0], [30.0, -0.3], [40.0, 0.2]])
    xy = np.vstack([xy, cars])
    z = np.concatenate([z, np.full(len(cars), 11.5)])
    fits = _fit(LineString([(0, 0), (50, 0)]), xy, z, "residential")
    assert len(fits) == 1 and fits[0].reason is None
    profile = np.asarray(fits[0].profile_z)
    assert np.max(np.abs(profile - 10.0)) < 0.05


def test_missing_intensity_still_builds_closed_prism() -> None:
    xy, z = _grid(0, 40, -4, 4, 8.0, step=2.0)
    fits = _fit(LineString([(0, 0), (40, 0)]), xy, z, "residential")
    assert len(fits) == 1 and fits[0].reason is None
    assert fits[0].raw_faces
    assert boundary_edge_count(fits[0].raw_faces) == 0
    assert all(len(coord) == 3 for face in fits[0].raw_faces for coord in face.coords)
    assert abs(float(np.mean(fits[0].profile_z)) - 8.0) < 0.05


def test_wide_blob_is_clipped_to_class_width() -> None:
    xy, z = _grid(0, 50, -20, 20, 5.0, step=2.0)
    fits = _fit(LineString([(0, 0), (50, 0)]), xy, z, "residential")
    assert len(fits) == 1 and fits[0].reason is None
    assert fits[0].half_width_m is not None
    assert fits[0].half_width_m <= 4.0 + 1e-6
    for face in fits[0].raw_faces:
        for x, y, _z in face.coords:
            assert abs(y) <= 4.01
            assert 0.0 - 0.01 <= x <= 50.0 + 0.01


def test_compact_service_blob_is_parking_bleed() -> None:
    xy, z = _grid(0, 20, -15, 15, 3.0, step=2.0)
    fits = _fit(LineString([(0, 0), (20, 0)]), xy, z, "service")
    assert len(fits) == 1
    assert fits[0].reason == "parking_bleed"
    assert fits[0].raw_faces == []


def test_driveway_tag_produces_no_solid() -> None:
    xy, z = _grid(0, 20, -4, 4, 6.0)
    gdf = _ways(
        [
            {
                "highway": "service",
                "service": "driveway",
                "osmid": 9,
                "geometry": LineString([(0, 0), (20, 0)]),
            }
        ]
    )
    result = assign_carriageways(gdf, _layers(xy, z), target_crs="EPSG:26986")
    assert result.carriageways == []
    assert [skip.reason for skip in result.skips] == ["driveway"]


def test_dead_end_spur_is_a_driveway() -> None:
    ground_xy, ground_z = _grid(0, 100, -6, 6, 4.0, step=2.0)
    spur_xy, spur_z = _grid(48, 52, 0, 20, 4.0, step=2.0)
    xy = np.vstack([ground_xy, spur_xy])
    z = np.concatenate([ground_z, spur_z])
    gdf = _ways(
        [
            {
                "highway": "residential",
                "service": "",
                "osmid": 1,
                "geometry": LineString([(0, 0), (100, 0)]),
            },
            {
                "highway": "residential",
                "service": "",
                "osmid": 2,
                "geometry": LineString([(50, 0), (50, 20)]),
            },
        ]
    )
    result = assign_carriageways(gdf, _layers(xy, z), target_crs="EPSG:26986")
    assert [skip.reason for skip in result.skips] == ["driveway"]
    assert len(result.carriageways) == 1
    assert result.carriageways[0].osm_ids == [1]
    assert result.carriageways[0].faces


def test_stacked_profiles_make_a_road_void() -> None:
    ground_xy, ground_z = _grid(0, 40, -20, 20, 10.0, step=2.0)
    bridge_xy, bridge_z = _grid(16, 24, -20, 20, 18.0, step=2.0)
    gdf = _ways(
        [
            {
                "highway": "primary",
                "osmid": 10,
                "geometry": LineString([(0, 0), (40, 0)]),
            },
            {
                "highway": "primary",
                "osmid": 11,
                "geometry": LineString([(20, -20), (20, 20)]),
            },
        ]
    )
    result = assign_carriageways(
        gdf,
        _layers(ground_xy, ground_z, bridge_xy, bridge_z),
        target_crs="EPSG:26986",
        deck_thickness_m=0.8,
        height_above_ground_m=2.5,
    )
    lower = next(c for c in result.carriageways if c.osm_ids == [10])
    upper = next(c for c in result.carriageways if c.osm_ids == [11])
    assert lower.layer == "terrain"
    assert upper.layer == "deck"
    assert abs(float(lower.z_mean_m) - 10.0) < 0.15
    assert abs(float(upper.z_mean_m) - 18.0) < 0.15
    assert len(result.decks) == 1
    road_voids = [void for void in result.voids if void.kind == "road"]
    assert len(road_voids) == 1
    assert road_voids[0].clearance_m is not None
    assert road_voids[0].clearance_m > 0


def test_embankment_is_fill_down_to_the_toe() -> None:
    """A bridge tag on a bank that still meets the pavement is fill, not a void."""
    pavement_xy, pavement_z = _grid(0, 80, -4, 4, 16.0, step=2.0)
    lip_xy, lip_z = _grid(0, 80, 6, 8, 15.9, step=2.0)
    lip_b_xy, lip_b_z = _grid(0, 80, -8, -6, 15.9, step=2.0)
    toe_xy, toe_z = _grid(0, 80, 14, 22, 10.0, step=2.0)
    toe_b_xy, toe_b_z = _grid(0, 80, -22, -14, 10.0, step=2.0)
    xy = np.vstack([pavement_xy, lip_xy, lip_b_xy, toe_xy, toe_b_xy])
    z = np.concatenate([pavement_z, lip_z, lip_b_z, toe_z, toe_b_z])
    gdf = _ways(
        [
            {
                "highway": "secondary",
                "bridge": "yes",
                "osmid": 8,
                "geometry": LineString([(0, 0), (80, 0)]),
            }
        ]
    )
    result = assign_carriageways(
        gdf,
        _layers(xy, z),
        target_crs="EPSG:26986",
        deck_thickness_m=0.8,
        height_above_ground_m=2.5,
    )
    assert len(result.carriageways) == 1
    assert result.carriageways[0].layer == "terrain"
    assert result.decks == []
    lows = [coord[2] for face in result.carriageways[0].faces for coord in face.triangle.exterior.coords]
    assert min(lows) < 11.0
    assert max(lows) > 15.0
    ys = [coord[1] for face in result.carriageways[0].faces for coord in face.triangle.exterior.coords]
    assert min(ys) < -10.0
    assert max(ys) > 10.0


def test_a_cut_on_one_side_does_not_spike_the_fill() -> None:
    pavement_xy, pavement_z = _grid(0, 80, -4, 4, 20.0, step=2.0)
    lip_xy, lip_z = _grid(0, 80, -8, -6, 19.6, step=2.0)
    toe_xy, toe_z = _grid(0, 80, -22, -14, 17.0, step=2.0)
    cut_lip_xy, cut_lip_z = _grid(0, 80, 6, 8, 0.0, step=2.0)
    cut_toe_xy, cut_toe_z = _grid(0, 80, 14, 22, 0.0, step=2.0)
    xy = np.vstack([pavement_xy, lip_xy, toe_xy, cut_lip_xy, cut_toe_xy])
    z = np.concatenate([pavement_z, lip_z, toe_z, cut_lip_z, cut_toe_z])
    fits = _fit(LineString([(0, 0), (80, 0)]), xy, z, "secondary")
    fit = fits[0]
    assert fit.reason is None
    lows = [coord[2] for face in fit.raw_faces for coord in face.coords]
    assert min(lows) > 14.0
    assert min(lows) < 18.0
    ys = [coord[1] for face in fit.raw_faces for coord in face.coords]
    assert min(ys) < -10.0
    assert max(ys) < 8.0


def test_class17_deck_matches_ground_on_the_span() -> None:
    """USGS often classifies the deck as ground and class 17 at the same height."""
    center_xy, center_z = _grid(0, 80, -5, 5, 25.0, step=2.0)
    shoulder_xy, shoulder_z = _grid(0, 80, 12, 20, 18.0, step=2.0)
    other_xy, other_z = _grid(0, 80, -20, -12, 18.0, step=2.0)
    ground_xy = np.vstack([center_xy, shoulder_xy, other_xy])
    ground_z = np.concatenate([center_z, shoulder_z, other_z])
    gdf = _ways(
        [
            {
                "highway": "secondary",
                "bridge": "yes",
                "osmid": 7,
                "geometry": LineString([(0, 0), (80, 0)]),
            }
        ]
    )
    result = assign_carriageways(
        gdf,
        _layers(ground_xy, ground_z, center_xy, center_z),
        target_crs="EPSG:26986",
        deck_thickness_m=0.8,
        height_above_ground_m=2.5,
    )
    assert len(result.carriageways) == 1
    assert result.carriageways[0].layer == "deck"
    assert result.decks
    assert result.voids
    assert result.voids[0].clearance_m is not None
    assert result.voids[0].clearance_m > 0


def test_export_writes_solid_faces(tmp_path: Path) -> None:
    xy, z = _grid(0, 30, -4, 4, 2.0, step=2.0)
    fit = _fit(LineString([(0, 0), (30, 0)]), xy, z, "residential")[0]
    carriageway = Carriageway(
        id="cw",
        centerline=fit.centerline,
        polygon=fit.polygon,
        highway="residential",
        layer="terrain",
        z_mean_m=fit.z_mean,
        faces=[],
    )
    from roader.legacy.model import SolidFace

    carriageway.faces = [
        SolidFace(
            "cw",
            face.role,
            Polygon([face.coords[0], face.coords[1], face.coords[2], face.coords[0]]),
        )
        for face in fit.raw_faces
    ]
    out = tmp_path / "roads.gpkg"
    write_gpkg(RoadScene(crs="EPSG:26986", carriageways=[carriageway]), out)
    solids = gpd.read_file(out, layer="carriageway_solids")
    assert set(solids["role"]) == {"top", "bottom", "wall"}
    assert len(solids) == len(fit.raw_faces)


def test_ndvi_sampler_uses_red_and_nir(tmp_path: Path) -> None:
    path = tmp_path / "naip.tif"
    transform = from_bounds(0, 0, 1, 1, 1, 1)
    data = np.zeros((4, 1, 1), dtype=np.uint16)
    data[0, 0, 0] = 100
    data[3, 0, 0] = 200
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=4,
        dtype="uint16",
        crs="EPSG:26986",
        transform=transform,
    ) as dst:
        dst.write(data)
    sampler = open_ndvi_sampler(path)
    assert sampler is not None
    try:
        value = sampler(np.array([[0.5, 0.5]]))
    finally:
        sampler.close()
    assert abs(float(value[0]) - (100 / 300)) < 1e-6

    rgb = tmp_path / "rgb.tif"
    with rasterio.open(
        rgb,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=3,
        dtype="uint16",
        crs="EPSG:26986",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((3, 1, 1), dtype=np.uint16))
    black = open_ndvi_sampler(rgb)
    assert black is not None
    try:
        assert float(black(np.array([[0.5, 0.5]]))[0]) == 0.0
    finally:
        black.close()
    green = tmp_path / "green.tif"
    canopy = np.zeros((3, 1, 1), dtype=np.uint16)
    canopy[:, 0, 0] = (40, 160, 40)
    with rasterio.open(
        green,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=3,
        dtype="uint16",
        crs="EPSG:26986",
        transform=transform,
    ) as dst:
        dst.write(canopy)
    plants = open_ndvi_sampler(green)
    assert plants is not None
    try:
        assert float(plants(np.array([[0.5, 0.5]]))[0]) > 0.35
    finally:
        plants.close()
    assert open_ndvi_sampler(tmp_path / "missing.tif") is None


def test_intensity_dimension_is_optional(tmp_path: Path) -> None:
    class Header:
        def parse_crs(self):
            return None

    class PointFormat:
        dimension_names = ("X", "Y", "Z", "classification")

    class Las:
        header = Header()
        point_format = PointFormat()
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        z = np.array([1.0, 8.0])
        classification = np.array([2, 17])

    layers = _layers_from_las(Las(), None)
    assert layers.has_intensity is False
    assert layers.ground_intensity is None
    assert len(layers.ground_xy) == 1
    assert len(layers.bridge_xy) == 1

    laspy = __import__("laspy")
    las = laspy.create(point_format=0, file_version="1.2")
    las.x = np.array([1.0, 2.0, 3.0])
    las.y = np.array([4.0, 4.0, 4.0])
    las.z = np.array([5.0, 6.0, 9.0])
    las.classification = np.array([2, 2, 17], dtype=np.uint8)
    las.intensity = np.array([10, 20, 30], dtype=np.uint16)
    path = tmp_path / "points.las"
    las.write(path)
    loaded = __import__("roader.legacy.lidar_layers", fromlist=["load_lidar_layers"]).load_lidar_layers(path)
    assert loaded.has_intensity
    assert loaded.ground_intensity.tolist() == [10.0, 20.0]
    assert loaded.bridge_intensity.tolist() == [30.0]


def test_unclassified_has_no_invented_sidewalk() -> None:
    line = LineString([(0, 0), (40, 0)])
    plain = Carriageway(
        id="cw",
        centerline=line,
        polygon=line.buffer(3),
        highway="unclassified",
        layer="terrain",
        attrs={"half_width_m": 3.5},
    )
    assert sidewalks_from_tags([plain], enabled=True) == []
    assert infer_sidewalk_buffers([plain], enabled=True) == []
    tagged = Carriageway(
        id="cw2",
        centerline=line,
        polygon=line.buffer(3),
        highway="residential",
        layer="terrain",
        attrs={"sidewalk": "both", "half_width_m": 4.0},
    )
    found = sidewalks_from_tags([tagged], enabled=True)
    assert len(found) == 1
    assert found[0].source == "osm"
    assert not found[0].geometry.is_empty
    motorway = Carriageway(
        id="cw3",
        centerline=line,
        polygon=line.buffer(8),
        highway="motorway",
        layer="terrain",
    )
    assert sidewalks_from_tags([motorway], enabled=True) == []


def test_colinear_edges_become_one_road() -> None:
    xy, z = _grid(0, 80, -6, 6, 8.0, step=2.0)
    gdf = _ways(
        [
            {"highway": "residential", "osmid": 1, "geometry": LineString([(0, 0), (40, 0)])},
            {"highway": "residential", "osmid": 2, "geometry": LineString([(40, 0), (80, 0)])},
        ]
    )
    result = assign_carriageways(gdf, _layers(xy, z), target_crs="EPSG:26986")
    assert len(result.carriageways) == 1
    assert result.carriageways[0].centerline.length > 70


def test_a_bridge_tag_stays_a_separate_piece() -> None:
    xy, z = _grid(0, 80, -6, 6, 8.0, step=2.0)
    gdf = _ways(
        [
            {"highway": "residential", "osmid": 1, "geometry": LineString([(0, 0), (40, 0)])},
            {
                "highway": "residential",
                "bridge": "yes",
                "osmid": 2,
                "geometry": LineString([(40, 0), (80, 0)]),
            },
        ]
    )
    result = assign_carriageways(gdf, _layers(xy, z), target_crs="EPSG:26986")
    assert len(result.carriageways) == 2


def test_a_ramp_reaches_the_parent_edge() -> None:
    xy, z = _grid(0, 80, -8, 24, 6.0, step=2.0)
    gdf = _ways(
        [
            {"highway": "motorway", "osmid": 1, "geometry": LineString([(0, 0), (80, 0)])},
            {"highway": "motorway_link", "osmid": 2, "geometry": LineString([(40, 20), (40, 12)])},
        ]
    )
    result = assign_carriageways(gdf, _layers(xy, z), target_crs="EPSG:26986")
    link = next(item for item in result.carriageways if item.highway == "motorway_link")
    assert min(coord[1] for coord in link.centerline.coords) < 9.0


def test_a_service_copy_of_a_ramp_is_dropped() -> None:
    xy, z = _grid(0, 80, -8, 8, 6.0, step=2.0)
    line = LineString([(0, 0), (80, 0)])
    gdf = _ways(
        [
            {"highway": "motorway_link", "osmid": 1, "geometry": line},
            {"highway": "service", "osmid": 2, "geometry": LineString(line.coords)},
        ]
    )
    result = assign_carriageways(gdf, _layers(xy, z), target_crs="EPSG:26986")
    assert len(result.carriageways) == 1
    assert result.carriageways[0].highway == "motorway_link"
    assert any(skip.reason == "duplicate" for skip in result.skips)


def test_canopy_does_not_widen_the_road() -> None:
    xy, z = _grid(0, 40, -10, 10, 8.0, step=2.0)
    line = LineString([(0, 0), (40, 0)])

    def score(points: np.ndarray) -> np.ndarray:
        return np.where(np.abs(points[:, 1]) > 4.0, 0.9, 0.0)

    wide = _fit(line, xy, z, "motorway")
    kept = ribbons_from_clouds(
        line,
        Cloud.from_arrays(xy, z, None),
        Cloud.from_arrays(np.zeros((0, 2)), np.zeros((0,)), None),
        highway="motorway",
        pavement_thickness_m=0.2,
        deck_thickness_m=0.8,
        deck_height_m=2.5,
        ndvi=score,
    )
    assert wide[0].half_width_m == 8.0
    assert kept[0].half_width_m <= 5.0


def test_a_short_gap_does_not_break_a_run() -> None:
    distances = np.arange(6, dtype=float) * 5.0
    held = _hold_runs(np.array([True, True, False, False, True, True]), distances, 18.0)
    assert held.all()
    broken = _hold_runs(np.array([True, True, False, False, True]), np.array([0.0, 5.0, 15.0, 30.0, 55.0]), 18.0)
    assert not broken[2] and not broken[3]
