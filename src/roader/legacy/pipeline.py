"""End-to-end pilot pipeline."""

from __future__ import annotations

from collections import Counter

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from roader.legacy.assign import assign_carriageways, group_interchanges
from roader.legacy.config import PilotConfig
from roader.legacy.export import write_gpkg
from roader.legacy.lidar_layers import load_lidar_layers
from roader.legacy.model import RoadScene
from roader.legacy.osm_seed import fetch_road_edges, fetch_sidewalk_geometries
from roader.legacy.sidewalks import sidewalks_from_osm, sidewalks_from_tags
from roader.legacy.surface import open_ndvi_sampler


def _bbox_to_projected(
    bbox_wgs84: tuple[float, float, float, float],
    crs: str,
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox_wgs84
    poly = box(west, south, east, north)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    proj = transform(transformer.transform, poly)
    return proj.bounds  # minx, miny, maxx, maxy


def run_pilot(cfg: PilotConfig) -> RoadScene:
    if not cfg.lidar_path.exists():
        raise FileNotFoundError(
            f"LiDAR not found: {cfg.lidar_path}. Place a LAS/LAZ under data/pilot/."
        )

    # Ask OpenStreetMap before reading the cloud. A refused call then does not
    # load the tile, and the next tile waits instead of asking again at once.
    edges = fetch_road_edges(cfg.bbox, network_type=cfg.osm_network_type)
    osm_sw = fetch_sidewalk_geometries(cfg.bbox)

    bbox_proj = _bbox_to_projected(cfg.bbox, cfg.crs)
    layers = load_lidar_layers(cfg.lidar_path, bbox_projected=bbox_proj)
    # Prefer CRS from LiDAR if present
    crs = f"EPSG:{layers.crs_epsg}" if layers.crs_epsg else cfg.crs
    ndvi = open_ndvi_sampler(cfg.naip_path)
    try:
        result = assign_carriageways(
            edges,
            layers,
            target_crs=crs,
            height_above_ground_m=cfg.deck_height_above_ground_m,
            pavement_thickness_m=cfg.pavement_thickness_m,
            deck_thickness_m=cfg.deck_thickness_m,
            ndvi=ndvi,
        )
    finally:
        if ndvi is not None:
            ndvi.close()
    interchanges = group_interchanges(
        result.carriageways,
        z_threshold=cfg.deck_height_above_ground_m,
    )

    sidewalks = sidewalks_from_osm(osm_sw, target_crs=crs)
    if cfg.sidewalk_infer:
        sidewalks.extend(sidewalks_from_tags(result.carriageways, enabled=True))

    scene = RoadScene(
        crs=crs,
        carriageways=result.carriageways,
        decks=result.decks,
        voids=result.voids,
        sidewalks=sidewalks,
        interchanges=interchanges,
        skips=result.skips,
    )
    write_gpkg(scene, cfg.output_gpkg)
    return scene


def scene_summary(scene: RoadScene) -> dict:
    reasons = Counter(skip.reason for skip in scene.skips)
    clearances = [
        void.clearance_m for void in scene.voids if void.clearance_m is not None
    ]
    return {
        "crs": scene.crs,
        "n_carriageways": len(scene.carriageways),
        "n_decks": len(scene.decks),
        "n_voids": len(scene.voids),
        "n_road_voids": sum(1 for void in scene.voids if void.kind == "road"),
        "max_clearance_m": max(clearances) if clearances else None,
        "n_sidewalks": len(scene.sidewalks),
        "n_interchanges": len(scene.interchanges),
        "n_solid_faces": sum(len(carriageway.faces) for carriageway in scene.carriageways),
        "n_skips": len(scene.skips),
        "skip_reasons": dict(sorted(reasons.items())),
        "highways": sorted({c.highway for c in scene.carriageways}),
    }
