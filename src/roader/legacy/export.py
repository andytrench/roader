"""Write RoadScene to GeoPackage."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from roader.legacy.model import RoadScene


def write_gpkg(scene: RoadScene, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    layers: dict[str, gpd.GeoDataFrame] = {}

    if scene.carriageways:
        layers["carriageways"] = gpd.GeoDataFrame(
            [
                {
                    "id": c.id,
                    "highway": c.highway,
                    "layer": c.layer,
                    "deck_id": c.deck_id,
                    "interchange_id": c.interchange_id,
                    "z_mean_m": c.z_mean_m,
                    "half_width_m": c.attrs.get("half_width_m"),
                    "section": c.attrs.get("section"),
                    "osm_ids": ",".join(str(x) for x in c.osm_ids),
                    "geometry": c.polygon if c.polygon is not None else c.centerline,
                }
                for c in scene.carriageways
            ],
            crs=scene.crs,
        )
        layers["centerlines"] = gpd.GeoDataFrame(
            [
                {
                    "id": c.id,
                    "highway": c.highway,
                    "layer": c.layer,
                    "z_mean_m": c.z_mean_m,
                    "geometry": c.centerline,
                }
                for c in scene.carriageways
            ],
            crs=scene.crs,
        )
        solid_rows = [
            {
                "solid_id": face.solid_id,
                "role": face.role,
                "geometry": face.triangle,
            }
            for carriageway in scene.carriageways
            for face in carriageway.faces
        ]
        if solid_rows:
            layers["carriageway_solids"] = gpd.GeoDataFrame(solid_rows, crs=scene.crs)

    if scene.decks:
        layers["decks"] = gpd.GeoDataFrame(
            [
                {
                    "id": d.id,
                    "z_p95_m": d.z_p95_m,
                    "z_max_m": d.z_max_m,
                    "point_count": d.point_count,
                    "source": d.source,
                    "clearance_below_m": d.clearance_below_m,
                    "geometry": d.polygon,
                }
                for d in scene.decks
            ],
            crs=scene.crs,
        )

    if scene.voids:
        layers["voids"] = gpd.GeoDataFrame(
            [
                {
                    "id": v.id,
                    "deck_id": v.deck_id,
                    "z_floor_m": v.z_floor_m,
                    "z_ceiling_m": v.z_ceiling_m,
                    "clearance_m": v.clearance_m,
                    "kind": v.kind,
                    "geometry": v.polygon,
                }
                for v in scene.voids
            ],
            crs=scene.crs,
        )

    if scene.sidewalks:
        layers["sidewalks"] = gpd.GeoDataFrame(
            [
                {
                    "id": s.id,
                    "carriageway_id": s.carriageway_id,
                    "source": s.source,
                    "geometry": s.geometry,
                }
                for s in scene.sidewalks
            ],
            crs=scene.crs,
        )

    if scene.interchanges:
        # Pointless without geometry — store as attribute table via empty
        # multipoint from member centroids if available
        rows = []
        cw = {c.id: c for c in scene.carriageways}
        for ix in scene.interchanges:
            geoms = [cw[i].centerline for i in ix.carriageway_ids if i in cw]
            if not geoms:
                continue
            from shapely.ops import unary_union

            rows.append(
                {
                    "id": ix.id,
                    "n_links": len(ix.carriageway_ids),
                    "geometry": unary_union(geoms).convex_hull,
                }
            )
        if rows:
            layers["interchanges"] = gpd.GeoDataFrame(rows, crs=scene.crs)

    if scene.skips:
        skip_rows = []
        for skip in scene.skips:
            if skip.xy is None:
                continue
            skip_rows.append(
                {
                    "reason": skip.reason,
                    "highway": skip.highway,
                    "osm_ids": ",".join(str(x) for x in skip.osm_ids),
                    "geometry": Point(skip.xy),
                }
            )
        if skip_rows:
            layers["skips"] = gpd.GeoDataFrame(skip_rows, crs=scene.crs)

    if not layers:
        # Write an empty placeholder layer so the file exists
        empty = gpd.GeoDataFrame(
            {"id": pd.Series(dtype=str)},
            geometry=gpd.GeoSeries([], crs=scene.crs),
            crs=scene.crs,
        )
        empty.to_file(path, layer="empty", driver="GPKG")
        return path

    for name, gdf in layers.items():
        gdf.to_file(path, layer=name, driver="GPKG")
    return path
