"""Offline unit tests — no LiDAR / OSM network required."""

from pathlib import Path

from shapely.geometry import LineString, box

from roader.legacy.export import write_gpkg
from roader.legacy.model import Carriageway, Deck, RoadScene, VoidUnder


def test_write_gpkg(tmp_path: Path) -> None:
    line = LineString([(0, 0), (100, 0)])
    scene = RoadScene(
        crs="EPSG:3857",
        carriageways=[
            Carriageway(
                id="cw_0",
                centerline=line,
                polygon=line.buffer(5),
                highway="residential",
                layer="terrain",
                z_mean_m=10.0,
            )
        ],
        decks=[
            Deck(
                id="deck_0",
                polygon=box(40, -10, 60, 10),
                z_p95_m=18.0,
                z_max_m=19.0,
                point_count=100,
                source="elevated_cluster",
                clearance_below_m=5.0,
            )
        ],
        voids=[
            VoidUnder(
                id="void_deck_0",
                polygon=box(40, -10, 60, 10),
                deck_id="deck_0",
                z_floor_m=13.0,
                z_ceiling_m=18.0,
                clearance_m=5.0,
                kind="terrain",
            )
        ],
    )
    out = tmp_path / "roads.gpkg"
    write_gpkg(scene, out)
    assert out.exists()
