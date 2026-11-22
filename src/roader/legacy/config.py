"""TOML config loading."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class AreaConfig:
    """One city run. Paths may be absolute. Relative paths are from the repo root."""

    name: str
    label: str
    tiles: Path
    crs: str
    out: Path
    work: Path
    cache: Path | None = None
    collections: str = ""
    group: int = 1
    force: bool = False


def load_area_config(path: Path) -> AreaConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    area = raw["area"]
    root = path.resolve().parents[1]

    def resolve(value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (root / candidate).resolve()

    cache = area.get("cache")
    return AreaConfig(
        name=str(area["name"]),
        label=str(area.get("label", area["name"])),
        tiles=resolve(str(area["tiles"])),
        crs=str(area["crs"]),
        out=resolve(str(area["out"])),
        work=resolve(str(area.get("work", "data/areas"))),
        cache=resolve(str(cache)) if cache else None,
        collections=str(area.get("collections", "")),
        group=int(area.get("group", 1)),
        force=bool(area.get("force", False)),
    )


@dataclass
class PilotConfig:
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north (WGS84)
    crs: str
    lidar_path: Path
    naip_path: Path | None
    output_gpkg: Path
    osm_network_type: str = "all"
    deck_height_above_ground_m: float = 2.5
    sidewalk_infer: bool = True
    pavement_thickness_m: float = 0.2
    deck_thickness_m: float = 0.8


def load_pilot_config(path: Path) -> PilotConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    pilot = raw["pilot"]
    bbox = tuple(float(x) for x in pilot["bbox"])
    if len(bbox) != 4:
        raise ValueError("pilot.bbox must be [west, south, east, north]")
    root = path.parent.parent
    lidar = Path(pilot["lidar_path"])
    naip = Path(pilot["naip_path"]) if pilot.get("naip_path") else None
    out = Path(pilot["output_gpkg"])
    if not lidar.is_absolute():
        lidar = (root / lidar).resolve()
    if naip is not None and not naip.is_absolute():
        naip = (root / naip).resolve()
    if not out.is_absolute():
        out = (root / out).resolve()
    return PilotConfig(
        name=str(pilot.get("name", path.stem)),
        bbox=bbox,  # type: ignore[arg-type]
        crs=str(pilot.get("crs", "EPSG:3857")),
        lidar_path=lidar,
        naip_path=naip,
        output_gpkg=out,
        osm_network_type=str(pilot.get("osm_network_type", "all")),
        deck_height_above_ground_m=float(
            pilot.get("deck_height_above_ground_m", 2.5)
        ),
        sidewalk_infer=bool(pilot.get("sidewalk_infer", True)),
        pavement_thickness_m=float(pilot.get("pavement_thickness_m", 0.2)),
        deck_thickness_m=float(pilot.get("deck_thickness_m", 0.8)),
    )
