"""Per-unit configuration and the FR-IN-2 unit record.

A unit config is a TOML file in configs/units/<site_id>.toml:

    [unit]
    site_id = "..."
    aoi_lonlat = [west, south, east, north]
    imagery_enabled = false          # D-009

    [lidar]
    path = "data/fixtures/<site_id>/lidar.laz"
    name = "<3DEP project>"
    year = 2021
    quality_level = "QL1"
    sha256 = "..."                   # pinned input (FR-PL-3)

    [vectors.osm]                    # one table per source (FR-IN-7)
    enabled = true
    path = "data/fixtures/<site_id>/osm.json"
    license = "ODbL-1.0"
    sha256 = "..."

Relative paths resolve against the repository root (two levels above
configs/units/), or ``root`` when given.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import roader
from roader.io.frame import aoi_anchor, unit_frame
from roader.model.schema import Frame, LidarCollection, UnitRecord, VectorSourceRecord


@dataclass
class LidarSourceConfig:
    path: Path
    name: str
    year: int | None = None
    quality_level: str | None = None
    sha256: str = ""


@dataclass
class VectorSourceConfig:
    name: str
    enabled: bool
    path: Path
    license: str
    sha256: str = ""


@dataclass
class UnitConfig:
    site_id: str
    aoi_lonlat: tuple[float, float, float, float]
    lidar: list[LidarSourceConfig]
    vectors: dict[str, VectorSourceConfig] = field(default_factory=dict)
    imagery_enabled: bool = False
    anchor_lonlat: tuple[float, float] | None = None

    def frame(self) -> Frame:
        lon, lat = self.anchor_lonlat or aoi_anchor(self.aoi_lonlat)
        return unit_frame(lon, lat)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_unit_config(path: str | Path, root: str | Path | None = None) -> UnitConfig:
    path = Path(path).resolve()
    base = Path(root) if root is not None else path.parents[2]
    data = tomllib.loads(path.read_text())
    unit = data["unit"]

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else base / q

    lidar_tables = data.get("lidar", [])
    if isinstance(lidar_tables, dict):
        lidar_tables = [lidar_tables]
    lidar = [
        LidarSourceConfig(resolve(t["path"]), t["name"], t.get("year"), t.get("quality_level"), t.get("sha256", ""))
        for t in lidar_tables
    ]
    vectors = {
        name: VectorSourceConfig(name, bool(t.get("enabled", True)), resolve(t["path"]), t["license"], t.get("sha256", ""))
        for name, t in sorted(data.get("vectors", {}).items())
    }
    anchor = unit.get("anchor_lonlat")
    return UnitConfig(
        site_id=unit["site_id"],
        aoi_lonlat=tuple(float(v) for v in unit["aoi_lonlat"]),
        lidar=lidar,
        vectors=vectors,
        imagery_enabled=bool(unit.get("imagery_enabled", False)),
        anchor_lonlat=tuple(float(v) for v in anchor) if anchor else None,
    )


def _checked_sha(path: Path, pinned: str, what: str) -> str:
    actual = sha256_file(path)
    if pinned and actual != pinned:
        raise ValueError(f"{what} {path} sha256 {actual} does not match pinned {pinned} (FR-PL-3)")
    return actual


def verify_inputs(config: UnitConfig) -> None:
    """Raise if any enabled input differs from its pinned hash (FR-PL-3)."""
    for src in config.lidar:
        _checked_sha(src.path, src.sha256, "lidar")
    for src in config.vectors.values():
        if src.enabled:
            _checked_sha(src.path, src.sha256, f"vector source {src.name}")


def build_unit_record(config: UnitConfig, lidar: list[LidarCollection]) -> UnitRecord:
    """FR-IN-2 record. Enabled vector files are hashed and checked against their pins."""
    vectors = []
    for src in config.vectors.values():
        sha = _checked_sha(src.path, src.sha256, f"vector source {src.name}") if src.enabled else src.sha256
        vectors.append(VectorSourceRecord(src.name, src.license, src.enabled, sha))
    return UnitRecord(
        unit_id=config.site_id,
        aoi_lonlat=config.aoi_lonlat,
        software_version=roader.__version__,
        lidar=list(lidar),
        vectors=vectors,
        imagery_enabled=config.imagery_enabled,
    )
