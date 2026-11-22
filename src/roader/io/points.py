"""Point input contract (stage 2). Owned by lane `data`.

Every stage after io reads a PointSet in the unit frame, never a LAS file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
import shapely
from pyproj import CRS, Transformer

from roader.io.frame import LONLAT, FrameTransformer
from roader.model.schema import Frame, LidarCollection

NOISE_CLASSES = (7, 18)  # ASPRS low noise, high noise (FR-PN-1)
ASSUMED_VERTICAL = "NAVD88 (assumed; not declared in header)"


@dataclass
class PointSet:
    """Points in the unit frame (D-007), noise and withheld points already removed (FR-PN-1).

    All per-point arrays have the same length. ``classification`` is vendor
    evidence only (FR-IN-3); ``collection`` indexes UnitRecord.lidar.
    """

    xyz: np.ndarray  # (N, 3) float64
    classification: np.ndarray  # (N,) uint8
    intensity: np.ndarray  # (N,) float64, NaN when absent
    return_number: np.ndarray  # (N,) uint8
    number_of_returns: np.ndarray  # (N,) uint8
    scan_angle: np.ndarray  # (N,) float64 degrees, NaN when absent
    gps_time: np.ndarray  # (N,) float64, NaN when absent
    point_source_id: np.ndarray  # (N,) uint16, flight strip
    collection: np.ndarray  # (N,) uint8

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def subset(self, mask: np.ndarray) -> PointSet:
        return PointSet(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})


def _concat(parts: list[PointSet]) -> PointSet:
    names = PointSet.__dataclass_fields__
    if not parts:
        return empty_pointset()
    return PointSet(**{name: np.concatenate([getattr(p, name) for p in parts]) for name in names})


def empty_pointset() -> PointSet:
    return PointSet(
        xyz=np.zeros((0, 3)),
        classification=np.zeros(0, np.uint8),
        intensity=np.zeros(0),
        return_number=np.zeros(0, np.uint8),
        number_of_returns=np.zeros(0, np.uint8),
        scan_angle=np.zeros(0),
        gps_time=np.zeros(0),
        point_source_id=np.zeros(0, np.uint16),
        collection=np.zeros(0, np.uint8),
    )


# ---------------------------------------------------------------- reference systems


@dataclass
class _SourceCrs:
    horizontal: CRS
    horizontal_name: str
    vertical_name: str
    z_to_m: float


def _crs_name(crs: CRS) -> str:
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.name


def _source_crs(header_crs: CRS | None, override: str | None) -> _SourceCrs:
    """Horizontal and vertical reference systems of a file (FR-IN-2).

    An undeclared vertical CRS is recorded as NAVD88 (assumed), with z in the
    horizontal linear unit. A declared vertical CRS other than NAVD88 is refused:
    mixing datums silently would break the single-frame rule (FR-IN-2, D-007).
    """
    crs = CRS.from_user_input(override) if override else header_crs
    if crs is None:
        raise ValueError("point file declares no CRS; pass source_crs")
    horizontal, vertical = crs, None
    if crs.is_compound:
        horizontal, vertical = crs.sub_crs_list[0], crs.sub_crs_list[1]
    elif crs.is_vertical:
        raise ValueError(f"CRS {crs.name} has no horizontal part")
    if vertical is not None:
        if "NAVD88" not in vertical.name and vertical.to_epsg() not in (5703, 6360, 8228):
            raise ValueError(f"vertical CRS {vertical.name} is not NAVD88; conversion is not implemented")
        vertical_name = f"{vertical.name} ({_crs_name(vertical)})"
        return _SourceCrs(horizontal, _crs_name(horizontal), vertical_name, vertical.axis_info[0].unit_conversion_factor)
    z_to_m = horizontal.axis_info[0].unit_conversion_factor if horizontal.is_projected else 1.0
    vertical_name = ASSUMED_VERTICAL if z_to_m == 1.0 else f"{ASSUMED_VERTICAL}; z unit taken from horizontal ({z_to_m} m)"
    return _SourceCrs(horizontal, _crs_name(horizontal), vertical_name, z_to_m)


# ---------------------------------------------------------------- reading


def _aoi_polygon_unit(aoi_lonlat, frame: Frame) -> shapely.Polygon:
    west, south, east, north = aoi_lonlat
    n = 21
    t = np.linspace(0.0, 1.0, n)
    lon = np.concatenate([west + (east - west) * t, np.full(n, east), east - (east - west) * t, np.full(n, west)])
    lat = np.concatenate([np.full(n, south), south + (north - south) * t, np.full(n, north), north - (north - south) * t])
    x, y = FrameTransformer(frame, LONLAT).lonlat_to_unit(lon, lat)
    return shapely.Polygon(np.column_stack([x, y]))


def _dim(points, names: set[str], name: str, dtype, fill=np.nan) -> np.ndarray:
    if name in names:
        return np.asarray(getattr(points, name)).astype(dtype)
    return np.full(len(points), fill, dtype=dtype)


def _convert(points, names, src: _SourceCrs, to_unit: FrameTransformer, polygon, bbox, collection_index) -> PointSet:
    x = np.asarray(points.x, dtype=np.float64)
    y = np.asarray(points.y, dtype=np.float64)
    keep = (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
    cls = np.asarray(points.classification).astype(np.uint8)
    keep &= ~np.isin(cls, NOISE_CLASSES)
    keep &= ~np.asarray(points.withheld).astype(bool)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return empty_pointset()
    ux, uy = to_unit.to_unit(x[idx], y[idx])
    inside = shapely.contains_xy(polygon, ux, uy)
    idx, ux, uy = idx[inside], ux[inside], uy[inside]
    z = np.asarray(points.z, dtype=np.float64)[idx] * src.z_to_m
    if "scan_angle" in names:  # point formats 6+: int16 in 0.006 degree steps
        scan = np.asarray(points.scan_angle, dtype=np.float64)[idx] * 0.006
    elif "scan_angle_rank" in names:
        scan = np.asarray(points.scan_angle_rank, dtype=np.float64)[idx]
    else:
        scan = np.full(idx.size, np.nan)
    return PointSet(
        xyz=np.column_stack([ux, uy, z]),
        classification=cls[idx],
        intensity=_dim(points, names, "intensity", np.float64)[idx],
        return_number=_dim(points, names, "return_number", np.uint8, 0)[idx],
        number_of_returns=_dim(points, names, "number_of_returns", np.uint8, 0)[idx],
        scan_angle=scan,
        gps_time=_dim(points, names, "gps_time", np.float64)[idx],
        point_source_id=_dim(points, names, "point_source_id", np.uint16, 0)[idx],
        collection=np.full(idx.size, collection_index, dtype=np.uint8),
    )


def _is_copc(header: laspy.LasHeader) -> bool:
    return any(getattr(vlr, "user_id", "") == "copc" for vlr in header.vlrs)


def read_points(
    path_or_url: str | Path,
    aoi_lonlat: tuple[float, float, float, float] | list[float],
    frame: Frame,
    collection_index: int = 0,
    *,
    name: str | None = None,
    year: int | None = None,
    quality_level: str | None = None,
    source_crs: str | None = None,
    chunk_size: int = 1_000_000,
) -> tuple[PointSet, LidarCollection]:
    """Windowed read of one LAS / LAZ / COPC source into the unit frame (FR-IN-1, FR-PN-1).

    Drops withheld points and noise classes 7 and 18; keeps every other class
    (FR-IN-3). Horizontal coordinates go to the unit frame; heights stay NAVD88
    in meters. URLs must point at COPC files (HTTP range reads).
    """
    source = str(path_or_url)
    remote = source.startswith(("http://", "https://"))
    if remote:
        reader = laspy.CopcReader.open(source)
        copc = True
        header = reader.header
    else:
        with laspy.open(source) as probe:
            header = probe.header
        copc = _is_copc(header)
        reader = laspy.CopcReader.open(source) if copc else laspy.open(source)

    src = _source_crs(header.parse_crs(), source_crs)
    to_unit = FrameTransformer(frame, src.horizontal)
    west, south, east, north = aoi_lonlat
    bbox = Transformer.from_crs(LONLAT, src.horizontal, always_xy=True).transform_bounds(
        west, south, east, north, densify_pts=21
    )
    polygon = _aoi_polygon_unit(aoi_lonlat, frame)
    shapely.prepare(polygon)
    names = set(header.point_format.dimension_names)

    parts: list[PointSet] = []
    with reader:
        if copc:
            pad = 1e-6
            bounds = laspy.Bounds(
                mins=np.array([bbox[0] - pad, bbox[1] - pad]), maxs=np.array([bbox[2] + pad, bbox[3] + pad])
            )
            parts.append(_convert(reader.query(bounds=bounds), names, src, to_unit, polygon, bbox, collection_index))
        else:
            for chunk in reader.chunk_iterator(chunk_size):
                parts.append(_convert(chunk, names, src, to_unit, polygon, bbox, collection_index))

    pts = _concat(parts)
    record = LidarCollection(
        name=name or Path(source.split("?")[0]).name,
        year=year,
        quality_level=quality_level,
        density_pavement_pts_m2=None,
        horizontal_crs=src.horizontal_name,
        vertical_crs=src.vertical_name,
        point_count=len(pts),
    )
    return pts, record


# ---------------------------------------------------------------- synthetic


def from_synthetic(points, collection_index: int = 0, drop_noise: bool = True) -> PointSet:
    """PointSet from roader.model.synthetic.SyntheticPoints, already in the unit frame.

    Noise classes are dropped as read_points does; to align ``points.truth`` use
    ``points.truth[~np.isin(points.classification, NOISE_CLASSES)]``.
    """
    n = int(points.xyz.shape[0])
    pts = PointSet(
        xyz=np.asarray(points.xyz, dtype=np.float64),
        classification=np.asarray(points.classification, dtype=np.uint8),
        intensity=np.asarray(points.intensity, dtype=np.float64),
        return_number=np.asarray(points.return_number, dtype=np.uint8),
        number_of_returns=np.ones(n, dtype=np.uint8),
        scan_angle=np.full(n, np.nan),
        gps_time=np.full(n, np.nan),
        point_source_id=np.asarray(points.point_source_id, dtype=np.uint16),
        collection=np.full(n, collection_index, dtype=np.uint8),
    )
    return pts.subset(~np.isin(pts.classification, NOISE_CLASSES)) if drop_noise else pts
