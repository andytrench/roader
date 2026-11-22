"""Unit working frame (D-007, FR-DM-5, FR-IN-2).

Transverse Mercator on GRS80 centered on the unit anchor, scale 1, no false
origin, heights NAVD88 orthometric. Over a 10 km unit the scale error stays
under 1e-5 (k - 1 is about x^2 / 2R^2, 3e-7 at 5 km).

Inputs in WGS84-based CRSs (EPSG:3857 EPTs, OSM lon/lat) and NAD83-based CRSs
are both treated as nominal geographic coordinates: PROJ applies no datum shift
to a frame defined only by its ellipsoid. All sources of one unit go through the
same path, so they stay registered to each other.
"""

from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

from roader.model.schema import Frame

LONLAT = "EPSG:4326"


def frame_proj(anchor_lon: float, anchor_lat: float) -> str:
    return (
        f"+proj=tmerc +lat_0={anchor_lat!r} +lon_0={anchor_lon!r} +k=1 +x_0=0 +y_0=0 "
        "+ellps=GRS80 +units=m +no_defs"
    )


def unit_frame(anchor_lon: float, anchor_lat: float, source_crs: str = "") -> Frame:
    """The D-007 frame for a unit anchored at (anchor_lon, anchor_lat)."""
    return Frame(
        proj=frame_proj(float(anchor_lon), float(anchor_lat)),
        anchor_lon=float(anchor_lon),
        anchor_lat=float(anchor_lat),
        vertical_datum="NAVD88",
        geoid_model="GEOID18",
        source_crs=source_crs,
    )


def aoi_anchor(aoi_lonlat: tuple[float, float, float, float] | list[float]) -> tuple[float, float]:
    """Center of a [west, south, east, north] box, rounded to 1e-6 degree so anchors are stable."""
    west, south, east, north = aoi_lonlat
    return round((west + east) / 2.0, 6), round((south + north) / 2.0, 6)


class FrameTransformer:
    """Horizontal conversion between a source CRS, the unit frame and lon/lat.

    ``source_crs`` is anything pyproj accepts (``"EPSG:3857"``, a WKT string, a
    CRS). For a compound CRS only the horizontal part is used; heights pass
    through unchanged.
    """

    def __init__(self, frame: Frame, source_crs: str | CRS = LONLAT) -> None:
        unit = CRS.from_proj4(frame.proj)
        src = CRS.from_user_input(source_crs)
        if src.is_compound:
            src = src.sub_crs_list[0]
        self.frame = frame
        self.source_crs = src
        self._to_unit = Transformer.from_crs(src, unit, always_xy=True)
        self._from_unit = Transformer.from_crs(unit, src, always_xy=True)
        self._lonlat_to_unit = Transformer.from_crs(LONLAT, unit, always_xy=True)
        self._unit_to_lonlat = Transformer.from_crs(unit, LONLAT, always_xy=True)

    def to_unit(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        ux, uy = self._to_unit.transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return np.asarray(ux), np.asarray(uy)

    def from_unit(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        sx, sy = self._from_unit.transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return np.asarray(sx), np.asarray(sy)

    def lonlat_to_unit(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        ux, uy = self._lonlat_to_unit.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return np.asarray(ux), np.asarray(uy)

    def unit_to_lonlat(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._unit_to_lonlat.transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return np.asarray(lon), np.asarray(lat)


def lonlat_transformer(frame: Frame) -> FrameTransformer:
    return FrameTransformer(frame, LONLAT)
