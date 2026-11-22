"""Corridor sampling, grade-limited profiles, and closed carriageway prisms.

Intensity and NAIP NDVI are pavement-versus-vegetation cues. The geometry is
the class corridor, the low LiDAR surface, and the road-design limits.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.validation import make_valid

# A hole shorter than this is still one road. Longer than GAP_FAIL_M is a skip.
GAP_FILL_M = 18.0
# An interior hole longer than this is a skip, not an invented surface.
GAP_FAIL_M = 25.0
LOW_PERCENTILE = 15.0
RESIDUAL_M = 0.20
# A standard section may leave the accepted stations by this much. Beyond it, the station stays on the LiDAR.
PROFILE_TIE_M = 0.12
CROWN = 0.02
SE_MAX = 0.06
# Straighter than this stays in crown. Tighter than this takes full superelevation.
R_TANGENT_M = 600.0
R_FULL_M = 70.0
RUNOFF_M = 45.0
MAX_CROSS_SLOPE = 0.08
NDVI_MAX = 0.35
INTENSITY_KEEP_PCT = 65.0
DEAD_END_M = 40.0
DEAD_END_TOL_M = 8.0
CORRIDOR_MARGIN_M = 3.0
STATION_STEP_M = 5.0
# A short elevated stack sitting on continuous ground is the road above, not this one.
MIN_DECK_RUN_M = 18.0
# A bank whose toe is at least this far below the pavement is fill, unless the edge has already dropped.
FILL_RISE_M = 1.0
# The lip is still the road when it sits within this of the pavement.
LIP_CLOSE_M = 1.2
# A deeper drop on an attached lip is a missed span or a hole, not an embankment.
FILL_MAX_M = 15.0
BANK_LIP_INSET_M = 0.5
BANK_LIP_WIDTH_M = 2.5
BANK_TOE_INSET_M = 8.0
BANK_TOE_REACH_M = 18.0

CLASS_HALF_WIDTH_M = {
    "motorway": 8.0,
    "motorway_link": 5.0,
    "trunk": 7.0,
    "trunk_link": 5.0,
    "primary": 6.0,
    "primary_link": 5.0,
    "secondary": 5.5,
    "secondary_link": 5.0,
    "tertiary": 5.0,
    "tertiary_link": 4.5,
    "residential": 4.0,
    "unclassified": 3.5,
    "service": 3.0,
}

NdviFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Cloud:
    xy: np.ndarray
    z: np.ndarray
    intensity: np.ndarray | None
    tree: cKDTree | None

    @classmethod
    def from_arrays(
        cls,
        xy: np.ndarray,
        z: np.ndarray,
        intensity: np.ndarray | None = None,
    ) -> Cloud:
        xy_arr = np.asarray(xy, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        if xy_arr.size == 0:
            return cls(np.zeros((0, 2)), np.zeros((0,)), None, None)
        if xy_arr.ndim == 1:
            xy_arr = xy_arr.reshape(-1, 2)
        inten = None
        if intensity is not None:
            inten = np.asarray(intensity, dtype=float)
            if len(inten) != len(z_arr):
                inten = None
        return cls(xy_arr, z_arr, inten, cKDTree(xy_arr))


@dataclass
class StationFrame:
    s: np.ndarray
    xy: np.ndarray
    tangent: np.ndarray


@dataclass
class StationArray:
    z: np.ndarray
    slope: np.ndarray
    crown: np.ndarray
    template: np.ndarray
    measured_half: np.ndarray
    accepted: np.ndarray
    point_count: np.ndarray


@dataclass
class BankSample:
    """Ground just outside the pavement (lip) and at the toe, each side."""

    left_lip: np.ndarray
    right_lip: np.ndarray
    left_toe: np.ndarray
    right_toe: np.ndarray


@dataclass
class RawFace:
    role: str
    coords: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass
class DeckStrip:
    polygon: Polygon | MultiPolygon
    z_p95: float
    z_max: float
    point_count: int


@dataclass
class FittedSurface:
    reason: str | None = None
    polygon: Polygon | MultiPolygon | None = None
    raw_faces: list[RawFace] = field(default_factory=list)
    centerline: LineString | None = None
    z_mean: float | None = None
    half_width_m: float | None = None
    profile_xy: list[tuple[float, float]] = field(default_factory=list)
    profile_z: list[float] = field(default_factory=list)
    profile_ground_z: list[float] = field(default_factory=list)
    profile_deck: list[bool] = field(default_factory=list)
    deck_parts: list[DeckStrip] = field(default_factory=list)
    section: str = "lidar"


def class_half_width(highway: str) -> float:
    return CLASS_HALF_WIDTH_M.get(highway, 4.5)


def class_half_limits(highway: str) -> tuple[float, float]:
    nominal = class_half_width(highway)
    return max(2.0, nominal * 0.5), nominal


def is_parking_bleed(
    line: LineString,
    highway: str,
    measured_max: float,
    max_half: float,
) -> bool:
    """A short service way whose pavement is far wider than its class is a lot.

    Ordinary road edges are often shorter than 40 m because OSM splits them at
    intersections. Those stay, and the width clamp keeps the solid in the class.
    """
    if highway != "service" or not np.isfinite(measured_max):
        return False
    if float(line.length) >= DEAD_END_M:
        return False
    if measured_max <= max_half * 1.25:
        return False
    width = 2.0 * float(measured_max)
    return width > 0.0 and float(line.length) / width < 3.0


def max_grade_for(highway: str) -> float:
    if "link" in highway:
        return 0.08
    if highway in {"motorway", "trunk", "primary"}:
        return 0.06
    return 0.12


def tag_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return tag_str(val[0]) if len(val) else ""
    if isinstance(val, (float, np.floating)) and math.isnan(float(val)):
        return ""
    text = str(val).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def tag_skip_reason(line: LineString, highway: str, attrs: dict[str, Any]) -> str | None:
    if line is None or line.is_empty or float(line.length) < 1.0:
        return "too_short"
    service = tag_str(attrs.get("service")).lower()
    if highway == "service" and service == "driveway":
        return "driveway"
    return None


def dead_end_reason(
    line: LineString,
    highway: str,
    others: list[LineString],
) -> str | None:
    """A short spur that meets the network at exactly one end is a driveway."""
    if "link" in highway or float(line.length) >= DEAD_END_M:
        return None
    if _one_end_connected(line, others):
        return "driveway"
    return None


def _one_end_connected(line: LineString, others: list[LineString]) -> bool:
    if not others:
        return False
    start = Point(line.coords[0][:2])
    end = Point(line.coords[-1][:2])
    hits_start = any(other.distance(start) <= DEAD_END_TOL_M for other in others)
    hits_end = any(other.distance(end) <= DEAD_END_TOL_M for other in others)
    return hits_start != hits_end


def station_frame(line: LineString, step: float = STATION_STEP_M) -> StationFrame:
    length = float(line.length)
    count = max(2, int(length / step) + 1)
    distances = np.linspace(0.0, length, count)
    xy = np.empty((count, 2), dtype=float)
    for i, dist in enumerate(distances):
        coord = line.interpolate(float(dist)).coords[0]
        xy[i, 0] = coord[0]
        xy[i, 1] = coord[1]
    tangent = np.zeros((count, 2), dtype=float)
    for i in range(count):
        if i == 0:
            vec = xy[1] - xy[0]
        elif i == count - 1:
            vec = xy[-1] - xy[-2]
        else:
            vec = xy[i + 1] - xy[i - 1]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            tangent[i] = tangent[i - 1] if i else np.array([1.0, 0.0])
        else:
            tangent[i] = vec / norm
    return StationFrame(s=distances, xy=xy, tangent=tangent)


def true_runs(mask: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    flags = list(mask) + [False]
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def hole_lengths(mask: np.ndarray, distances: np.ndarray) -> list[float]:
    idx = np.flatnonzero(mask)
    holes: list[float] = []
    for left, right in itertools.pairwise(idx):
        if right > left + 1:
            holes.append(float(distances[right] - distances[left]))
    return holes


def boundary_edge_count(faces: list[RawFace]) -> int:
    """Count edges that are not shared by exactly two triangles."""
    counts: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
    for face in faces:
        coords = face.coords
        for start, end in zip(coords, coords[1:] + coords[:1], strict=True):
            edge_a = tuple(round(float(v), 4) for v in start)
            edge_b = tuple(round(float(v), 4) for v in end)
            if edge_a == edge_b:
                continue
            key = (edge_a, edge_b) if edge_a <= edge_b else (edge_b, edge_a)
            counts[key] = counts.get(key, 0) + 1
    return sum(1 for count in counts.values() if count != 2)


def sample_profile(
    frame: StationFrame,
    cloud: Cloud,
    *,
    search_half_m: float,
    max_grade: float,
    station_step_m: float = STATION_STEP_M,
    lateral_min_m: float = 0.0,
    section_half_m: float | None = None,
    as_center: bool = True,
    ndvi: NdviFn | None = None,
) -> StationArray:
    count = len(frame.s)
    stations = StationArray(
        z=np.full(count, np.nan),
        slope=np.zeros(count),
        crown=np.zeros(count),
        template=np.zeros(count, dtype=bool),
        measured_half=np.full(count, np.nan),
        accepted=np.zeros(count, dtype=bool),
        point_count=np.zeros(count, dtype=int),
    )
    if cloud.tree is None or len(cloud.xy) == 0:
        return stations

    radius = math.hypot(station_step_m * 0.55, search_half_m)
    along_limit = station_step_m * 0.55
    bins_lat: list[np.ndarray | None] = [None] * count
    bins_z: list[np.ndarray | None] = [None] * count
    lidar_z = np.full(count, np.nan)
    lidar_slope = np.zeros(count)
    for i in range(count):
        origin = frame.xy[i]
        tangent = frame.tangent[i]
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        found = cloud.tree.query_ball_point((float(origin[0]), float(origin[1])), radius)
        if not found:
            continue
        idx = np.asarray(found, dtype=int)
        delta = cloud.xy[idx] - origin
        along = delta @ tangent
        lateral = delta @ normal
        keep = (
            (np.abs(along) <= along_limit)
            & (np.abs(lateral) <= search_half_m)
            & (np.abs(lateral) >= lateral_min_m)
        )
        if not np.any(keep):
            continue
        idx = idx[keep]
        lateral = lateral[keep]
        zz = cloud.z[idx]
        xy = cloud.xy[idx]
        inten = cloud.intensity[idx] if cloud.intensity is not None else None
        cue = _intensity_mask(inten, len(idx)) & _ndvi_mask(xy, ndvi)
        if not np.any(cue):
            continue
        if as_center:
            fitted = _cross_section(lateral[cue], zz[cue], search_half_m)
            if fitted is None:
                continue
            z0, slope, measured, lat_bins, z_bins = fitted
            bins_lat[i] = lat_bins
            bins_z[i] = z_bins
            lidar_z[i] = z0
            lidar_slope[i] = slope
        else:
            kept_z = zz[cue]
            kept_lat = lateral[cue]
            if len(kept_z) == 0:
                continue
            z0 = float(np.percentile(kept_z, LOW_PERCENTILE))
            slope = 0.0
            measured = float(np.max(np.abs(kept_lat)))
        stations.z[i] = z0
        stations.slope[i] = slope
        stations.measured_half[i] = measured
        stations.accepted[i] = True
        stations.point_count[i] = int(np.count_nonzero(cue))

    if as_center:
        _apply_standard_section(
            stations,
            frame,
            bins_lat,
            bins_z,
            lidar_z,
            lidar_slope,
            section_half_m if section_half_m is not None else search_half_m,
        )
    _fill_short_gaps(stations, frame.s, GAP_FILL_M)
    if as_center:
        stations.z = _smooth_tied(stations.z, frame.s, stations.accepted, max_grade)
    return stations


def sample_banks(frame: StationFrame, cloud: Cloud, half: float) -> BankSample:
    """Low ground in the lip ring and the toe ring on each side of the carriageway.

    The lip is the first ground outside the pavement. The toe is the ground
    the embankment would sit on. Class 2 is already the ground, so intensity
    and NDVI are not applied.
    """
    count = len(frame.s)
    empty = BankSample(
        left_lip=np.full(count, np.nan),
        right_lip=np.full(count, np.nan),
        left_toe=np.full(count, np.nan),
        right_toe=np.full(count, np.nan),
    )
    if cloud.tree is None or len(cloud.xy) == 0:
        return empty
    reach = half + BANK_TOE_REACH_M
    radius = math.hypot(STATION_STEP_M * 0.55, reach)
    along_limit = STATION_STEP_M * 0.55
    lip_0 = half + BANK_LIP_INSET_M
    lip_1 = lip_0 + BANK_LIP_WIDTH_M
    toe_0 = half + BANK_TOE_INSET_M
    toe_1 = reach
    for i in range(count):
        origin = frame.xy[i]
        tangent = frame.tangent[i]
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        found = cloud.tree.query_ball_point((float(origin[0]), float(origin[1])), radius)
        if not found:
            continue
        idx = np.asarray(found, dtype=int)
        delta = cloud.xy[idx] - origin
        along = delta @ tangent
        lateral = delta @ normal
        keep = (np.abs(along) <= along_limit) & (np.abs(lateral) <= reach)
        if not np.any(keep):
            continue
        lateral = lateral[keep]
        zz = cloud.z[idx][keep]
        empty.left_lip[i] = _band_z(lateral, zz, lip_0, lip_1, positive=True)
        empty.right_lip[i] = _band_z(lateral, zz, lip_0, lip_1, positive=False)
        empty.left_toe[i] = _band_z(lateral, zz, toe_0, toe_1, positive=True)
        empty.right_toe[i] = _band_z(lateral, zz, toe_0, toe_1, positive=False)
    return empty


def _band_z(
    lateral: np.ndarray,
    z: np.ndarray,
    inner: float,
    outer: float,
    *,
    positive: bool,
) -> float:
    side = lateral >= 0.0 if positive else lateral <= 0.0
    chosen = side & (np.abs(lateral) >= inner) & (np.abs(lateral) <= outer)
    if int(np.count_nonzero(chosen)) < 3:
        return float("nan")
    return float(np.percentile(z[chosen], LOW_PERCENTILE))


def build_prism(
    xy: np.ndarray,
    tangent: np.ndarray,
    z: np.ndarray,
    slope: np.ndarray,
    crown: np.ndarray,
    half: float,
    thickness: np.ndarray,
    flare_pos: np.ndarray | None = None,
    flare_neg: np.ndarray | None = None,
) -> tuple[list[RawFace], Polygon | MultiPolygon] | None:
    """Closed prism. Fill stations flare the bottom out to the toe; the top stays at the lanes."""
    count = len(z)
    if count < 2 or half < 0.5:
        return None
    edges = _offset_edges(xy, tangent, half)
    if edges is None:
        return None
    neg_xy, pos_xy = edges
    pos_flare = np.zeros(count) if flare_pos is None else np.asarray(flare_pos, dtype=float)
    neg_flare = np.zeros(count) if flare_neg is None else np.asarray(flare_neg, dtype=float)
    neg_top: list[tuple[float, float, float]] = []
    pos_top: list[tuple[float, float, float]] = []
    mid_top: list[tuple[float, float, float]] = []
    neg_bot: list[tuple[float, float, float]] = []
    pos_bot: list[tuple[float, float, float]] = []
    mid_bot: list[tuple[float, float, float]] = []
    for i in range(count):
        # Positive lateral is left of the direction of travel. Positive slope raises that side.
        normal = np.array([-tangent[i, 1], tangent[i, 0]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        normal /= norm
        drop = float(crown[i]) * half
        z_neg = float(z[i] - drop - slope[i] * half)
        z_pos = float(z[i] - drop + slope[i] * half)
        thick = float(thickness[i])
        neg_top.append((float(neg_xy[i][0]), float(neg_xy[i][1]), z_neg))
        pos_top.append((float(pos_xy[i][0]), float(pos_xy[i][1]), z_pos))
        mid_top.append((float(xy[i][0]), float(xy[i][1]), float(z[i])))
        neg_base = xy[i] - normal * (half + max(0.0, float(neg_flare[i])))
        pos_base = xy[i] + normal * (half + max(0.0, float(pos_flare[i])))
        neg_bot.append((float(neg_base[0]), float(neg_base[1]), z_neg - thick))
        pos_bot.append((float(pos_base[0]), float(pos_base[1]), z_pos - thick))
        mid_bot.append((mid_top[-1][0], mid_top[-1][1], float(z[i]) - thick))

    faces: list[RawFace] = []

    def add(role: str, a, b, c) -> None:
        faces.append(RawFace(role, (a, b, c)))

    for i in range(count - 1):
        add("top", neg_top[i], mid_top[i], mid_top[i + 1])
        add("top", neg_top[i], mid_top[i + 1], neg_top[i + 1])
        add("top", mid_top[i], pos_top[i], pos_top[i + 1])
        add("top", mid_top[i], pos_top[i + 1], mid_top[i + 1])
        add("bottom", neg_bot[i], mid_bot[i + 1], mid_bot[i])
        add("bottom", neg_bot[i], neg_bot[i + 1], mid_bot[i + 1])
        add("bottom", mid_bot[i], pos_bot[i + 1], pos_bot[i])
        add("bottom", mid_bot[i], mid_bot[i + 1], pos_bot[i + 1])
        add("wall", neg_top[i], neg_top[i + 1], neg_bot[i + 1])
        add("wall", neg_top[i], neg_bot[i + 1], neg_bot[i])
        add("wall", pos_top[i], pos_bot[i], pos_bot[i + 1])
        add("wall", pos_top[i], pos_bot[i + 1], pos_top[i + 1])
    add("wall", neg_top[0], neg_bot[0], mid_bot[0])
    add("wall", neg_top[0], mid_bot[0], mid_top[0])
    add("wall", mid_top[0], mid_bot[0], pos_bot[0])
    add("wall", mid_top[0], pos_bot[0], pos_top[0])
    add("wall", neg_top[-1], mid_top[-1], mid_bot[-1])
    add("wall", neg_top[-1], mid_bot[-1], neg_bot[-1])
    add("wall", mid_top[-1], pos_top[-1], pos_bot[-1])
    add("wall", mid_top[-1], pos_bot[-1], mid_bot[-1])

    polygon = footprint_polygon(xy, tangent, half)
    if polygon is None:
        return None
    return faces, polygon


def footprint_polygon(
    xy: np.ndarray,
    tangent: np.ndarray,
    half: float,
) -> Polygon | MultiPolygon | None:
    edges = _offset_edges(xy, tangent, half)
    if edges is None:
        return None
    neg_xy, pos_xy = edges
    ring = [(float(p[0]), float(p[1])) for p in pos_xy]
    ring.extend((float(p[0]), float(p[1])) for p in reversed(neg_xy))
    if len(ring) < 3:
        return None
    ring.append(ring[0])
    geom = Polygon(ring)
    if not geom.is_valid:
        geom = make_valid(geom)
    return _largest_poly(geom)


class PavementSampler:
    """Vegetation score from a 3-band chip. High means green canopy, as NDVI does.

    Green excess is green minus the average of red and blue. Twice that ratio
    sits above the NDVI gate for canopy and near zero for grey pavement.
    """

    def __init__(self, dataset: Any) -> None:
        self._ds = dataset

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        samples = _sample_bands(self._ds, xy)
        if samples is None:
            return np.zeros((len(xy),))
        red = samples[:, 0]
        green = samples[:, 1]
        blue = samples[:, 2]
        denom = red + green + blue
        out = np.zeros(len(samples), dtype=float)
        ok = np.isfinite(denom) & (denom > 1.0)
        excess = green[ok] - 0.5 * (red[ok] + blue[ok])
        out[ok] = 2.0 * excess / denom[ok]
        return out

    def close(self) -> None:
        self._ds.close()


class NdviSampler:
    """Sample (NIR - red) / (NIR + red) from a 4-band NAIP chip.

    Band 1 is red and band 4 is near infrared, matching NAIP rgbir order.
    """

    def __init__(self, dataset: Any) -> None:
        self._ds = dataset

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        samples = _sample_bands(self._ds, xy)
        if samples is None or samples.shape[1] < 4:
            return np.zeros((0 if samples is None else len(samples),))
        red = samples[:, 0]
        nir = samples[:, 3]
        out = np.full(len(samples), np.nan)
        denom = nir + red
        ok = np.isfinite(denom) & (np.abs(denom) > 1e-6)
        out[ok] = (nir[ok] - red[ok]) / denom[ok]
        return out

    def close(self) -> None:
        self._ds.close()


def open_ndvi_sampler(path: Path | None) -> NdviSampler | PavementSampler | None:
    """Four bands use NDVI. Three bands use green excess. Both reject canopy."""
    if path is None or not Path(path).exists():
        return None
    import rasterio

    dataset = rasterio.open(path)
    if dataset.count >= 4:
        return NdviSampler(dataset)
    if dataset.count >= 3:
        return PavementSampler(dataset)
    dataset.close()
    return None


def _sample_bands(dataset: Any, xy: np.ndarray) -> np.ndarray | None:
    xy_arr = np.asarray(xy, dtype=float)
    if len(xy_arr) == 0:
        return None
    samples = np.array(
        [np.asarray(value, dtype=float) for value in dataset.sample(xy_arr)],
        dtype=float,
    )
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    return samples


def _intensity_mask(intensity: np.ndarray | None, count: int) -> np.ndarray:
    """Keep the darker returns. Missing or flat intensity keeps every point."""
    if intensity is None or len(intensity) < 8 or float(np.std(intensity)) < 1.0:
        return np.ones(count, dtype=bool)
    limit = float(np.percentile(intensity, INTENSITY_KEEP_PCT))
    keep = intensity <= limit
    if int(np.count_nonzero(keep)) < 3:
        return np.ones(count, dtype=bool)
    return keep


def _ndvi_mask(xy: np.ndarray, ndvi: NdviFn | None) -> np.ndarray:
    if ndvi is None or len(xy) == 0:
        return np.ones(len(xy), dtype=bool)
    values = np.asarray(ndvi(xy), dtype=float)
    if len(values) != len(xy) or not np.any(np.isfinite(values)):
        return np.ones(len(xy), dtype=bool)
    return ~np.isfinite(values) | (values <= NDVI_MAX)


def _cross_section(
    lateral: np.ndarray,
    z: np.ndarray,
    search_half: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray] | None:
    if len(z) == 0:
        return None
    # A cut or a river just outside the lanes sits far below the pavement.
    # Leave those points to the bank test. They do not set the road surface.
    inner = np.abs(lateral) <= 2.5
    if int(np.count_nonzero(inner)) >= 4:
        deck = float(np.percentile(z[inner], LOW_PERCENTILE))
        keep = z >= deck - 1.0
        if int(np.count_nonzero(keep)) >= 4:
            lateral = lateral[keep]
            z = z[keep]
    bin_count = max(3, round(search_half * 2.0))
    edges = np.linspace(-search_half, search_half, bin_count + 1)
    centers: list[float] = []
    values: list[float] = []
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        if i == bin_count - 1:
            chosen = (lateral >= lo) & (lateral <= hi)
        else:
            chosen = (lateral >= lo) & (lateral < hi)
        if not np.any(chosen):
            continue
        centers.append(float((lo + hi) * 0.5))
        values.append(float(np.percentile(z[chosen], LOW_PERCENTILE)))
    if not centers:
        return None
    center_arr = np.asarray(centers, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    if len(value_arr) == 1:
        slope = 0.0
        z0 = float(value_arr[0])
    else:
        slope_fit, z0_fit = np.polyfit(center_arr, value_arr, 1)
        slope = float(slope_fit)
        z0 = float(z0_fit)
        # Cars and canopy sit above the pavement plane. A high edge of a cross slope does not.
        above = value_arr - (z0 + slope * center_arr)
        keep = above <= RESIDUAL_M
        if int(np.count_nonzero(keep)) >= 2 and int(np.count_nonzero(keep)) < len(value_arr):
            center_arr = center_arr[keep]
            value_arr = value_arr[keep]
            slope_fit, z0_fit = np.polyfit(center_arr, value_arr, 1)
            slope = float(slope_fit)
            z0 = float(z0_fit)
    slope = float(np.clip(slope, -MAX_CROSS_SLOPE, MAX_CROSS_SLOPE))
    measured = float(np.max(np.abs(center_arr)))
    return z0, slope, measured, center_arr, value_arr


def _fill_short_gaps(stations: StationArray, distances: np.ndarray, max_bracket: float) -> None:
    known = np.flatnonzero(stations.accepted)
    if len(known) < 2:
        return
    interp = PchipInterpolator(distances[known], stations.z[known])
    for left, right in itertools.pairwise(known):
        if right <= left + 1:
            continue
        if float(distances[right] - distances[left]) > max_bracket:
            continue
        for index in range(left + 1, right):
            stations.z[index] = float(interp(distances[index]))
            stations.slope[index] = stations.slope[left]
            stations.crown[index] = stations.crown[left]
            stations.template[index] = stations.template[left]
            stations.measured_half[index] = stations.measured_half[left]
            stations.accepted[index] = True
            stations.point_count[index] = 0


def _apply_standard_section(
    stations: StationArray,
    frame: StationFrame,
    bins_lat: list[np.ndarray | None],
    bins_z: list[np.ndarray | None],
    lidar_z: np.ndarray,
    lidar_slope: np.ndarray,
    section_half: float,
) -> None:
    """Use a crowned or superelevated section where it stays within the LiDAR bins."""
    tilt = _target_tilt(frame)
    for i in np.flatnonzero(stations.accepted):
        lat = bins_lat[i]
        val = bins_z[i]
        if lat is None or val is None or len(val) == 0:
            stations.slope[i] = lidar_slope[i]
            stations.crown[i] = 0.0
            stations.template[i] = False
            continue
        z_center = _center_elevation(lat, val)
        if _template_fits(lat, val, z_center, float(tilt[i]), section_half):
            stations.z[i] = z_center
            stations.slope[i] = float(tilt[i])
            stations.crown[i] = max(0.0, CROWN - abs(float(tilt[i])))
            stations.template[i] = True
        else:
            stations.z[i] = lidar_z[i]
            stations.slope[i] = lidar_slope[i]
            stations.crown[i] = 0.0
            stations.template[i] = False


def _center_elevation(lateral: np.ndarray, values: np.ndarray) -> float:
    inner = np.abs(lateral) <= 1.25
    if int(np.count_nonzero(inner)) >= 1:
        return float(np.percentile(values[inner], LOW_PERCENTILE))
    return float(np.percentile(values, LOW_PERCENTILE))


def _template_fits(
    lateral: np.ndarray,
    values: np.ndarray,
    z_center: float,
    tilt: float,
    section_half: float,
) -> bool:
    keep = np.abs(lateral) <= section_half + 1.0
    if int(np.count_nonzero(keep)) < 2:
        keep = np.ones(len(lateral), dtype=bool)
    lat = lateral[keep]
    val = values[keep]
    crown = max(0.0, CROWN - abs(tilt))
    predicted = z_center - crown * np.abs(lat) + tilt * lat
    return float(np.max(np.abs(val - predicted))) <= RESIDUAL_M


def _target_tilt(frame: StationFrame) -> np.ndarray:
    heading = np.unwrap(np.arctan2(frame.tangent[:, 1], frame.tangent[:, 0]))
    count = len(heading)
    kappa = np.zeros(count)
    for i in range(1, count - 1):
        span = float(frame.s[i + 1] - frame.s[i - 1])
        if span > 1e-6:
            kappa[i] = (heading[i + 1] - heading[i - 1]) / span
    if count > 1:
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]
    tilt = np.array([_tilt_from_kappa(float(value)) for value in kappa])
    return _runoff(tilt, frame.s)


def _tilt_from_kappa(kappa: float) -> float:
    """Positive tilt raises the left edge. A left turn raises the outside, which is the right edge."""
    if abs(kappa) < 1.0 / R_TANGENT_M:
        return 0.0
    radius = 1.0 / abs(kappa)
    if radius <= R_FULL_M:
        magnitude = SE_MAX
    else:
        magnitude = SE_MAX * (R_TANGENT_M - radius) / (R_TANGENT_M - R_FULL_M)
    return -math.copysign(magnitude, kappa)


def _runoff(tilt: np.ndarray, distances: np.ndarray) -> np.ndarray:
    rate = SE_MAX / RUNOFF_M
    forward = np.array(tilt, dtype=float)
    backward = np.array(tilt, dtype=float)
    for i in range(1, len(forward)):
        cap = rate * max(float(distances[i] - distances[i - 1]), 1e-6)
        forward[i] = min(max(forward[i], forward[i - 1] - cap), forward[i - 1] + cap)
    for i in range(len(backward) - 2, -1, -1):
        cap = rate * max(float(distances[i + 1] - distances[i]), 1e-6)
        backward[i] = min(max(backward[i], backward[i + 1] - cap), backward[i + 1] + cap)
    blended = 0.5 * (forward + backward)
    for i in range(1, len(blended)):
        cap = rate * max(float(distances[i] - distances[i - 1]), 1e-6)
        blended[i] = min(max(blended[i], blended[i - 1] - cap), blended[i - 1] + cap)
    return np.clip(blended, -SE_MAX, SE_MAX)


def _smooth_tied(
    z: np.ndarray,
    distances: np.ndarray,
    accepted: np.ndarray,
    max_grade: float,
) -> np.ndarray:
    """Smooth each run, but keep every sustained station within PROFILE_TIE_M of the LiDAR."""
    out = _limit_grade(z, distances, accepted, max_grade)
    for start, stop in true_runs(accepted, min_len=4):
        raw = np.asarray(z[start:stop], dtype=float)
        dist = distances[start:stop]
        anchor = _despike(raw)
        best = _tie(_clamp_run(anchor, dist, max_grade), anchor)
        best_rough = _roughness(best)
        for lam in np.logspace(-3, 5, 17):
            try:
                trial = _spline_ls(anchor, float(lam))
            except np.linalg.LinAlgError:
                continue
            trial = _clamp_run(trial, dist, max_grade)
            trial = _tie(trial, anchor)
            rough = _roughness(trial)
            if rough <= best_rough:
                best = trial
                best_rough = rough
        out[start:stop] = best
    return out


def _despike(z: np.ndarray) -> np.ndarray:
    """Replace a one-station reversal with the chord. A sustained grade does not reverse."""
    out = np.array(z, dtype=float)
    for i in range(1, len(z) - 1):
        arriving = float(z[i] - z[i - 1])
        leaving = float(z[i + 1] - z[i])
        chord = 0.5 * (z[i - 1] + z[i + 1])
        if arriving * leaving < 0.0 and abs(float(z[i] - chord)) > PROFILE_TIE_M:
            out[i] = chord
    return out


def _tie(trial: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    out = np.array(trial, dtype=float)
    delta = out - anchor
    over = np.abs(delta) > PROFILE_TIE_M
    out[over] = anchor[over] + np.sign(delta[over]) * PROFILE_TIE_M
    return out


def _roughness(z: np.ndarray) -> float:
    if len(z) < 3:
        return 0.0
    second = z[:-2] - 2.0 * z[1:-1] + z[2:]
    return float(np.sum(second * second))


def _spline_ls(raw: np.ndarray, lam: float) -> np.ndarray:
    count = len(raw)
    second = np.zeros((count - 2, count))
    for i in range(count - 2):
        second[i, i] = 1.0
        second[i, i + 1] = -2.0
        second[i, i + 2] = 1.0
    system = np.eye(count) + lam * (second.T @ second)
    return np.linalg.solve(system, raw)


def _limit_grade(
    z: np.ndarray,
    distances: np.ndarray,
    accepted: np.ndarray,
    max_grade: float,
) -> np.ndarray:
    limited = np.array(z, dtype=float)
    for start, stop in true_runs(accepted, min_len=2):
        limited[start:stop] = _clamp_run(limited[start:stop], distances[start:stop], max_grade)
    return limited


def _clamp_run(z: np.ndarray, distances: np.ndarray, max_grade: float) -> np.ndarray:
    forward = np.array(z, dtype=float)
    for i in range(1, len(forward)):
        cap = max_grade * max(float(distances[i] - distances[i - 1]), 1e-6)
        forward[i] = min(max(forward[i], forward[i - 1] - cap), forward[i - 1] + cap)
    backward = np.array(z, dtype=float)
    for i in range(len(backward) - 2, -1, -1):
        cap = max_grade * max(float(distances[i + 1] - distances[i]), 1e-6)
        backward[i] = min(max(backward[i], backward[i + 1] - cap), backward[i + 1] + cap)
    blended = 0.5 * (forward + backward)
    for i in range(1, len(blended)):
        cap = max_grade * max(float(distances[i] - distances[i - 1]), 1e-6)
        blended[i] = min(max(blended[i], blended[i - 1] - cap), blended[i - 1] + cap)
    return blended


def _offset_edges(
    xy: np.ndarray,
    tangent: np.ndarray,
    half: float,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    neg: list[np.ndarray] = []
    pos: list[np.ndarray] = []
    for i in range(len(xy)):
        normal = np.array([-tangent[i, 1], tangent[i, 0]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        normal /= norm
        neg.append(xy[i] - normal * half)
        pos.append(xy[i] + normal * half)
    return neg, pos


def _largest_poly(geom: Any) -> Polygon | MultiPolygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polys = [part for part in getattr(geom, "geoms", []) if isinstance(part, (Polygon, MultiPolygon))]
    if not polys:
        return None
    return max(polys, key=lambda part: part.area)
