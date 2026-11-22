"""Bridge / overpass deck extraction from LiDAR layers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.validation import make_valid

from roader.legacy.lidar_layers import (
    LidarLayers,
    elevated_candidate_points,
    estimate_ground_z,
    points_to_alpha_outline,
)
from roader.legacy.model import Carriageway, Deck, VoidUnder
from roader.legacy.surface import (
    BANK_TOE_INSET_M,
    CORRIDOR_MARGIN_M,
    FILL_MAX_M,
    FILL_RISE_M,
    GAP_FAIL_M,
    LIP_CLOSE_M,
    MIN_DECK_RUN_M,
    BankSample,
    Cloud,
    DeckStrip,
    FittedSurface,
    StationArray,
    build_prism,
    class_half_limits,
    footprint_polygon,
    hole_lengths,
    is_parking_bleed,
    max_grade_for,
    sample_banks,
    sample_profile,
    station_frame,
    true_runs,
)


def extract_decks(
    layers: LidarLayers,
    height_above_ground_m: float = 2.5,
    cell_m: float = 1.0,
    min_points: int = 40,
) -> list[Deck]:
    xy, z = elevated_candidate_points(layers, height_above_ground_m)
    if len(xy) < min_points:
        return []

    source = "las_class_17" if len(layers.bridge_xy) else "elevated_cluster"
    outline = points_to_alpha_outline(xy, cell_m=cell_m)
    if outline is None or outline.is_empty:
        return []

    geom = make_valid(outline)
    parts: list[Polygon] = []
    if isinstance(geom, Polygon):
        parts = [geom]
    elif isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
    else:
        # GeometryCollection etc.
        parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]

    decks: list[Deck] = []
    for i, poly in enumerate(parts):
        if poly.is_empty or poly.area < (cell_m * cell_m * 4):
            continue
        # Points inside this component
        from shapely import vectorized

        inside = vectorized.contains(poly, xy[:, 0], xy[:, 1])
        if inside.sum() < min_points // 2:
            continue
        zz = z[inside]
        decks.append(
            Deck(
                id=f"deck_{i}",
                polygon=poly,
                z_p95_m=float(np.percentile(zz, 95)),
                z_max_m=float(np.max(zz)),
                point_count=int(inside.sum()),
                source=source,
            )
        )
    return decks


def voids_under_decks(
    decks: list[Deck],
    layers: LidarLayers,
) -> list[VoidUnder]:
    voids: list[VoidUnder] = []
    for deck in decks:
        # Sample ground under deck centroid grid
        minx, miny, maxx, maxy = deck.polygon.bounds
        xs = np.linspace(minx, maxx, num=7)
        ys = np.linspace(miny, maxy, num=7)
        grid = np.array([(x, y) for x in xs for y in ys])
        # Keep samples inside polygon
        from shapely import vectorized

        inside = vectorized.contains(deck.polygon, grid[:, 0], grid[:, 1])
        grid = grid[inside]
        if len(grid) == 0:
            centroid = deck.polygon.centroid
            grid = np.array([[centroid.x, centroid.y]])
        gz = estimate_ground_z(layers, grid)
        gz = gz[np.isfinite(gz)]
        z_floor = float(np.percentile(gz, 50)) if len(gz) else None
        z_ceil = deck.z_p95_m
        clearance = (z_ceil - z_floor) if z_floor is not None else None
        deck.clearance_below_m = clearance
        voids.append(
            VoidUnder(
                id=f"void_{deck.id}",
                polygon=deck.polygon,
                deck_id=deck.id,
                z_floor_m=z_floor,
                z_ceiling_m=z_ceil,
                clearance_m=clearance,
                kind="terrain" if z_floor is not None else "unknown",
            )
        )
    return voids


def profile_z_at(carriageway: Carriageway, x: float, y: float) -> float | None:
    if carriageway.profile_xy and carriageway.profile_z:
        xy = np.asarray(carriageway.profile_xy, dtype=float)
        dist = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2
        return float(carriageway.profile_z[int(np.argmin(dist))])
    return carriageway.z_mean_m


def ribbons_from_clouds(
    line,
    ground: Cloud,
    bridge: Cloud,
    *,
    highway: str,
    pavement_thickness_m: float,
    deck_thickness_m: float,
    deck_height_m: float,
    bridge_tag: bool = False,
    ndvi=None,
) -> list[FittedSurface]:
    """Fit one or more closed ribbons. A single reasoned result means skip the way."""
    frame = station_frame(line)
    min_half, max_half = class_half_limits(highway)
    search_half = max_half + CORRIDOR_MARGIN_M
    grade = max_grade_for(highway)
    ground_profile = sample_profile(
        frame,
        ground,
        search_half_m=search_half,
        max_grade=grade,
        section_half_m=max_half,
        ndvi=ndvi,
    )
    bridge_profile = sample_profile(
        frame,
        bridge,
        search_half_m=search_half,
        max_grade=grade,
        section_half_m=max_half,
        ndvi=ndvi,
    )
    # Class 2 on a deck is the deck. The lip and the toe say whether that height is fill or a span.
    banks = sample_banks(frame, ground, max_half)
    support = _support(ground_profile, bridge_profile, banks, frame.s, deck_height_m, bridge_tag)
    deck = support.kind == "deck"
    covered = deck | ground_profile.accepted
    if any(length > GAP_FAIL_M for length in hole_lengths(covered, frame.s)):
        return [FittedSurface(reason="lidar_gap")]
    if is_parking_bleed(line, highway, _measured_max(deck, ground_profile, bridge_profile, covered), max_half):
        return [FittedSurface(reason="parking_bleed")]

    runs = true_runs(covered, min_len=2)
    if not runs:
        return [FittedSurface(reason="lidar_gap")]

    fitted: list[FittedSurface] = []
    for start, stop in runs:
        surface = _ribbon_run(
            line,
            frame,
            ground_profile,
            bridge_profile,
            deck,
            support,
            start,
            stop,
            min_half,
            max_half,
            pavement_thickness_m,
            deck_thickness_m,
        )
        if surface is not None:
            fitted.append(surface)
    if not fitted:
        return [FittedSurface(reason="lidar_gap")]
    return fitted


def voids_from_profiles(
    carriageways: list[Carriageway],
    decks: list[Deck],
    *,
    deck_thickness_m: float,
    z_threshold: float,
) -> list[VoidUnder]:
    """Clearance from the upper soffit down to the next road, else to terrain."""
    by_id = {carriageway.id: carriageway for carriageway in carriageways}
    voids: list[VoidUnder] = []
    for deck in decks:
        host = by_id.get(str(deck.attrs.get("carriageway_id", "")))
        if host is None:
            continue
        z_ground = _deck_ground_z(host, deck)
        best_lower: tuple[float, float] | None = None
        for other in carriageways:
            if other.id == host.id or other.polygon is None:
                continue
            overlap = deck.polygon.intersection(other.polygon)
            if overlap.is_empty or overlap.area < 1.0:
                continue
            point = overlap.centroid
            z_other = profile_z_at(other, point.x, point.y)
            z_here = profile_z_at(host, point.x, point.y)
            if z_here is None or z_other is None:
                continue
            if z_here - z_other <= z_threshold:
                continue
            if best_lower is None or z_other > best_lower[0]:
                best_lower = (z_other, z_here)
        if best_lower is not None:
            z_floor = best_lower[0]
            z_ceil = best_lower[1] - deck_thickness_m
            kind = "road"
        else:
            z_floor = z_ground
            z_surface = _deck_surface_z(host, deck)
            z_ceil = None if z_surface is None else z_surface - deck_thickness_m
            kind = "terrain" if z_floor is not None else "unknown"
        clearance = None
        if z_ceil is not None and z_floor is not None:
            clearance = float(z_ceil - z_floor)
        deck.clearance_below_m = clearance
        voids.append(
            VoidUnder(
                id=f"void_{deck.id}",
                polygon=deck.polygon,
                deck_id=deck.id,
                z_floor_m=z_floor,
                z_ceiling_m=z_ceil,
                clearance_m=clearance,
                kind=kind,
            )
        )
    return voids


@dataclass
class _Support:
    kind: np.ndarray
    toe_z: np.ndarray
    left_flare: np.ndarray
    right_flare: np.ndarray


def _support(
    ground: StationArray,
    bridge: StationArray,
    banks: BankSample,
    distances: np.ndarray,
    deck_height_m: float,
    bridge_tag: bool,
) -> _Support:
    """Fill where the shoulder still meets the pavement. Deck where the edge has dropped.

    Class 17 seeds a span. A bridge tag can too, once both lips are already below
    the pavement. Fill uses only a side whose lip still meets the pavement, and
    only down to that side's toe. A river or a cut on the other side does not
    pull the solid down. A run shorter than the deck minimum goes back to fill
    or grade.
    """
    count = len(distances)
    kind = np.array(["grade"] * count, dtype=object)
    toe_z = np.full(count, np.nan)
    fill_toe = np.full(count, np.nan)
    left_flare = np.zeros(count)
    right_flare = np.zeros(count)
    seed = np.zeros(count, dtype=bool)
    cliff = np.zeros(count, dtype=bool)
    for i in range(count):
        surface = _surface_z(ground, bridge, i)
        opening = _toe_z(banks, i)
        if not np.isfinite(surface):
            continue
        class_17 = (
            bridge.accepted[i]
            and np.isfinite(bridge.z[i])
            and (not np.isfinite(opening) or float(bridge.z[i] - opening) >= deck_height_m)
        )
        fill_toe[i] = _attached_toe(banks, surface, i)
        left_flare[i] = _side_flare(banks.left_lip[i], banks.left_toe[i], surface)
        right_flare[i] = _side_flare(banks.right_lip[i], banks.right_toe[i], surface)
        if not np.isfinite(opening):
            seed[i] = class_17
            toe_z[i] = fill_toe[i]
            continue
        lips = [float(banks.left_lip[i]), float(banks.right_lip[i])]
        present = [value for value in lips if np.isfinite(value)]
        both_low = not present or all(float(surface - value) >= deck_height_m for value in present)
        cliff[i] = both_low and float(surface - opening) >= deck_height_m
        seed[i] = class_17 or (bridge_tag and cliff[i])
        toe_z[i] = opening
    deck = _grow_deck(seed, cliff)
    for start, stop in true_runs(deck, min_len=2):
        if float(distances[stop - 1] - distances[start]) < MIN_DECK_RUN_M:
            deck[start:stop] = False
    # A lip that returns for less than a deck length does not break the span or the bank.
    deck = _hold_runs(deck, distances, MIN_DECK_RUN_M)
    fill = np.isfinite(fill_toe)
    fill = _hold_runs(fill, distances, MIN_DECK_RUN_M, fill_toe, left_flare, right_flare)
    kind[fill] = "fill"
    kind[deck] = "deck"
    left_flare[kind != "fill"] = 0.0
    right_flare[kind != "fill"] = 0.0
    use_fill = kind == "fill"
    toe_z[use_fill] = fill_toe[use_fill]
    return _Support(kind=kind, toe_z=toe_z, left_flare=left_flare, right_flare=right_flare)


def _hold_runs(
    mask: np.ndarray,
    distances: np.ndarray,
    hold_m: float,
    *carry: np.ndarray,
) -> np.ndarray:
    """Keep a run across a gap shorter than hold_m, copying the previous station's values."""
    runs = list(true_runs(mask, min_len=1))
    if len(runs) < 2:
        return mask
    out = mask.copy()
    for (left_start, left_stop), (right_start, _right_stop) in itertools.pairwise(runs):
        if right_start <= left_stop:
            continue
        gap_m = float(distances[right_start] - distances[left_stop - 1])
        if gap_m >= hold_m:
            continue
        out[left_stop:right_start] = True
        for array in carry:
            array[left_stop:right_start] = array[left_stop - 1]
    return out


def _side_flare(lip: float, toe: float, surface: float) -> float:
    """How far the fill bottom steps out when this side still meets the pavement."""
    if not np.isfinite(lip) or not np.isfinite(toe) or not np.isfinite(surface):
        return 0.0
    if float(surface - lip) > LIP_CLOSE_M:
        return 0.0
    drop = float(surface - toe)
    if drop < FILL_RISE_M or drop > FILL_MAX_M:
        return 0.0
    return BANK_TOE_INSET_M


def _grow_deck(seed: np.ndarray, cliff: np.ndarray) -> np.ndarray:
    """Carry a span onto the next station while that edge is still a drop."""
    deck = seed.copy()
    changed = True
    while changed:
        changed = False
        for i in range(len(deck)):
            if deck[i] or not cliff[i]:
                continue
            left = i > 0 and deck[i - 1]
            right = i + 1 < len(deck) and deck[i + 1]
            if left or right:
                deck[i] = True
                changed = True
    return deck


def _surface_z(ground: StationArray, bridge: StationArray, index: int) -> float:
    if ground.accepted[index] and np.isfinite(ground.z[index]):
        return float(ground.z[index])
    if bridge.accepted[index] and np.isfinite(bridge.z[index]):
        return float(bridge.z[index])
    return float("nan")


def _attached_toe(banks: BankSample, surface: float, index: int) -> float:
    """Toe of a side whose lip still meets the pavement.

    The higher of those toes is the smaller bank. The other side may be a cut
    or a river, and that drop is not fill.
    """
    toes: list[float] = []
    for lip, toe in (
        (banks.left_lip[index], banks.left_toe[index]),
        (banks.right_lip[index], banks.right_toe[index]),
    ):
        if not np.isfinite(lip) or not np.isfinite(toe):
            continue
        if float(surface - lip) > LIP_CLOSE_M:
            continue
        drop = float(surface - toe)
        if drop < FILL_RISE_M or drop > FILL_MAX_M:
            continue
        toes.append(float(toe))
    if not toes:
        return float("nan")
    return max(toes)


def _toe_z(banks: BankSample, index: int) -> float:
    values = [
        float(value)
        for value in (banks.left_toe[index], banks.right_toe[index])
        if np.isfinite(value)
    ]
    if not values:
        return float("nan")
    return min(values)


def _measured_max(
    deck: np.ndarray,
    ground: StationArray,
    bridge: StationArray,
    covered: np.ndarray,
) -> float:
    values: list[float] = []
    for i, flag in enumerate(covered):
        if not flag:
            continue
        source = bridge if deck[i] else ground
        if np.isfinite(source.measured_half[i]):
            values.append(float(source.measured_half[i]))
    if not values:
        return float("nan")
    return max(values)


def _ribbon_run(
    line,
    frame,
    ground: StationArray,
    bridge: StationArray,
    deck: np.ndarray,
    support: _Support,
    start: int,
    stop: int,
    min_half: float,
    max_half: float,
    pavement_thickness_m: float,
    deck_thickness_m: float,
) -> FittedSurface | None:
    count = stop - start
    z = np.empty(count)
    slope = np.empty(count)
    crown = np.empty(count)
    template = np.zeros(count, dtype=bool)
    thickness = np.empty(count)
    ground_z = np.full(count, np.nan)
    flare_pos = np.zeros(count)
    flare_neg = np.zeros(count)
    measured: list[float] = []
    for offset, index in enumerate(range(start, stop)):
        on_class_17 = deck[index] and bridge.accepted[index] and np.isfinite(bridge.z[index])
        source = bridge if on_class_17 else ground
        z[offset] = source.z[index]
        slope[offset] = source.slope[index]
        crown[offset] = source.crown[index]
        template[offset] = bool(source.template[index])
        station_kind = str(support.kind[index])
        toe = float(support.toe_z[index])
        if station_kind == "deck":
            thickness[offset] = deck_thickness_m
        elif station_kind == "fill" and np.isfinite(toe):
            bank = float(z[offset] - toe)
            thickness[offset] = max(pavement_thickness_m, bank)
            flare_pos[offset] = float(support.left_flare[index])
            flare_neg[offset] = float(support.right_flare[index])
        else:
            thickness[offset] = pavement_thickness_m
        measured.append(float(source.measured_half[index]))
        if np.isfinite(toe):
            ground_z[offset] = toe
        elif ground.accepted[index] and np.isfinite(ground.z[index]):
            ground_z[offset] = float(ground.z[index])
    half = float(np.median([_clamp_half(value, min_half, max_half) for value in measured]))
    xy = frame.xy[start:stop]
    tangent = frame.tangent[start:stop]
    mesh = build_prism(
        xy,
        tangent,
        z,
        slope,
        crown,
        half,
        thickness,
        flare_pos=flare_pos,
        flare_neg=flare_neg,
    )
    if mesh is None:
        return None
    faces, polygon = mesh
    deck_slice = deck[start:stop]
    parts = []
    for part_start, part_stop in true_runs(deck_slice, min_len=2):
        part_poly = footprint_polygon(xy[part_start:part_stop], tangent[part_start:part_stop], half)
        if part_poly is None:
            continue
        zz = z[part_start:part_stop]
        absolute = np.arange(start, stop)[part_start:part_stop]
        parts.append(_deck_strip(part_poly, zz, bridge.point_count[absolute]))
    profile_xy = [(float(point[0]), float(point[1])) for point in xy]
    center = line if start == 0 and stop == len(frame.s) else _station_line(profile_xy)
    return FittedSurface(
        polygon=polygon,
        raw_faces=faces,
        centerline=center,
        z_mean=float(np.nanmean(z)),
        half_width_m=half,
        profile_xy=profile_xy,
        profile_z=[float(value) for value in z],
        profile_ground_z=[float(value) for value in ground_z],
        profile_deck=[bool(flag) for flag in deck_slice],
        deck_parts=parts,
        section=_section_label(template),
    )


def _deck_strip(polygon, zz: np.ndarray, point_count: np.ndarray) -> DeckStrip:
    return DeckStrip(
        polygon=polygon,
        z_p95=float(np.percentile(zz, 95)),
        z_max=float(np.max(zz)),
        point_count=int(np.sum(point_count)),
    )


def _section_label(template: np.ndarray) -> str:
    if len(template) == 0:
        return "lidar"
    share = float(np.mean(template))
    if share >= 0.7:
        return "standard"
    if share <= 0.3:
        return "lidar"
    return "mixed"


def _clamp_half(measured: float, min_half: float, max_half: float) -> float:
    if not np.isfinite(measured):
        return max_half
    return float(min(max(measured, min_half), max_half))


def _station_line(coords: list[tuple[float, float]]) -> LineString | None:
    if len(coords) < 2:
        return None
    return LineString(coords)


def _points_inside(carriageway: Carriageway, deck: Deck) -> list[int]:
    indexes: list[int] = []
    for i, (x, y) in enumerate(carriageway.profile_xy):
        point = Point(x, y)
        if deck.polygon.covers(point):
            indexes.append(i)
    return indexes


def _deck_surface_z(carriageway: Carriageway, deck: Deck) -> float | None:
    values = [
        carriageway.profile_z[i]
        for i in _points_inside(carriageway, deck)
        if carriageway.profile_deck[i]
    ]
    if not values and carriageway.z_mean_m is not None:
        return float(carriageway.z_mean_m)
    if not values:
        return None
    return float(np.median(values))


def _deck_ground_z(carriageway: Carriageway, deck: Deck) -> float | None:
    values = []
    for i in _points_inside(carriageway, deck):
        if not carriageway.profile_deck[i]:
            continue
        ground_z = carriageway.profile_ground_z[i]
        if np.isfinite(ground_z):
            values.append(ground_z)
    if not values:
        return None
    return float(np.median(values))

