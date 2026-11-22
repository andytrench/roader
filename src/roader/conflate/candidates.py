"""Paved surfaces with no matching vector (FR-VC-5).

At interchange scale a sheet is shared: a ramp nobody mapped joins the
mainline sheet at its taper, so "a sheet no road tracks" misses it, and one
sheet may carry several roads. Candidates are therefore found per paved
layer-cell, not per sheet. A paved cell is covered when some fitted road runs
over it in plan (inside its pavement plus ``COVER_PAD_M``, or within
``END_PAD_M`` past its ends, where roads are clipped) at its own height
(within ``COVER_DZ_M`` of that road's profile), so a deck never covers the road
beneath it. Uncovered paved cells joined at sheet height form components; a
component is reported when it is large, at least ``MIN_WIDTH_M`` wide (not a
shoulder sliver) and no wider than ``MAX_WIDTH_M`` (a road, not a parking lot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt

from roader.evidence import Evidence
from roader.evidence.build import PAVEMENT_P
from roader.evidence.features import _components
from roader.evidence.layers import JOIN_DZ_M, neighbour_pairs
from roader.fit.road import RoadFit
from roader.model.schema import Flag
from roader.model.serialize import stable_id

COVER_PAD_M = 3.0
END_PAD_M = 15.0
COVER_DZ_M = 1.0
MIN_CELLS = 400  # FR-VC-5: paved area this large (m2 at 1 m cells) with no road is reported
MIN_LENGTH_M = 40.0
MIN_WIDTH_M = 3.0
MAX_WIDTH_M = 24.0
OUTLINE_SIMPLIFY_M = 1.0


@dataclass
class SheetCandidate:
    """A paved surface no vector explains: geometry in the unit frame."""

    id: str
    sheets: list[int]  # surface sheets its cells belong to (FR-RS-2)
    area_m2: float
    length_m: float  # area over width
    width_m: float  # twice the largest distance from a cell to the component boundary
    z_min: float
    z_max: float
    centroid: tuple[float, float]
    outline: list[tuple[float, float]]  # plan polygon, counter-clockwise, simplified

    def wkt(self) -> str:
        return shapely.Polygon(self.outline).wkt


def covered_cells(ev: Evidence, fits: list[RoadFit], cells: np.ndarray) -> np.ndarray:
    """Per layer index in ``cells``: True when a fitted road runs over it at its height."""
    lay = ev.layers
    xy = lay.center_xy(cells)
    z = lay.z[cells]
    out = np.zeros(len(cells), dtype=bool)
    for f in sorted(fits, key=lambda f: f.road.id):
        road = f.road
        xs = road.cross_section
        reach = max(max(k.value for k in xs.width_left.knots), max(k.value for k in xs.width_right.knots)) + COVER_PAD_M
        s, u, _ok = f.line.project(xy, max_dist=reach + END_PAD_M)
        has = np.isfinite(s) & np.isfinite(u)
        if not has.any():
            continue
        st = np.arange(0.0, road.length + 1.0, 1.0)
        sc = np.clip(s[has], 0.0, road.length)
        zp = np.interp(sc, st, [road.profile.z(float(v)) for v in np.clip(st, 0.0, road.length)])
        wl = np.interp(sc, [k.s for k in xs.width_left.knots], [k.value for k in xs.width_left.knots]) + COVER_PAD_M
        wr = np.interp(sc, [k.s for k in xs.width_right.knots], [k.value for k in xs.width_right.knots]) + COVER_PAD_M
        ok = (s[has] >= -END_PAD_M) & (s[has] <= road.length + END_PAD_M) & (u[has] <= wl) & (u[has] >= -wr)
        ok &= np.abs(z[has] - zp) <= COVER_DZ_M
        idx = np.flatnonzero(has)
        out[idx[ok]] = True
    return out


def unmatched_sheets(ev: Evidence, fits: list[RoadFit]) -> list[SheetCandidate]:
    """FR-VC-5 candidates, ordered by id (FR-PL-3)."""
    lay = ev.layers
    paved = (ev.layer_prob >= PAVEMENT_P) & (ev.layer_label == 0) & (lay.sheet >= 0)
    cells = np.flatnonzero(paved)
    if cells.size == 0:
        return []
    free = np.zeros(len(lay), dtype=bool)
    free[cells[~covered_cells(ev, fits, cells)]] = True
    a, b = neighbour_pairs(lay, free, JOIN_DZ_M * lay.cell_m, diagonal=True)
    labels = _components(len(lay), a, b)
    idx = np.flatnonzero(free)
    roots = labels[idx]
    order = np.lexsort((idx, roots))
    idx, roots = idx[order], roots[order]
    out = []
    for g in np.split(idx, np.flatnonzero(np.diff(roots)) + 1):
        if g.size < MIN_CELLS:
            continue
        cand = _candidate(ev, g)
        if cand is not None:
            out.append(cand)
    return sorted(out, key=lambda c: c.id)


def candidate_flag(c: SheetCandidate) -> Flag:
    """The review-queue record of one candidate, with its outline as WKT in the unit frame."""
    return Flag(
        "FR-VC-5:candidate",
        c.id,
        "FR-VC-5",
        f"paved surface without a vector: {c.area_m2:.0f} m2, about {c.length_m:.0f} m long and {c.width_m:.0f} m wide, "
        f"z {c.z_min:.1f}-{c.z_max:.1f}, centroid ({c.centroid[0]:.1f}, {c.centroid[1]:.1f}), outline {c.wkt()}",
    )


def _candidate(ev: Evidence, g: np.ndarray) -> SheetCandidate | None:
    lay = ev.layers
    c = lay.cell_m
    ix, iy = lay.ix[g], lay.iy[g]
    grid = np.zeros((int(iy.max() - iy.min()) + 3, int(ix.max() - ix.min()) + 3), dtype=bool)
    grid[iy - iy.min() + 1, ix - ix.min() + 1] = True
    width = 2.0 * float(distance_transform_edt(grid).max()) * c
    area = float(np.unique(lay.key[g]).size) * c * c
    length = area / max(width, c)
    if width < MIN_WIDTH_M or width > MAX_WIDTH_M or length < MIN_LENGTH_M:
        return None
    x0 = lay.x0 + ix * c
    y0 = lay.y0 + iy * c
    shape = shapely.union_all(shapely.box(x0, y0, x0 + c, y0 + c))
    parts = [p for p in getattr(shape, "geoms", [shape]) if isinstance(p, shapely.Polygon)]
    poly = shapely.Polygon(max(parts, key=lambda p: p.area).exterior).simplify(OUTLINE_SIMPLIFY_M)
    poly = shapely.geometry.polygon.orient(poly, 1.0)
    outline = [(round(float(x), 2), round(float(y), 2)) for x, y in poly.exterior.coords[:-1]]
    xy = lay.center_xy(g)
    cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    z = lay.z[g]
    sheets = sorted({int(s) for s in lay.sheet[g] if s >= 0})
    return SheetCandidate(
        id=stable_id("cand", round(cx, 0), round(cy, 0), round(float(z.min()), 0)),
        sheets=sheets,
        area_m2=area,
        length_m=length,
        width_m=width,
        z_min=float(z.min()),
        z_max=float(z.max()),
        centroid=(cx, cy),
        outline=outline,
    )
