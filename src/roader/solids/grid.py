"""Per-road cell grid in (s, u): station lines crossed by longitudinal u-lines.

Every body on a road (pavement, earthwork, deck, parapets, abutments, pier caps)
is a union of these cells, so bodies that touch share stations and vertices by
construction (FR-SG-2, FR-SV-7). A u-line is u = f(s), active over a station range:
pavement edges, crown, deck edges, barrier breaklines, embankment toes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from roader.model.schema import Road
from roader.solids.stations import STATION_MERGE_M, StationLine

U_MERGE_M = 1e-9
FOLD_MARGIN = 0.05

Tri = tuple[int, int, int]


@dataclass
class ULine:
    key: str
    func: Callable[[float], float]
    s0: float
    s1: float


@dataclass
class Cell:
    loop: list[int]  # plan vertex ids, counter-clockwise in plan
    tris: list[Tri]  # counter-clockwise in plan


class PlanVertices:
    """Plan positions shared by all bodies; (s, u) are NaN for vertices not on a road grid.

    Grid vertices also record their road, their station-line index (row) and the keys
    of the u-lines through them, so shading creases and UVs follow the model (FR-AS-3/4).
    """

    def __init__(self) -> None:
        self.x: list[float] = []
        self.y: list[float] = []
        self.s: list[float] = []
        self.u: list[float] = []
        self.road: list[str | None] = []
        self.row: list[int] = []
        self.keys: dict[int, set[str]] = {}

    def add(self, x: float, y: float, s: float = math.nan, u: float = math.nan, road: str | None = None, row: int = -1) -> int:
        self.x.append(x)
        self.y.append(y)
        self.s.append(s)
        self.u.append(u)
        self.road.append(road)
        self.row.append(row)
        return len(self.x) - 1

    def d2(self, a: int, b: int) -> float:
        return (self.x[a] - self.x[b]) ** 2 + (self.y[a] - self.y[b]) ** 2

    def area2(self, a: int, b: int, c: int) -> float:
        x, y = self.x, self.y
        return (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a])


def ladder(left: list[int], right: list[int], shorter: Callable[[int, int, int, int], bool]) -> list[Tri]:
    """Triangulate the strip between two ordered point rows, left row at lower station.

    shorter(l_next, r, l, r_next) decides whether to advance on the left row.
    Output is counter-clockwise when rows run in increasing u.
    """
    tris: list[Tri] = []
    i = j = 0
    while i < len(left) - 1 or j < len(right) - 1:
        if j == len(right) - 1 or (i < len(left) - 1 and shorter(left[i + 1], right[j], left[i], right[j + 1])):
            tris.append((left[i], right[j], left[i + 1]))
            i += 1
        else:
            tris.append((left[i], right[j], right[j + 1]))
            j += 1
    return tris


def _solve(func: Callable[[float], float], line: StationLine) -> float:
    u = func(line.s)
    if line.t == 0.0:
        return u
    u0 = u
    for _ in range(200):
        nu = func(line.at(u))
        if abs(nu - u) <= 1e-12:
            return nu
        u = nu
    # Fixed point oscillates when |du/ds| * |t| >= 1 (a toe over rough terrain): bisect instead.
    def h(v: float) -> float:
        return func(line.at(v)) - v

    for reach in (1.0, 4.0, 16.0, 64.0):
        lo, hi = u0 - reach, u0 + reach
        if h(lo) * h(hi) <= 0.0:
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if h(lo) * h(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            return func(line.at(0.5 * (lo + hi)))
    raise ValueError(f"u-line does not converge on the skewed station line at s={line.s:.3f}")


class RoadGrid:
    def __init__(self, road: Road, lines: list[ULine], station_lines: list[StationLine], pv: PlanVertices) -> None:
        self.road = road
        self.sl = station_lines
        self.pv = pv
        n = len(station_lines)
        span: dict[str, tuple[int, int]] = {}
        for line in lines:
            i0, i1 = self.index(line.s0), self.index(line.s1)
            if i1 > i0:
                span[line.key] = (i0, i1)
        self.span = span

        self.vid: dict[tuple[int, str], int] = {}
        self.row: list[list[int]] = []
        rank: dict[int, int] = {}
        funcs = {line.key: line.func for line in lines}
        length = road.length
        for i, sline in enumerate(station_lines):
            items = sorted(
                ((_solve(funcs[key], sline), order, key) for order, (key, (i0, i1)) in enumerate(span.items()) if i0 <= i <= i1),
                key=lambda item: (item[0], item[1]),
            )
            row: list[int] = []
            last_u = -math.inf
            for u, _order, key in items:
                if row and u - last_u <= U_MERGE_M:
                    self.vid[(i, key)] = row[-1]
                    pv.keys[row[-1]].add(key)
                    continue
                s = sline.at(u)
                if s < -1e-9 or s > length + 1e-9:
                    raise ValueError(f"road {road.id}: skewed station line at s={sline.s:.2f} leaves the road at u={u:.2f}")
                s = min(max(s, 0.0), length)
                k = road.alignment.evaluate(s)[3]
                if 1.0 - k * u < FOLD_MARGIN:
                    raise ValueError(f"road {road.id}: offset u={u:.2f} folds over the centre of curvature at s={s:.2f}")
                x, y = road.alignment.offset_point(s, u)
                p = pv.add(x, y, s, u, road.id, i)
                pv.keys[p] = {key}
                rank[p] = len(row)
                row.append(p)
                self.vid[(i, key)] = p
                last_u = u
            self.row.append(row)

        self.key_pos: dict[tuple[int, str], int] = {}
        self.cells: dict[tuple[int, int], Cell] = {}
        for i in range(n - 1):
            entries = sorted(
                (rank[self.vid[(i, key)]], rank[self.vid[(i + 1, key)]], order, key)
                for order, (key, (i0, i1)) in enumerate(span.items())
                if i0 <= i and i1 >= i + 1
            )
            groups: list[tuple[int, int]] = []
            for li, ri, _order, key in entries:
                if groups and groups[-1] == (li, ri):
                    self.key_pos[(i, key)] = len(groups) - 1
                    continue
                if groups and ri < groups[-1][1]:
                    raise ValueError(f"road {road.id}: u-line {key} crosses another between s={station_lines[i].s:.2f} and s={station_lines[i + 1].s:.2f}")
                groups.append((li, ri))
                self.key_pos[(i, key)] = len(groups) - 1
            for k in range(len(groups) - 1):
                (l0, r0), (l1, r1) = groups[k], groups[k + 1]
                left = self.row[i][l0 : l1 + 1]
                right = self.row[i + 1][r0 : r1 + 1]
                tris = ladder(left, right, lambda ln, r, lc, rn: pv.d2(ln, r) <= pv.d2(lc, rn))
                for a, b, c in tris:
                    if pv.area2(a, b, c) <= 1e-12:
                        raise ValueError(f"road {road.id}: station lines cross near s={station_lines[i].s:.2f} (reduce skew or ramp)")
                self.cells[(i, k)] = Cell([left[0], *right, *left[:0:-1]], tris)

    def index(self, s: float) -> int:
        best = min(range(len(self.sl)), key=lambda i: abs(self.sl[i].s - s))
        if abs(self.sl[best].s - s) > STATION_MERGE_M * 1.01:
            raise ValueError(f"road {self.road.id}: s={s:.3f} is not a sweep station")
        return best

    def cells_between(self, s_a: float, s_b: float, key_lo: str, key_hi: str) -> list[Cell]:
        out: list[Cell] = []
        for i in range(self.index(s_a), self.index(s_b)):
            try:
                p_lo, p_hi = self.key_pos[(i, key_lo)], self.key_pos[(i, key_hi)]
            except KeyError as exc:
                raise ValueError(f"road {self.road.id}: u-line {exc.args[0][1]} inactive at s={self.sl[i].s:.2f}") from exc
            if p_hi < p_lo:
                raise ValueError(f"road {self.road.id}: {key_lo} lies left of {key_hi} at s={self.sl[i].s:.2f}")
            out.extend(self.cells[(i, k)] for k in range(p_lo, p_hi))
        return out
