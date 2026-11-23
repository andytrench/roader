"""L1 horizontal alignment, L2 vertical profile, L3 piecewise cross-section values.

Stations ``s`` are meters along the reference line from its start. Lateral offset
``u`` is meters, positive to the left of the direction of travel. Headings are
radians counter-clockwise from the +x (east) axis. Curvature is 1/m, positive
for a left turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from roader.model.provenance import Provenance

# Gauss-Legendre nodes for spiral integration. Fixed, so evaluation is deterministic.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


@dataclass
class AlignElement:
    """One L1 element. line: k0 = k1 = 0. arc: k0 = k1 != 0. spiral: k varies linearly."""

    kind: str  # "line" | "arc" | "spiral"
    s0: float
    length: float
    x0: float
    y0: float
    heading0: float
    k0: float
    k1: float
    prov: Provenance

    def __post_init__(self) -> None:
        if self.kind not in {"line", "arc", "spiral"}:
            raise ValueError(f"unknown element kind {self.kind}")
        if self.length <= 0.0:
            raise ValueError("element length must be positive")

    def curvature(self, t: float) -> float:
        return self.k0 + (self.k1 - self.k0) * t / self.length

    def heading(self, t: float) -> float:
        return self.heading0 + self.k0 * t + 0.5 * (self.k1 - self.k0) * t * t / self.length

    def point(self, t: float) -> tuple[float, float]:
        if self.kind == "line" or (abs(self.k0) < 1e-15 and abs(self.k1) < 1e-15):
            return (
                self.x0 + t * math.cos(self.heading0),
                self.y0 + t * math.sin(self.heading0),
            )
        if self.kind == "arc":
            k = self.k0
            h1 = self.heading0 + k * t
            return (
                self.x0 + (math.sin(h1) - math.sin(self.heading0)) / k,
                self.y0 - (math.cos(h1) - math.cos(self.heading0)) / k,
            )
        # Spiral: integrate cos/sin of heading over [0, t], in pieces of at most 10 m.
        pieces = max(1, math.ceil(t / 10.0))
        x, y = self.x0, self.y0
        for i in range(pieces):
            a = t * i / pieces
            b = t * (i + 1) / pieces
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            for node, weight in zip(_GL_X, _GL_W, strict=True):
                h = self.heading(mid + half * float(node))
                x += half * float(weight) * math.cos(h)
                y += half * float(weight) * math.sin(h)
        return x, y

    def end_state(self) -> tuple[float, float, float, float]:
        x, y = self.point(self.length)
        return x, y, self.heading(self.length), self.k1


@dataclass
class Alignment:
    elements: list[AlignElement] = field(default_factory=list)

    @property
    def length(self) -> float:
        if not self.elements:
            return 0.0
        last = self.elements[-1]
        return last.s0 + last.length

    def _element_at(self, s: float) -> tuple[AlignElement, float]:
        if not self.elements:
            raise ValueError("empty alignment")
        s = min(max(s, 0.0), self.length)
        for element in self.elements:
            if s <= element.s0 + element.length:
                return element, s - element.s0
        last = self.elements[-1]
        return last, last.length

    def evaluate(self, s: float) -> tuple[float, float, float, float]:
        """(x, y, heading, curvature) at station s, clamped to the alignment."""
        element, t = self._element_at(s)
        x, y = element.point(t)
        return x, y, element.heading(t), element.curvature(t)

    def offset_point(self, s: float, u: float) -> tuple[float, float]:
        x, y, heading, _k = self.evaluate(s)
        return x - u * math.sin(heading), y + u * math.cos(heading)

    def _samples(self, step: float):
        """(stations, KD-tree) of plan samples every ``step`` m, cached until an element changes (D-125)."""
        from scipy.spatial import cKDTree

        key = (step, tuple((id(e), e.s0, e.length, e.x0, e.y0, e.heading0, e.k0, e.k1) for e in self.elements))
        cached = self.__dict__.get("_proj_cache")
        if cached is None or cached[0] != key:
            length = self.length
            count = max(2, math.ceil(length / step) + 1)
            stations = np.array([length * i / (count - 1) for i in range(count)])
            tree = cKDTree(np.array([self.evaluate(float(s))[:2] for s in stations]))
            cached = (key, stations, tree)
            self.__dict__["_proj_cache"] = cached
        return cached[1], cached[2]

    def _refine(self, x: float, y: float, s: float) -> tuple[float, float]:
        length = self.length
        for _ in range(8):
            px, py, heading, k = self.evaluate(s)
            tx, ty = math.cos(heading), math.sin(heading)
            dx, dy = x - px, y - py
            along = dx * tx + dy * ty
            lateral = -dx * ty + dy * tx
            denom = 1.0 - k * lateral
            if abs(denom) < 1e-6:
                break
            ds = along / denom
            s = min(max(s + ds, 0.0), length)
            if abs(ds) < 1e-9:
                break
        px, py, heading, _k = self.evaluate(s)
        return s, -(x - px) * math.sin(heading) + (y - py) * math.cos(heading)

    def project(self, x: float, y: float, step: float = 1.0) -> tuple[float, float]:
        """Nearest station and signed offset (u positive left) of a plan point.

        Stations are clamped to [0, length]; a point beyond an end gets the end
        station and its distance along the end normal. The nearest of the samples
        every ``step`` m seeds Newton steps; the samples are cached on the alignment
        (D-125), so repeated projections do not rescan the road.
        """
        stations, tree = self._samples(step)
        _d, i = tree.query((x, y))
        return self._refine(x, y, float(stations[int(i)]))

    def project_many(self, xy, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """``project`` for an (N, 2) array of points: (stations, offsets)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        stations, tree = self._samples(step)
        _d, idx = tree.query(xy)
        s_out, u_out = np.empty(len(xy)), np.empty(len(xy))
        for k in range(len(xy)):
            s_out[k], u_out[k] = self._refine(float(xy[k, 0]), float(xy[k, 1]), float(stations[int(idx[k])]))
        return s_out, u_out

    def near(self, xy, max_dist: float, step: float = 1.0) -> np.ndarray:
        """Points whose nearest sample is within ``max_dist`` plus one sample spacing."""
        stations, tree = self._samples(step)
        spacing = float(stations[1] - stations[0])
        d, _ = tree.query(np.asarray(xy, dtype=np.float64).reshape(-1, 2), distance_upper_bound=max_dist + spacing)
        return np.isfinite(d)


@dataclass
class VerticalElement:
    """One L2 element. Grade g0 at the start, g1 at the end; g0 == g1 is a constant grade.

    z(t) = z0 + g0 t + (g1 - g0) t^2 / (2 length), a parabolic vertical curve.
    """

    s0: float
    length: float
    z0: float
    g0: float
    g1: float
    prov: Provenance

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("element length must be positive")

    def z(self, t: float) -> float:
        return self.z0 + self.g0 * t + 0.5 * (self.g1 - self.g0) * t * t / self.length

    def grade(self, t: float) -> float:
        return self.g0 + (self.g1 - self.g0) * t / self.length


@dataclass
class Profile:
    elements: list[VerticalElement] = field(default_factory=list)

    @property
    def length(self) -> float:
        if not self.elements:
            return 0.0
        last = self.elements[-1]
        return last.s0 + last.length

    def _element_at(self, s: float) -> tuple[VerticalElement, float]:
        if not self.elements:
            raise ValueError("empty profile")
        for element in self.elements:
            if s <= element.s0 + element.length:
                return element, s - element.s0
        last = self.elements[-1]
        return last, s - last.s0

    def z(self, s: float) -> float:
        """Elevation; beyond either end the end grade continues as a straight line."""
        element, t = self._element_at(s)
        inside = min(max(t, 0.0), element.length)
        return element.z(inside) + element.grade(inside) * (t - inside)

    def grade(self, s: float) -> float:
        element, t = self._element_at(s)
        return element.grade(min(max(t, 0.0), element.length))


@dataclass
class Knot:
    s: float
    value: float
    prov: Provenance


@dataclass
class Piecewise:
    """Piecewise-linear value along the road; constant beyond the end knots.

    A change between two knots at distinct stations is a taper or transition of
    nonzero length (AC-P2). Two knots at the same station would be a step and are
    rejected.
    """

    knots: list[Knot] = field(default_factory=list)

    def __post_init__(self) -> None:
        for left, right in zip(self.knots, self.knots[1:], strict=False):
            if right.s <= left.s:
                raise ValueError("knot stations must strictly increase (AC-P2)")

    def __call__(self, s: float) -> float:
        if not self.knots:
            raise ValueError("empty piecewise value")
        if s <= self.knots[0].s:
            return self.knots[0].value
        for left, right in zip(self.knots, self.knots[1:], strict=False):
            if s <= right.s:
                f = (s - left.s) / (right.s - left.s)
                return left.value + f * (right.value - left.value)
        return self.knots[-1].value

    def breakpoints(self) -> list[float]:
        return [knot.s for knot in self.knots]


def constant(value: float, prov: Provenance, s: float = 0.0) -> Piecewise:
    return Piecewise([Knot(s, value, prov)])
