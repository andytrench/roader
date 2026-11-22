"""Vectorized station/offset geometry on densified plan lines.

Stages 3 to 9 project hundreds of thousands of returns onto road lines. A line
(an input centerline, or an L1 alignment sampled finely) is kept as a dense
polyline with stations; projection is a KD-tree lookup of the nearest vertex
followed by an exact projection onto the two adjacent segments. At 0.5 m vertex
spacing the chord error on a 300 m radius is 0.1 mm.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial import cKDTree

from roader.model.geometry import Alignment


class Polyline:
    """Plan polyline with stations; u is positive to the left of the direction of travel."""

    def __init__(self, xy: np.ndarray, step: float = 0.5) -> None:
        xy = _dedupe(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        if len(xy) < 2:
            raise ValueError("polyline needs two distinct points")
        self.xy = _densify(xy, step)
        seg = np.diff(self.xy, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.tangent = seg / self.seg_len[:, None]
        self.tree = cKDTree(self.xy)

    @classmethod
    def from_alignment(cls, alignment: Alignment, step: float = 0.5) -> Polyline:
        n = max(2, math.ceil(alignment.length / step) + 1)
        stations = np.linspace(0.0, alignment.length, n)
        line = cls.__new__(cls)
        line.xy = np.array([alignment.evaluate(float(s))[:2] for s in stations])
        seg = np.diff(line.xy, axis=0)
        line.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        # Stations are the alignment's own, not the chord sums, so s matches Alignment.evaluate.
        line.s = stations
        line.tangent = seg / line.seg_len[:, None]
        line.tree = cKDTree(line.xy)
        return line

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def point(self, s: np.ndarray | float) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        return np.column_stack([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])

    def heading(self, s: np.ndarray | float) -> np.ndarray:
        s = np.clip(np.atleast_1d(np.asarray(s, dtype=np.float64)), 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_len) - 1)
        return np.arctan2(self.tangent[i, 1], self.tangent[i, 0])

    def normal(self, s: np.ndarray | float) -> np.ndarray:
        h = self.heading(s)
        return np.column_stack([-np.sin(h), np.cos(h)])

    def offset_point(self, s: np.ndarray | float, u: np.ndarray | float) -> np.ndarray:
        return self.point(s) + np.asarray(u, dtype=np.float64).reshape(-1, 1) * self.normal(s)

    def project(self, xy: np.ndarray, max_dist: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(s, u, ok) per point. ok is False beyond max_dist or past either end of the line."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        bound = np.inf if max_dist is None else max_dist + float(self.seg_len.max())
        d, k = self.tree.query(xy, distance_upper_bound=bound)
        found = np.isfinite(d)
        s_out = np.full(len(xy), np.nan)
        u_out = np.full(len(xy), np.nan)
        if not found.any():
            return s_out, u_out, found
        idx = np.flatnonzero(found)
        k = k[found]
        p = xy[idx]
        best_d = np.full(len(idx), np.inf)
        best_s = np.zeros(len(idx))
        best_u = np.zeros(len(idx))
        nseg = len(self.seg_len)
        for seg in (k - 1, k):
            valid = (seg >= 0) & (seg < nseg)
            j = np.clip(seg, 0, nseg - 1)
            a = self.xy[j]
            t = self.tangent[j]
            rel = p - a
            along = np.einsum("ij,ij->i", rel, t)
            lat = t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0]
            clamped = np.clip(along, 0.0, self.seg_len[j])
            # Points past the line ends keep their along-distance so they can be rejected.
            first, last = j == 0, j == nseg - 1
            clamped = np.where(first & (along < 0.0), along, clamped)
            clamped = np.where(last & (along > self.seg_len[j]), along, clamped)
            dist = np.hypot(along - clamped, lat)
            better = valid & (dist < best_d)
            best_d = np.where(better, dist, best_d)
            best_s = np.where(better, self.s[j] + clamped, best_s)
            best_u = np.where(better, lat, best_u)
        s_out[idx] = best_s
        u_out[idx] = best_u
        ok = found.copy()
        ok[idx] = (best_s >= 0.0) & (best_s <= self.length)
        if max_dist is not None:
            ok[idx] &= np.abs(best_u) <= max_dist
        return s_out, u_out, ok

    def shifted(self, s: np.ndarray, offset: np.ndarray, step: float = 0.5) -> Polyline:
        """New line displaced to the left by offset(s), sampled at stations s."""
        pts = self.offset_point(s, offset)
        return Polyline(pts, step=step)

    def clipped(self, s0: float, s1: float, step: float = 0.5) -> Polyline:
        inner = (self.s > s0) & (self.s < s1)
        pts = np.vstack([self.point(s0), self.xy[inner], self.point(s1)])
        return Polyline(pts, step=step)


def _dedupe(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 2:
        return xy
    keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-6])
    return xy[keep]


def _densify(xy: np.ndarray, step: float) -> np.ndarray:
    out = [xy[:1]]
    for a, b in itertools.pairwise(xy):
        n = max(1, math.ceil(math.hypot(*(b - a)) / step))
        t = np.arange(1, n + 1)[:, None] / n
        out.append(a + t * (b - a))
    return np.vstack(out)


def resample(xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """(stations, points) at uniform arclength spacing, ends included."""
    line = Polyline(xy, step=min(step, 0.5))
    n = max(2, math.ceil(line.length / step) + 1)
    s = np.linspace(0.0, line.length, n)
    return s, line.point(s)
