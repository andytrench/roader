"""Many-point station/offset projection, the batch form of ``Alignment.project``.

``Alignment.project`` scans the whole alignment for every point, which is too
slow for tens of thousands of lidar returns. Here the coarse scan is one KD-tree
query over the same 1 m samples, followed by the same Newton refinement.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from roader.model.geometry import Alignment


class Projector:
    def __init__(self, alignment: Alignment, step: float = 1.0) -> None:
        self.alignment = alignment
        self.length = alignment.length
        count = max(2, math.ceil(self.length / step) + 1)
        self.stations = np.linspace(0.0, self.length, count)
        self.samples = np.array([alignment.evaluate(float(s))[:2] for s in self.stations])
        self.tree = cKDTree(self.samples)
        self.step = self.length / (count - 1)

    def near(self, xy: np.ndarray, max_dist: float) -> np.ndarray:
        """Points whose nearest sample is within max_dist + one sample spacing."""
        d, _ = self.tree.query(xy, distance_upper_bound=max_dist + self.step)
        return np.isfinite(d)

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Station and signed offset (u positive left) for each point."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        _d, idx = self.tree.query(xy)
        s_out = np.empty(len(xy))
        u_out = np.empty(len(xy))
        evaluate = self.alignment.evaluate
        length = self.length
        for k in range(len(xy)):
            x, y = float(xy[k, 0]), float(xy[k, 1])
            s = float(self.stations[idx[k]])
            for _ in range(8):
                px, py, heading, curv = evaluate(s)
                tx, ty = math.cos(heading), math.sin(heading)
                dx, dy = x - px, y - py
                along = dx * tx + dy * ty
                lateral = -dx * ty + dy * tx
                denom = 1.0 - curv * lateral
                if abs(denom) < 1e-6:
                    break
                ds = along / denom
                s = min(max(s + ds, 0.0), length)
                if abs(ds) < 1e-9:
                    break
            px, py, heading, _k = evaluate(s)
            s_out[k] = s
            u_out[k] = -(x - px) * math.sin(heading) + (y - py) * math.cos(heading)
        return s_out, u_out
