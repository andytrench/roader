"""Supports carry the skew the sweep can hold (FR-ST-4, FR-SG-2, D-171).

The solids square a structure's skew off toward a cut road end (D-118) and limit it
near steep embankment toes (D-161). Supports laid out at the full skew are then swept
at less, so a pier or an abutment can turn onto the road below. The skew the sweep
holds is read from the solids themselves (``swept_lines``) on a trial model, and the
structures are laid out at it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from roader.model.schema import RoadModel
from roader.solids import swept_lines
from roader.solids.terrain import TerrainSampler

HOLD_TOL_M = 0.01  # the analytic chord tolerance, at which structures are always swept
CAP_MIN = 0.01  # a support whose |tan skew| drops by less than this keeps it


def held_skew(model: RoadModel) -> dict[str, Callable[[float], float]]:
    """Per road carrying a skewed structure: |tan skew| the sweep holds at station s.

    A road the solids cannot sweep at its skew is squared by them, so it holds 0.
    """
    sampler = TerrainSampler(model.terrain) if model.terrain is not None else None
    out: dict[str, Callable[[float], float]] = {}
    for road in model.roads:
        if not any(st.carried_road == road.id and any(sp.skew_rad != 0.0 for sp in st.supports) for st in model.structures):
            continue
        lines = swept_lines(model, road, sampler, HOLD_TOL_M)
        if lines is None:
            out[road.id] = lambda _s: 0.0
            continue
        s = np.array([ln.s for ln in lines])
        t = np.abs([ln.t for ln in lines])
        out[road.id] = lambda v, s=s, t=t: float(np.interp(v, s, t))
    return out


def capped(t: float, hold: Callable[[float], float] | None, s: float) -> float:
    """``t`` (tan skew) no steeper than the sweep holds at station ``s``."""
    if hold is None:
        return t
    return math.copysign(min(abs(t), hold(s)), t)
