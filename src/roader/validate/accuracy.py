"""Accuracy targets AC-A1, AC-A3, AC-A4, AC-A5 (spec section 9) and the holdout split.

These functions only measure: they compare the model with values the pipeline
measured and never refit anything. AC-A1 uses lidar returns withheld from
fitting in 25 m spatial blocks covering about 20% of pavement area; the fitting
stages must exclude ``holdout_mask(xy)`` so both sides use the same split.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from roader.model.geometry import Piecewise
from roader.model.schema import Road, RoadModel
from roader.validate._align import Projector
from roader.validate.report import GateResult, Violation

BLOCK_M = 25.0
HOLDOUT_FRACTION = 0.2
HOLDOUT_SALT = "roader:AC-A1:holdout:v1"
DECK_EDGE_TOL_M = 1.0  # AC-A1: deck-outline returns beyond the pavement must be this close to the deck surface


# ---------------------------------------------------------------- holdout


def _block_rank(bx: int, by: int, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{bx}:{by}".encode()).digest()[:8], "big")


def holdout_mask(
    xy: np.ndarray, *, block_m: float = BLOCK_M, fraction: float = HOLDOUT_FRACTION, salt: str = HOLDOUT_SALT
) -> np.ndarray:
    """True for points in held-out blocks.

    Blocks are ``block_m`` squares on the unit-frame grid. They are taken in the
    order of a hash of their indices until ``fraction`` of the pavement area
    (occupied 1 m cells) is held out, so the split depends only on where the
    pavement is, never on point order or the run.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    blocks = np.floor(xy / block_m).astype(np.int64)
    cells = np.floor(xy).astype(np.int64)
    ub, inverse = np.unique(blocks, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    occupied = np.unique(np.column_stack([inverse, cells]), axis=0)
    area = np.bincount(occupied[:, 0], minlength=len(ub)).astype(np.float64)
    rank = np.array([_block_rank(int(bx), int(by), salt) for bx, by in ub], dtype=np.uint64)
    order = np.lexsort((ub[:, 1], ub[:, 0], rank))
    before = np.cumsum(area[order]) - area[order]
    held = np.zeros(len(ub), dtype=bool)
    held[order[before < fraction * area.sum()]] = True
    return held[inverse]


# ---------------------------------------------------------------- helpers


def _pw(piece: Piecewise, s: np.ndarray) -> np.ndarray:
    """Vectorized Piecewise: linear between knots, constant beyond the ends."""
    ks = np.array([k.s for k in piece.knots])
    kv = np.array([k.value for k in piece.knots])
    return np.interp(s, ks, kv)


def _pw_max(piece: Piecewise) -> float:
    return max(k.value for k in piece.knots)


def _skipped(gate: str, reason: str) -> GateResult:
    return GateResult(gate, passed=False, hard=False, skipped_reason=reason)


# ---------------------------------------------------------------- AC-A1


def surface_accuracy(
    model: RoadModel,
    xyz: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    gate: str = "AC-A1",
    rms_max: float = 0.10,
    p95_max: float = 0.20,
) -> GateResult:
    """Vertical residuals of held-out pavement and deck returns against the model surface.

    ``xyz`` are pavement/deck points (the evidence stage's labels). ``mask``
    defaults to ``holdout_mask``. Each point is projected to (s, u) on every road
    whose pavement it falls within (-width_right <= u <= width_left, strictly
    inside the station range); where roads overlap in plan the road with the
    smallest residual is taken, which assigns stacked points to their own layer.
    Use ``gate="AC-A2", rms_max=0.20`` for QL3 or pre-2015 data.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if mask is None:
        mask = holdout_mask(xyz[:, :2])
    pts = xyz[np.asarray(mask, dtype=bool)]
    if len(pts) == 0:
        return _skipped(gate, "no held-out pavement points")
    best = np.full(len(pts), np.inf)
    for road in model.roads:
        decks = [(st.s_start, st.s_end, st.deck.overhang_m) for st in model.structures if st.carried_road == road.id]
        _assign_residuals(road, pts, best, decks)
    used = np.isfinite(best)
    if not used.any():
        return _skipped(gate, "no held-out points fall on any road's pavement")
    res = best[used]
    rms = float(np.sqrt(np.mean(res**2)))
    p95 = float(np.percentile(np.abs(res), 95.0))
    result = GateResult(
        gate,
        passed=rms <= rms_max and p95 <= p95_max,
        hard=False,
        metrics={
            "rms_m": rms,
            "p95_m": p95,
            "bias_m": float(np.mean(res)),
            "n_points": float(used.sum()),
            "n_unassigned": float((~used).sum()),
        },
    )
    if not result.passed:
        result.violations.append(Violation(gate, model.unit.unit_id, f"vertical RMS {rms:.3f} m (<= {rms_max}), p95 {p95:.3f} m (<= {p95_max})"))
    return result


def _assign_residuals(road: Road, pts: np.ndarray, best: np.ndarray, decks: list[tuple[float, float, float]] = ()) -> None:
    """Residuals against this road for points inside its footprint; keeps the smallest per point.

    Over a structure the footprint is the deck outline (pavement plus overhang), so
    returns on the deck beyond the curb (walkways, curb tops) are compared with the
    deck they lie on, not with the road under the deck (AC-A1).
    """
    xs = road.cross_section
    proj = Projector(road.alignment)
    reach = max(_pw_max(xs.width_left), _pw_max(xs.width_right)) + max((d[2] for d in decks), default=0.0)
    idx = np.nonzero(proj.near(pts[:, :2], reach))[0]
    if len(idx) == 0:
        return
    s, u = proj.project(pts[idx, :2])
    extra = np.zeros(len(s))
    for s0, s1, overhang in decks:
        extra = np.where((s >= s0) & (s <= s1), np.maximum(extra, overhang), extra)
    wl, wr = _pw(xs.width_left, s), _pw(xs.width_right, s)
    inside = (s > 0.0) & (s < proj.length) & (u <= wl + extra) & (u >= -wr - extra)
    on_pavement = (u <= wl) & (u >= -wr)
    idx, s, u = idx[inside], s[inside], u[inside]
    on_pavement = on_pavement[inside]
    z = np.array([road.surface_z(float(a), float(b)) for a, b in zip(s, u, strict=True)])
    res = pts[idx, 2] - z
    # Beyond the pavement only returns at deck level belong to the deck; ground seen past its edge does not.
    res = np.where(on_pavement | (np.abs(res) <= DECK_EDGE_TOL_M), res, np.inf)
    better = np.abs(res) < np.abs(best[idx])
    best[idx[better]] = res[better]


# ---------------------------------------------------------------- AC-A3


def horizontal_accuracy(model: RoadModel, edges: Iterable[Mapping], *, rms_max: float = 0.5) -> GateResult:
    """Pavement-edge and deck-outline points against the model edges, measured across the road.

    Each entry has ``x``, ``y``, ``side`` ("left" | "right") and either
    ``road_id`` (pavement edge at width_left / width_right) or ``structure_id``
    (deck edge: pavement width plus deck overhang on the carried road).
    """
    gate = "AC-A3"
    groups: dict[tuple[str, float], list[tuple[float, float, float]]] = defaultdict(list)
    for e in edges:
        if e.get("structure_id"):
            st = model.structure(e["structure_id"])
            road_id, extra = st.carried_road, st.deck.overhang_m
        else:
            road_id, extra = e["road_id"], 0.0
        sign = 1.0 if e["side"] == "left" else -1.0
        groups[(road_id, extra)].append((float(e["x"]), float(e["y"]), sign))
    if not groups:
        return _skipped(gate, "no measured edge points")
    residuals = []
    for (road_id, extra), rows in sorted(groups.items()):
        road = model.road(road_id)
        arr = np.array(rows)
        s, u = Projector(road.alignment).project(arr[:, :2])
        sign = arr[:, 2]
        width = np.where(sign > 0, _pw(road.cross_section.width_left, s), _pw(road.cross_section.width_right, s))
        residuals.append(sign * u - width - extra)
    res = np.concatenate(residuals)
    rms = float(np.sqrt(np.mean(res**2)))
    result = GateResult(gate, passed=rms <= rms_max, hard=False, metrics={"rms_m": rms, "bias_m": float(res.mean()), "n_points": float(len(res))})
    if not result.passed:
        result.violations.append(Violation(gate, model.unit.unit_id, f"horizontal edge RMS {rms:.3f} m (<= {rms_max})"))
    return result


# ---------------------------------------------------------------- AC-A4, AC-A5


def model_crossing_dz(model: RoadModel) -> dict[str, float]:
    """Upper minus lower model surface at each crossing point."""
    out = {}
    for c in model.crossings:
        heights = []
        for road_id in (c.upper_road, c.lower_road):
            road = model.road(road_id)
            s, u = road.alignment.project(c.x, c.y)
            heights.append(road.surface_z(s, u))
        out[c.id] = heights[0] - heights[1]
    return out


def crossing_dz_accuracy(model: RoadModel, measured: Mapping[str, float], *, tol: float = 0.15) -> GateResult:
    """AC-A4: model surface-to-surface height difference within ``tol`` of the measured one."""
    gate = "AC-A4"
    if not measured:
        return _skipped(gate, "no measured crossing height differences")
    model_dz = model_crossing_dz(model)
    result = GateResult(gate, passed=True, hard=False)
    errors = []
    for cid, dz in sorted(measured.items()):
        if cid not in model_dz:
            result.violations.append(Violation(gate, cid, "measured crossing not in the model"))
            continue
        err = model_dz[cid] - float(dz)
        errors.append(abs(err))
        if abs(err) > tol:
            result.violations.append(Violation(gate, cid, f"dz model {model_dz[cid]:.3f} vs measured {float(dz):.3f} m (|err| > {tol})"))
    result.metrics = {
        "max_abs_err_m": max(errors) if errors else math.nan,
        "n_crossings": float(len(errors)),
        "n_unmeasured": float(len(set(model_dz) - set(measured))),
    }
    result.passed = not result.violations
    return result


def structure_station_accuracy(
    model: RoadModel, measured: Mapping[str, tuple[float, float]], *, tol: float = 2.0
) -> GateResult:
    """AC-A5: structure start and end stations within ``tol`` of the measured ones."""
    gate = "AC-A5"
    if not measured:
        return _skipped(gate, "no measured structure stations")
    result = GateResult(gate, passed=True, hard=False)
    errors = []
    for sid, (s0, s1) in sorted(measured.items()):
        try:
            st = model.structure(sid)
        except KeyError:
            result.violations.append(Violation(gate, sid, "measured structure not in the model"))
            continue
        for name, model_s, meas_s in (("start", st.s_start, s0), ("end", st.s_end, s1)):
            err = model_s - float(meas_s)
            errors.append(abs(err))
            if abs(err) > tol:
                result.violations.append(Violation(gate, sid, f"{name} station model {model_s:.2f} vs measured {float(meas_s):.2f} (|err| > {tol} m)"))
    result.metrics = {"max_abs_err_m": max(errors) if errors else math.nan, "n_structures": float(len(measured))}
    result.passed = not result.violations
    return result
