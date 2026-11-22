"""L4 side slopes (FR-OC-3).

Side slopes are measured from the ground just beyond each pavement edge and
grouped into runs of one kind (fill, cut, wall, none). Where the ground beside a
road is not observed (under a deck), the neighbouring run is carried through
and tagged; over a structure the carried road has no side slope. Cuts and walls
are written as ``none`` with the measured kind and ratio in the provenance
flags, because milestone 1 solids render fills only.
"""

from __future__ import annotations

import math

import numpy as np

from roader.fit.road import RoadFit, confidence
from roader.model.provenance import Provenance, Source
from roader.model.schema import SideSlope, Structure
from roader.model.serialize import stable_id
from roader.rules import RuleBook

STEP_M = 10.0
WINDOW_M = 5.0
BAND_INNER_M = 0.5
BAND_OUTER_M = 3.0
FLAT_SLOPE = 1.0 / 20.0  # gentler than 1:20 is no side slope
WALL_RATIO = 0.5  # steeper than 1:0.5 (h:v) is a wall
MIN_RUN_M = 20.0
MERGE_RATIO_REL = 0.15
# A measured fill gentler than this times the rule's flattest fill is the natural ground, not an embankment.
GENTLE_FACTOR = 1.5


def _station_slopes(fit: RoadFit, side: str, skip: list[tuple[float, float]]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    road = fit.road
    lp = fit.points
    piece = road.cross_section.width_left if side == "left" else road.cross_section.width_right
    sign = 1.0 if side == "left" else -1.0
    st = np.arange(0.5 * STEP_M, road.length, STEP_M)
    kinds, ratios, counts = [], np.full(len(st), np.nan), np.zeros(len(st), dtype=np.int64)
    w = np.interp(lp.s, [k.s for k in piece.knots], [k.value for k in piece.knots])
    d = sign * lp.u - w
    band = (d >= BAND_INNER_M) & (d <= BAND_OUTER_M)
    sb, db, zb = lp.s[band], d[band], lp.z[band]
    for i, s in enumerate(st):
        if any(a - 1.0 <= s <= b + 1.0 for a, b in skip):
            kinds.append("skip")
            continue
        m = np.abs(sb - s) <= WINDOW_M
        if m.sum() < 15:
            kinds.append("unobserved")
            continue
        edge_z = road.surface_z(float(s), sign * float(piece(float(s))))
        # Lowest returns per 0.5 m of distance: ground under grass and shrubs.
        bins = np.floor(db[m] / 0.5).astype(np.int64)
        xs, ys = [], []
        for b in np.unique(bins):
            sel = bins == b
            xs.append(float(np.mean(db[m][sel])))
            ys.append(float(np.percentile(zb[m][sel], 10.0)) - edge_z)
        if len(xs) < 3:
            kinds.append("unobserved")
            continue
        slope = float(np.polyfit(xs, ys, 1)[0])
        counts[i] = int(m.sum())
        if abs(slope) < FLAT_SLOPE:
            kinds.append("none")
            ratios[i] = 0.0
        else:
            ratio = 1.0 / abs(slope)
            kinds.append("wall" if ratio < WALL_RATIO else ("fill" if slope < 0 else "cut"))
            ratios[i] = ratio
    return st, kinds, ratios, counts


def side_slopes(fit: RoadFit, structures: list[Structure], rules: RuleBook, strip_factor: float = 1.0) -> list[SideSlope]:
    road = fit.road
    skip = [(st.s_start, st.s_end) for st in structures if st.carried_road == road.id]
    out = []
    for side in ("left", "right"):
        st, kinds, ratios, counts = _station_slopes(fit, side, skip)
        # Unobserved stations take the kind of the previous observed one (carried through, FR-OC-3).
        filled = list(kinds)
        last = None
        for i, k in enumerate(filled):
            if k in ("fill", "cut", "wall", "none"):
                last = i
            elif k == "unobserved" and last is not None:
                filled[i] = "carried:" + filled[last]
        first_obs = next((i for i, k in enumerate(kinds) if k in ("fill", "cut", "wall", "none")), None)
        for i in range(len(filled)):
            if filled[i] == "unobserved" and first_obs is not None:
                filled[i] = "carried:" + kinds[first_obs]
        base = [k.split(":")[-1] for k in filled]
        runs = []
        for i, k in enumerate(base):
            if k in ("skip", "unobserved"):
                continue
            s0 = max(0.0, st[i] - 0.5 * STEP_M)
            s1 = min(road.length, st[i] + 0.5 * STEP_M)
            if runs and runs[-1][2] == k and abs(runs[-1][1] - s0) < 1e-6:
                runs[-1][1] = s1
                runs[-1][3].append(i)
            else:
                runs.append([s0, s1, k, [i]])
        # Absorb short runs into their longer neighbour.
        merged: list[list] = []
        for run in runs:
            if merged and (run[1] - run[0] < MIN_RUN_M) and abs(merged[-1][1] - run[0]) < 1e-6:
                merged[-1][1] = run[1]
                merged[-1][3].extend(run[3])
            else:
                merged.append(run)
        for s0, s1, kind, idx in merged:
            vals = ratios[idx]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            observed = [i for i in idx if not filled[i].startswith("carried")]
            obs_frac = len(observed) / max(len(idx), 1)
            if kind == "fill" and vals.size and float(np.median(vals)) > rules.get("slope.fill_h_per_v").high * GENTLE_FACTOR:
                kind = "gentle"
            if kind in ("cut", "wall", "gentle"):
                # Milestone 1 solids render fills only; the measured cut or wall is kept in the flags.
                prov = Provenance(Source.MEASURED, confidence(obs_frac, None, strip_factor), n_points=int(counts[idx].sum()), observed_fraction=obs_frac)
                label = "fill_gentler_than_rule" if kind == "gentle" else kind
                prov.flags.append(f"L4:measured_{label}:ratio={float(np.median(vals)) if vals.size else 0.0:.2f}")
                out.append(SideSlope(stable_id("ss", road.id, side, round(s0, 0)), road.id, side, float(s0), float(s1), "none", 0.0, prov))
                continue
            if kind == "none" or vals.size == 0:
                ratio = 0.0 if kind == "none" else float("nan")
            else:
                ratio = float(np.median(vals))
            if kind in ("fill", "cut") and not math.isfinite(ratio):
                ratio, prov = rules.fill(f"slope.{kind}_h_per_v")
            else:
                prov = Provenance(Source.MEASURED, confidence(obs_frac, None, strip_factor), n_points=int(counts[idx].sum()), observed_fraction=obs_frac)
                if obs_frac < 1.0:
                    prov.flags.append("FR-OC-3:carried")
                if kind in ("fill", "cut"):
                    ok, rule = rules.evaluate(f"slope.{kind}_h_per_v", ratio)
                    if not ok:
                        prov.flags.append(f"FR-RL-3:out_of_range:{rule.id}")
            out.append(SideSlope(stable_id("ss", road.id, side, round(s0, 0)), road.id, side, float(s0), float(s1), kind, float(ratio), prov))
    return _merge_similar(out)


def _merge_similar(items: list[SideSlope]) -> list[SideSlope]:
    """Join touching runs of one kind on one side whose ratios differ by less than MERGE_RATIO_REL.

    Two fills meeting at one station with nearly equal ratios would leave their toes
    a fraction of a millimetre apart, a sliver no solid should carry (FR-SV-5).
    """
    items = sorted(items, key=lambda s: (s.road_id, s.side, s.s0))
    out: list[SideSlope] = []
    for ss in items:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.road_id == ss.road_id
            and prev.side == ss.side
            and prev.kind == ss.kind
            and abs(prev.s1 - ss.s0) < 1e-6
            and abs(prev.ratio - ss.ratio) <= MERGE_RATIO_REL * max(prev.ratio, ss.ratio, 1e-9)
        ):
            la, lb = prev.s1 - prev.s0, ss.s1 - ss.s0
            prev.ratio = (prev.ratio * la + ss.ratio * lb) / (la + lb)
            prev.s1 = ss.s1
            prev.prov.n_points += ss.prov.n_points
            continue
        out.append(ss)
    for ss in out:
        ss.ratio = round(ss.ratio, 3)
    return out
