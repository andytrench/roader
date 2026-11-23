"""Crossing resolution (FR-CR-1, FR-CR-2).

Every plan intersection of two fitted roads becomes a Crossing. Which road is
above, and by how much, comes from the fitted surfaces, which are measured on
both roads; the lower road's surface under a deck is carried through from both
sides (FR-OC-1). A direct data estimate of the height difference is returned
beside the model (upper returns near the crossing point, lower returns 15 m
either side along the lower road, interpolated), for AC-A4 and the report.
Vector layer / bridge / tunnel tags are logged, and a disagreement with the
measured order is flagged with both values. Lines meeting on one level while
running side by side share pavement and are not a crossing (D-135).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree

from roader.conflate.graph import RoadSeed
from roader.fit.road import RoadFit
from roader.model.provenance import Provenance, Source
from roader.model.schema import Crossing, Flag
from roader.model.serialize import stable_id

UPPER_RADIUS_M = 3.0
LOWER_OFFSET_M = 15.0
LOWER_RADIUS_M = 1.5
SURFACE_TOL_M = 0.3
SHARED_NODE_M = 0.05  # vertices of two inputs this close are one node
NODE_REACH_M = 1.0  # an input intersection this close to a shared node is that node
MATCH_M = 30.0  # a fitted intersection this close to an input crossing locates it
SHARED_DZ_M = 1.0  # two surfaces this close in height at an intersection are one level
SHARED_ANGLE = math.radians(20.0)  # lines this close to parallel there run side by side (D-135)


def shares_pavement(dz: float, h_a: float, h_b: float) -> bool:
    """An intersection of two lines on one level and running side by side is shared pavement, not a crossing (D-135).

    Two roads whose pavements overlap over a stretch (a fork whose branches share
    their first metres, a C-D road abutting its mainline) have centrelines that
    weave across each other within the noise; they never pass over or under.
    """
    return dz < SHARED_DZ_M and abs(math.remainder(h_a - h_b, math.pi)) < SHARED_ANGLE


@dataclass
class CrossingMeasure:
    crossing_id: str
    dz_data_m: float | None
    upper_z_data: float | None
    lower_z_data: float | None


def input_crossings(a: RoadSeed, b: RoadSeed) -> list[shapely.Point]:
    """Plan intersections of two input centerlines that are not a shared vector node (D-037).

    Roads joined at a node (a ramp merging into a mainline, a ramp terminal on a
    street) meet; they do not cross. After registration their fitted lines often
    cross near such a node, so crossings are decided on the inputs and only
    located on the fitted lines.
    """
    inter = shapely.LineString(a.line.xy).intersection(shapely.LineString(b.line.xy))
    pts = [g for g in getattr(inter, "geoms", [inter]) if isinstance(g, shapely.Point)]
    if a.nodes is None or b.nodes is None or not pts:
        return sorted(pts, key=lambda q: (q.x, q.y))
    tree = cKDTree(b.nodes)
    d, _ = tree.query(a.nodes, distance_upper_bound=SHARED_NODE_M)
    shared = a.nodes[np.isfinite(d)]
    out = []
    for p in pts:
        if shared.size and float(np.min(np.hypot(shared[:, 0] - p.x, shared[:, 1] - p.y))) <= SHARED_NODE_M + NODE_REACH_M:
            continue
        out.append(p)
    return sorted(out, key=lambda q: (q.x, q.y))


def _tag_text(tags: dict[str, str]) -> str:
    parts = [f"{k}={tags[k]}" for k in ("layer", "level", "bridge", "tunnel") if tags.get(k)]
    return ";".join(parts)


def _tag_level(tags: dict[str, str]) -> float:
    try:
        level = float(tags.get("layer", "0") or 0)
    except ValueError:
        level = 0.0
    if tags.get("bridge", "no") not in ("no", ""):
        level += 0.5
    if tags.get("tunnel", "no") not in ("no", ""):
        level -= 0.5
    return level


def _surface_points(fit: RoadFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lp = fit.points
    road = fit.road
    z = np.array([road.surface_z(float(a), float(b)) for a, b in zip(lp.s, lp.u, strict=True)])
    on = np.abs(lp.z - z) <= SURFACE_TOL_M
    return lp.s[on], lp.u[on], lp.z[on] - (z[on] - np.array([road.profile.z(float(a)) for a in lp.s[on]]))


def _data_height(fit: RoadFit, s: float, radius: float, cache: dict) -> float | None:
    if fit.road.id not in cache:
        cache[fit.road.id] = _surface_points(fit)
    ss, uu, zz = cache[fit.road.id]
    m = (np.abs(ss - s) <= radius) & (np.abs(uu) <= radius)
    if m.sum() < 5:
        return None
    return float(np.median(zz[m]))


def find_crossings(fits: list[RoadFit]) -> tuple[list[Crossing], list[CrossingMeasure], list[Flag]]:
    crossings, measures, flags = [], [], []
    cache: dict = {}
    fits = sorted(fits, key=lambda f: f.road.id)
    lines = {f.road.id: shapely.LineString(f.line.xy) for f in fits}
    for i, a in enumerate(fits):
        for b in fits[i + 1 :]:
            inter = lines[a.road.id].intersection(lines[b.road.id])
            fitted = [g for g in getattr(inter, "geoms", [inter]) if isinstance(g, shapely.Point)]
            pts = []
            for q in input_crossings(a.seed, b.seed):
                near = [p for p in fitted if p.distance(q) <= MATCH_M]
                if near:
                    pts.append(min(near, key=lambda p: (p.distance(q), p.x, p.y)))
                else:
                    flags.append(
                        Flag("FR-CR-1:not_located", f"{a.road.id}|{b.road.id}", "FR-CR-1", f"input centerlines of {a.road.name} and {b.road.name} cross at ({q.x:.1f}, {q.y:.1f}); fitted lines do not")
                    )
            for p in sorted({(q.x, q.y): q for q in pts}.values(), key=lambda q: (q.x, q.y)):
                xy = np.array([[p.x, p.y]])
                sa, ua, _ = a.line.project(xy)
                sb, ub, _ = b.line.project(xy)
                sa, ua, sb, ub = float(sa[0]), float(ua[0]), float(sb[0]), float(ub[0])
                za = a.road.surface_z(sa, ua)
                zb = b.road.surface_z(sb, ub)
                upper, lower = (a, b) if za >= zb else (b, a)
                su, sl = (sa, sb) if upper is a else (sb, sa)
                dz = abs(za - zb)
                hu = float(upper.line.heading(su)[0])
                hl = float(lower.line.heading(sl)[0])
                if shares_pavement(dz, hu, hl):
                    flags.append(
                        Flag(
                            "FR-CR-1:shared_pavement",
                            f"{a.road.id}|{b.road.id}",
                            "FR-CR-1",
                            f"{a.road.name} and {b.road.name} meet at ({p.x:.1f}, {p.y:.1f}) on one level ({dz:.2f} m apart), "
                            f"{math.degrees(abs(math.remainder(hu - hl, math.pi))):.1f} degrees apart: shared pavement, not a crossing",
                        )
                    )
                    continue
                skew = math.remainder(hu - hl - math.pi / 2.0, math.pi)
                tu, tl = upper.seed.tags_at(su), lower.seed.tags_at(sl)
                lu, ll = _tag_level(tu), _tag_level(tl)
                agree = None if lu == ll else (lu > ll)
                cid = stable_id("cx", upper.road.id, lower.road.id, round(su, 0))
                pu = _element_prov(upper, su)
                pl = _element_prov(lower, sl)
                prov = Provenance(
                    Source.MEASURED,
                    min(pu.confidence, pl.confidence),
                    rms_m=pu.rms_m,
                    p95_m=pu.p95_m,
                    n_points=pu.n_points + pl.n_points,
                    observed_fraction=pl.observed_fraction,
                )
                if agree is False:
                    prov.flags.append("FR-CR-2:tags_disagree")
                    flags.append(
                        Flag(
                            "FR-CR-2:tags_disagree",
                            cid,
                            "FR-CR-2",
                            f"measured {upper.road.name} over {lower.road.name} by {dz:.2f} m; tags say "
                            f"upper '{_tag_text(tu)}' vs lower '{_tag_text(tl)}'",
                        )
                    )
                crossings.append(
                    Crossing(
                        id=cid,
                        upper_road=upper.road.id,
                        upper_s=su,
                        lower_road=lower.road.id,
                        lower_s=sl,
                        x=float(p.x),
                        y=float(p.y),
                        dz_m=float(dz),
                        skew_rad=float(skew),
                        prov=prov,
                        tag_upper=_tag_text(tu),
                        tag_lower=_tag_text(tl),
                        tags_agree=agree,
                    )
                )
                zu = _data_height(upper, su, UPPER_RADIUS_M, cache)
                lows = [_data_height(lower, sl + d, LOWER_RADIUS_M, cache) for d in (-LOWER_OFFSET_M, LOWER_OFFSET_M)]
                zl = 0.5 * (lows[0] + lows[1]) if all(v is not None for v in lows) else None
                measures.append(CrossingMeasure(cid, (zu - zl) if zu is not None and zl is not None else None, zu, zl))
    return crossings, measures, flags


def _element_prov(fit: RoadFit, s: float) -> Provenance:
    for el in fit.road.profile.elements:
        if el.s0 - 1e-6 <= s <= el.s0 + el.length + 1e-6:
            return el.prov
    return fit.road.profile.elements[-1].prov
