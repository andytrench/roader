"""Structure inference (FR-ST-2 .. FR-ST-6, FR-XS-5).

No inventory matching yet (FR-ST-1 lands with the NBI reader), so every
structure proceeds on data plus rules:

- Ends: measured (``roader.conflate.structures``).
- Supports: abutments at the measured ends; piers only in the gaps between the
  crossed roads' pavements less the lateral clearance rule, never over a travel
  lane (FR-ST-3 layout rule). Observed pier returns are not used yet.
- Supports align with the skew of what they cross (FR-ST-4).
- Family from the longest span (``structure.slab_max_span``), depth from the
  family's depth-to-span rule, slab thickness and columns by rule (FR-ST-2, FR-ST-5).
- Review queue above ``structure.review_span`` (FR-ST-6).
- Deck overhang from deck-level returns beyond the pavement edge (FR-XS-5),
  never less than what the parapet base needs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import shapely

from roader.conflate.structures import StructureExtent
from roader.evidence import Evidence
from roader.fit.road import RoadFit
from roader.model.provenance import Provenance, Source
from roader.model.schema import Barrier, Crossing, DeckSpec, Flag, Structure, Support
from roader.model.serialize import stable_id
from roader.rules import RuleBook, era_of

OVERHANG_BAND_M = 3.0
DECK_LEVEL_BELOW_M = 0.3
DECK_LEVEL_ABOVE_M = 1.6
GROUND_INSIDE_M = 3.0
# FR-ST-5 / FR-RD-4: a deck face seen by the scanner beats the depth rule.
FASCIA_BAND_M = 0.4
FASCIA_TOP_M = 0.3
FASCIA_BOTTOM_M = 4.0
FASCIA_MIN_POINTS = 50
FASCIA_COVERAGE = 0.3
PIER_END_CLEAR_M = 5.0  # no pier closer than this to an abutment
MIN_PIER_GAP_M = 8.0
METHOD_CONFIDENCE = {"class17": 0.9, "sheet_break": 0.7, "class17_only": 0.5}


@dataclass
class StructureResult:
    structure: Structure
    crossings: list[Crossing]
    flags: list[Flag]
    spans: list[float]


def _footprint_on(upper: RoadFit, lower: RoadFit, c: Crossing, clearance: float) -> tuple[float, float]:
    """Stations on the carried road covered by the lower road's pavement plus lateral clearance."""
    road = lower.road
    xs = road.cross_section
    s_lo = np.arange(max(0.0, c.lower_s - 80.0), min(road.length, c.lower_s + 80.0), 1.0)
    stations = []
    upper_line = shapely.LineString(upper.line.xy)
    for sign, piece in ((1.0, xs.width_left), (-1.0, xs.width_right)):
        u = np.array([sign * (piece(float(s)) + clearance) for s in s_lo])
        edge = lower.line.offset_point(s_lo, u)
        if len(edge) < 2:
            continue
        inter = shapely.LineString(edge).intersection(upper_line)
        for g in getattr(inter, "geoms", [inter]):
            if isinstance(g, shapely.Point):
                s, _u, _ok = upper.line.project(np.array([[g.x, g.y]]))
                stations.append(float(s[0]))
    if len(stations) < 2:
        half = (xs.width_left(c.lower_s) + xs.width_right(c.lower_s) + 2 * clearance) / 2.0 / max(math.cos(c.skew_rad), 0.2)
        return c.upper_s - half, c.upper_s + half
    return min(stations), max(stations)


def _overhang(upper: RoadFit, ev: Evidence, s0: float, s1: float) -> float | None:
    """Median outer extent of deck-level returns beyond the pavement edge, both sides."""
    road = upper.road
    xy = ev.xyz[:, :2]
    s, u, ok = upper.line.project(xy, max_dist=30.0)
    idx = np.flatnonzero(ok & (s >= s0) & (s <= s1))
    if idx.size < 50:
        return None
    s, u, z = s[idx], u[idx], ev.xyz[idx, 2]
    pz = np.array([road.profile.z(float(v)) for v in s])
    level = (z >= pz - 1.0 - DECK_LEVEL_BELOW_M) & (z <= pz + 1.0 + DECK_LEVEL_ABOVE_M)
    out = []
    for sign, piece in ((1.0, road.cross_section.width_left), (-1.0, road.cross_section.width_right)):
        w = np.array([piece(float(v)) for v in s])
        d = sign * u - w
        surf = np.array([road.surface_z(float(a), float(b)) for a, b in zip(s, u, strict=True)])
        m = level & (d > 0.0) & (d <= OVERHANG_BAND_M) & (z >= surf - DECK_LEVEL_BELOW_M) & (z <= surf + DECK_LEVEL_ABOVE_M)
        if m.sum() < 20:
            continue
        bins = np.floor(s[m] / 5.0).astype(np.int64)
        per = [float(np.percentile(d[m][bins == b], 98.0)) for b in np.unique(bins) if (bins == b).sum() >= 4]
        if per:
            out.append(float(np.median(per)))
    return float(np.mean(out)) if out else None


def _ground(drop: tuple[np.ndarray, np.ndarray, np.ndarray], road, s: float) -> float | None:
    st, dl, dr = drop
    m = np.abs(st - s) <= 3.0
    vals = np.concatenate([dl[m], dr[m]])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return road.profile.z(s) + float(np.median(vals))


def measured_depth(upper: RoadFit, ev: Evidence, s0: float, s1: float, overhang: float) -> tuple[float | None, int, float]:
    """Deck depth from fascia returns: points in a thin band at the deck edge, below the deck surface.

    Returns (depth or None, number of fascia returns, fraction of the span they cover).
    Depth is the 95th percentile drop below the surface; None unless the returns cover
    at least FASCIA_COVERAGE of the span in 2 m bins with FASCIA_MIN_POINTS in all.
    """
    road = upper.road
    s, u, ok = upper.line.project(ev.xyz[:, :2], max_dist=30.0)
    idx = np.flatnonzero(ok & (s >= s0 + 2.0) & (s <= s1 - 2.0))
    if idx.size == 0:
        return None, 0, 0.0
    s, u, z = s[idx], u[idx], ev.xyz[idx, 2]
    drops, stations = [], []
    for sign, piece in ((1.0, road.cross_section.width_left), (-1.0, road.cross_section.width_right)):
        edge = np.array([piece(float(v)) for v in s]) + overhang
        band = np.abs(sign * u - edge) <= FASCIA_BAND_M
        dz = np.array([road.profile.z(float(v)) for v in s[band]]) - z[band]
        keep = (dz >= FASCIA_TOP_M) & (dz <= FASCIA_BOTTOM_M)
        drops.append(dz[keep])
        stations.append(s[band][keep])
    d = np.concatenate(drops)
    st = np.concatenate(stations)
    if d.size == 0:
        return None, 0, 0.0
    bins = np.unique(np.floor(st / 2.0))
    coverage = float(len(bins) * 2.0 / max(s1 - s0 - 4.0, 1.0))
    if d.size < FASCIA_MIN_POINTS or coverage < FASCIA_COVERAGE:
        return None, int(d.size), coverage
    return float(np.percentile(d, 95.0)), int(d.size), coverage


def _even_piers(
    extent: StructureExtent, piers: list[float], zones: list[tuple[float, float]], need: float, rules: RuleBook, era: str
) -> tuple[list[float], set[float]]:
    """FR-ST-3, last priority: split spans longer than ``structure.even_span`` high into equal spans.

    A pier that would land in a crossed road's clear zone moves to the nearer
    zone edge (never into a travel lane); one that then crowds a neighbour is
    dropped. Returns all piers, sorted, and the set added here.
    """
    rule = rules.get("structure.even_span")
    typical, _ = rules.fill("structure.even_span", era=era)
    stations = [extent.s_start, *sorted(piers), extent.s_end]
    added: list[float] = []
    for a, b in itertools.pairwise(stations):
        if b - a <= rule.high:
            continue
        n = math.ceil((b - a) / typical)
        for k in range(1, n):
            s = a + k * (b - a) / n
            for z0, z1 in zones:
                if z0 - 0.5 * need < s < z1 + 0.5 * need:
                    s = z0 - 0.5 * need if s - z0 < z1 - s else z1 + 0.5 * need
            if not a + PIER_END_CLEAR_M <= s <= b - PIER_END_CLEAR_M:
                continue
            if any(z0 - 0.5 * need < s < z1 + 0.5 * need for z0, z1 in zones):
                continue
            if added and s - added[-1] < MIN_PIER_GAP_M:
                continue
            added.append(s)
    return sorted([*piers, *added]), set(added)


def parapet_fill(road, structure: Structure, existing: list[Barrier], rules: RuleBook) -> list[Barrier]:
    """Rule parapets on deck sides where none was observed (FR-ST-5, FR-RD-4: observed parts win)."""
    era = era_of(structure.year_built)
    out = []
    for side in ("left", "right"):
        covered = sum(
            max(0.0, min(b.s1, structure.s_end) - max(b.s0, structure.s_start)) for b in existing if b.side == side and b.kind == "parapet"
        )
        if covered >= 0.5 * (structure.s_end - structure.s_start):
            continue
        height, prov = rules.fill("parapet.height", era=era)
        base, _ = rules.fill("parapet.base_width", era=era)
        top, _ = rules.fill("parapet.top_width", era=era)
        piece = road.cross_section.width_left if side == "left" else road.cross_section.width_right
        offset = float(piece(0.5 * (structure.s_start + structure.s_end)))
        # Outside any curb or walkway measured on this side of the deck.
        for b in existing:
            if b.side == side and b.kind == "curb" and b.s1 >= structure.s_start and b.s0 <= structure.s_end:
                offset = max(offset, b.offset_m + b.base_width_m)
        out.append(
            Barrier(
                stable_id("br", road.id, side, round(structure.s_start, 0), "rule"),
                side,
                structure.s_start,
                structure.s_end,
                offset,
                height,
                base,
                top,
                "parapet",
                prov,
            )
        )
    return out


def infer_structure(
    extent: StructureExtent,
    upper: RoadFit,
    crossings: list[Crossing],
    fits: dict[str, RoadFit],
    ev: Evidence,
    rules: RuleBook,
    drop: tuple[np.ndarray, np.ndarray, np.ndarray],
    barriers: list[Barrier],
) -> StructureResult:
    flags: list[Flag] = []
    road = upper.road
    year = None
    era = era_of(year)
    sid = stable_id("st", road.id, round(extent.s_start, 0))
    on = sorted([c for c in crossings if c.upper_road == road.id and extent.s_start - 1.0 <= c.upper_s <= extent.s_end + 1.0], key=lambda c: c.upper_s)
    skew = float(np.mean([c.skew_rad for c in on])) if on else 0.0
    lateral, _lp = rules.fill("clearance.lateral.support", road_class=fits[on[0].lower_road].road.road_class if on else "*")
    col_w, col_prov = rules.fill("column.width", era=era)

    # FR-ST-3: piers only between the crossed roads' clear zones.
    zones = [_footprint_on(upper, fits[c.lower_road], c, lateral) for c in on]
    zones.sort()
    piers = []
    need = col_w / max(math.cos(skew), 0.2) + 0.5
    inner = (extent.s_start + PIER_END_CLEAR_M, extent.s_end - PIER_END_CLEAR_M)
    for (a0, a1), (b0, b1) in itertools.pairwise(zones):
        gap0, gap1 = a1, b0
        mid = 0.5 * (gap0 + gap1)
        if gap1 - gap0 >= need and inner[0] <= mid <= inner[1]:
            piers.append(mid)
    piers, even = _even_piers(extent, piers, zones, need, rules, era)
    stations = [extent.s_start, *piers, extent.s_end]
    spans = list(np.diff(stations))
    max_span = float(max(spans)) if spans else extent.length

    slab_max, slab_prov = rules.fill("structure.slab_max_span", era=era)
    family = "slab" if max_span < slab_max else "multi_girder"
    ratio, ratio_prov = rules.fill(f"superstructure.depth_ratio.{family}", era=era)
    depth = ratio * max_span
    slab_t, _ = rules.fill("deck.slab_thickness", era=era)
    review_span, _ = rules.fill("structure.review_span", era=era)
    review = max_span >= review_span

    # FR-XS-5: overhang from deck-level returns; the parapet base must fit on the deck.
    measured = _overhang(upper, ev, extent.s_start, extent.s_end)
    deck_barriers = [b for b in barriers if b.s1 >= extent.s_start and b.s0 <= extent.s_end]
    need = 0.0
    for b in deck_barriers:
        w = (road.cross_section.width_left if b.side == "left" else road.cross_section.width_right)(0.5 * (b.s0 + b.s1))
        need = max(need, b.offset_m - w + b.base_width_m + 0.05)
    deck_flags = [f"FR-ST-5:depth_ratio:{ratio_prov.rule_id}", f"FR-ST-2:family_by_span:{slab_prov.rule_id}"]
    if measured is not None and measured >= need:
        overhang = measured
        deck_flags.append("FR-XS-5:overhang_measured")
        ok, _rule = rules.evaluate("deck.overhang", overhang, era=era)
        if not ok:
            deck_flags.append("FR-RL-3:deck.overhang_out_of_range")
    elif deck_barriers:
        overhang = need
        deck_flags.append("FR-XS-5:overhang_from_parapet_base")
    else:
        overhang, _ = rules.fill("deck.overhang", era=era)
        deck_flags.append("FR-XS-5:overhang_by_rule")
    m_depth, n_fascia, coverage = measured_depth(upper, ev, extent.s_start, extent.s_end, overhang)
    if m_depth is not None:
        depth = m_depth
        deck_prov = Provenance(Source.MEASURED, 0.6, n_points=n_fascia, observed_fraction=min(coverage, 1.0), flags=[*deck_flags, "FR-ST-5:depth_measured_from_fascia"])
        ok, rule = rules.evaluate(f"superstructure.depth_ratio.{family}", depth / max_span, era=era)
        if not ok:
            deck_prov.flags.append(f"FR-RL-3:out_of_range:{rule.id}")
    else:
        deck_flags.append(f"FR-ST-5:fascia_returns:{n_fascia}:coverage:{coverage:.2f}")
        deck_prov = Provenance(Source.RULE, ratio_prov.confidence, rule_id=ratio_prov.rule_id, flags=deck_flags + ratio_prov.flags)
    deck = DeckSpec(depth, slab_t, overhang, deck_prov)

    cap_d, cap_prov = rules.fill("cap.depth", era=era)
    cap_over, _ = rules.fill("cap.overhang", era=era)
    foot, foot_prov = rules.fill("footing.depth", era=era)
    per_col, _ = rules.fill("bent.width_per_column", era=era)
    deck_width = float(
        road.cross_section.width_left((extent.s_start + extent.s_end) / 2) + road.cross_section.width_right((extent.s_start + extent.s_end) / 2) + 2 * overhang
    )
    supports = []
    for n, s in enumerate(stations):
        kind = "abutment" if n in (0, len(stations) - 1) else "pier"
        probe = s + (GROUND_INSIDE_M if n == 0 else -GROUND_INSIDE_M if kind == "abutment" else 0.0)
        ground = _ground(drop, road, probe)
        if ground is None:
            lows = [fits[c.lower_road].road.profile.z(c.lower_s) for c in on]
            ground = float(min(lows)) if lows else road.profile.z(s) - depth - 5.0
            flags.append(Flag("FR-OC-3:ground_by_neighbour", sid, "FR-OC-3", f"no ground returns beside the deck at s={s:.1f}; footing from the lower road"))
        label = f"a{1 if n == 0 else 2}" if kind == "abutment" else f"p{n}"
        layout = "FR-ST-3:even_spacing" if s in even else "FR-ST-3:layout_rule"
        prov = Provenance(Source.RULE, cap_prov.confidence, rule_id=foot_prov.rule_id if kind == "abutment" else col_prov.rule_id, flags=[layout if kind == "pier" else "FR-ST-3:measured_end"])
        supports.append(
            Support(
                id=stable_id("sp", sid, label),
                station=float(s),
                kind=kind,
                skew_rad=skew,
                n_columns=0 if kind == "abutment" else max(1, math.ceil(deck_width / per_col)),
                column_width_m=0.0 if kind == "abutment" else col_w,
                cap_depth_m=cap_d,
                cap_width_m=col_w + 2.0 * cap_over,
                footing_z=float(ground - foot),
                prov=prov,
            )
        )
    st_prov = Provenance(Source.MEASURED, METHOD_CONFIDENCE.get(extent.method, 0.5), flags=[f"FR-VC-4:{extent.method}"])
    if extent.one_sided:
        st_prov.flags.append("FR-VC-4:one_sided")
    if extent.clipped:
        st_prov.flags.append("FR-VC-4:clipped_at_road_end")
        flags.append(Flag("FR-VC-4:clipped", sid, "FR-VC-4", f"structure ends clamped to its road ({extent.s_start:.1f}-{extent.s_end:.1f}); the deck may continue past the modelled road"))
    if extent.chained:
        st_prov.flags.append("FR-CR-3:chained " + "; ".join(f"{a:.1f}-{b:.1f} {m}" for a, b, m in extent.chained))
    if extent.geom and extent.class17:
        st_prov.flags.append(f"FR-VC-4:sheet_break {extent.geom[0]:.1f}-{extent.geom[1]:.1f}; class17 {extent.class17[0]:.1f}-{extent.class17[1]:.1f}")
    if review:
        flags.append(Flag("FR-ST-6:review", sid, "FR-ST-6", f"max span {max_span:.0f} m at or above the review threshold"))
    structure = Structure(
        id=sid,
        kind="bridge",
        carried_road=road.id,
        s_start=float(extent.s_start),
        s_end=float(extent.s_end),
        family=family,
        year_built=year,
        crossed_roads=list(dict.fromkeys(c.lower_road for c in on)),
        deck=deck,
        supports=supports,
        prov=st_prov,
        review=review,
    )
    for c in on:
        c.structure_id = sid
    return StructureResult(structure, on, flags, [float(x) for x in spans])
