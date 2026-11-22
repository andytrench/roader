"""Stage 2 metrics and stage 3 evidence: FR-PN-2, FR-PN-3, FR-RS-1..4."""

from __future__ import annotations

import numpy as np
import pytest

from roader.evidence import build_evidence
from roader.io.points import PointSet


@pytest.fixture(scope="module")
def evidence(points, truth):
    lines = [np.array([r.alignment.evaluate(float(s))[:2] for s in np.linspace(0.0, r.length, 60)]) for r in truth.roads]
    return build_evidence(points, lines)


def test_fr_rs_1_pavement_probability_separates_pavement_from_terrain(evidence, labels):
    lab = labels[evidence.index]
    pav = evidence.pavement_mask()
    assert pav[np.isin(lab, ["pavement", "deck"])].mean() >= 0.85
    assert pav[lab == "terrain"].mean() <= 0.05
    assert pav[lab == "parapet"].mean() <= 0.05


def test_fr_rs_3_vehicles_are_transient_and_never_in_a_sheet(evidence, labels):
    lab = labels[evidence.index]
    assert evidence.transient[lab == "vehicle"].all()
    assert not evidence.pavement_mask()[lab == "vehicle"].any()


def test_fr_rs_4_parapets_are_raised_linear_features(evidence):
    linear = [f for f in evidence.features if f.kind == "linear" and f.length_m >= 20.0]
    assert linear, "no parapet detected"
    assert all(abs(f.top_height_m - 1.07) < 0.15 for f in linear)


def _plane(x0, x1, y0, y1, z, spacing, rng):
    xs, ys = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
    xs, ys = xs.ravel(), ys.ravel()
    return np.column_stack([xs, ys, z + rng.normal(0.0, 0.02, xs.size)])


def _pointset(xyz, strip=None) -> PointSet:
    n = len(xyz)
    return PointSet(
        xyz=xyz,
        classification=np.full(n, 2, np.uint8),
        intensity=np.full(n, 30.0),
        return_number=np.ones(n, np.uint8),
        number_of_returns=np.ones(n, np.uint8),
        scan_angle=np.full(n, np.nan),
        gps_time=np.full(n, np.nan),
        point_source_id=np.ones(n, np.uint16) if strip is None else strip.astype(np.uint16),
        collection=np.zeros(n, np.uint8),
    )


def test_fr_rs_2_stacked_surfaces_are_separate_sheets():
    """A deck seven meters above a road it overlaps in plan: two sheets, never one."""
    rng = np.random.default_rng(1)
    road = _plane(0.0, 80.0, -6.0, 6.0, 100.0, 0.35, rng)
    deck = _plane(30.0, 50.0, -30.0, 30.0, 107.0, 0.35, rng)
    pts = _pointset(np.vstack([road, deck]))
    ev = build_evidence(pts, [np.array([[0.0, 0.0], [80.0, 0.0]]), np.array([[40.0, -30.0], [40.0, 30.0]])])
    sheet = ev.point_sheet
    road_sheets = set(np.unique(sheet[: len(road)]).tolist()) - {-1}
    deck_sheets = set(np.unique(sheet[len(road) :]).tolist()) - {-1}
    assert road_sheets and deck_sheets
    assert not road_sheets & deck_sheets
    # The road under the deck footprint keeps its own sheet.
    under = (road[:, 0] > 32.0) & (road[:, 0] < 48.0)
    assert set(np.unique(sheet[: len(road)][under]).tolist()) - {-1} <= road_sheets


def test_fr_pn_2_strip_offset_above_tolerance_is_corrected():
    rng = np.random.default_rng(2)
    a = _plane(0.0, 60.0, -6.0, 6.0, 100.0, 0.35, rng)
    b = _plane(0.0, 60.0, -6.0, 6.0, 100.1, 0.35, rng)
    b[:, :2] += 0.1
    xyz = np.vstack([a, b])
    strip = np.concatenate([np.full(len(a), 1), np.full(len(b), 2)])
    ev = build_evidence(_pointset(xyz, strip), [np.array([[0.0, 0.0], [60.0, 0.0]])])
    assert ev.strips.corrected == [2]
    assert abs(ev.strips.offsets_m[2] - 0.1) < 0.02
    second = ev.point_source_id == 2
    assert abs(np.median(ev.xyz[second, 2]) - np.median(ev.xyz[~second, 2])) < 0.02


def test_fr_pn_2_small_strip_offset_is_reported_not_corrected():
    rng = np.random.default_rng(3)
    a = _plane(0.0, 60.0, -6.0, 6.0, 100.0, 0.35, rng)
    b = _plane(0.0, 60.0, -6.0, 6.0, 100.02, 0.35, rng)
    strip = np.concatenate([np.full(len(a), 1), np.full(len(b), 2)])
    ev = build_evidence(_pointset(np.vstack([a, b]), strip), [np.array([[0.0, 0.0], [60.0, 0.0]])])
    assert ev.strips.corrected == []
    assert ev.strips.p95_m is not None and ev.strips.p95_m < 0.05


def test_fr_pn_3_density_on_candidate_pavement(evidence):
    d = evidence.density
    assert d.median_pts_m2 is not None and 6.0 <= d.median_pts_m2 <= 10.0
    assert d.unobserved_fraction < 0.01
