"""Stages 4 and 5: FR-VC-1..5, FR-CR-1..3 on the synthetic overpass."""

from __future__ import annotations

import numpy as np

import roader
from roader.conflate.graph import data_extent, select_roads
from roader.conflate.polyline import Polyline
from roader.io import VectorEdge
from roader.pipeline import run_pipeline
from tests.pipeline.conftest import by_feature


def test_import_path_is_this_worktree():
    assert "roader-lanes" in roader.__file__ or "src/roader" in roader.__file__


def test_fr_vc_1_scope_and_chaining(points, vectors):
    over = next(v for v in vectors if v.feature_id == "over")
    half = len(over.coords) // 2
    split = [v for v in vectors if v.feature_id != "over"] + [
        VectorEdge("synthetic", "over_a", over.coords[: half + 1], dict(over.tags)),
        VectorEdge("synthetic", "over_b", over.coords[half:], dict(over.tags)),
        VectorEdge("synthetic", "far", np.array([[-280.0, 250.0], [-200.0, 250.0]]), {"highway": "residential"}),
    ]
    seeds = select_roads(split, data_extent(points.xyz[:, :2]))
    assert len(seeds) == 3, [s.name for s in seeds]
    street = next(s for s in seeds if s.kind == "crossing_street")
    assert sorted(sp.feature_id for sp in street.spans) == ["over_a", "over_b"]
    assert {s.kind for s in seeds} == {"mainline", "crossing_street"}
    assert all(s.oneway for s in seeds if s.kind == "mainline")


def test_fr_vc_2_lower_roads_are_occluded_under_the_deck(model):
    for fid in ("eb", "wb"):
        road = by_feature(model, fid)
        occluded = [iv for iv in road.support if iv.kind == "occluded"]
        assert len(occluded) == 1
        iv = occluded[0]
        assert iv.reason == "under_deck"
        x, _y = road.alignment.offset_point(0.5 * (iv.s0 + iv.s1), 0.0)
        assert abs(x) < 10.0
    over = by_feature(model, "over")
    assert all(iv.kind == "observed" for iv in over.support)


def test_fr_vc_3_registration_moves_centerlines_onto_the_pavement(model, truth, vectors):
    for fid in ("eb", "wb", "over"):
        t = Polyline.from_alignment(by_feature(truth, fid).alignment)
        fitted = Polyline.from_alignment(by_feature(model, fid).alignment)
        vec = next(v for v in vectors if v.feature_id == fid)
        _s, u_in, ok_in = t.project(vec.coords)
        _s, u_out, ok_out = t.project(fitted.xy[20:-20:10])
        assert np.sqrt(np.mean(u_in[ok_in] ** 2)) > 0.9  # the input is several decimeters off
        assert np.max(np.abs(u_out[ok_out])) < 0.10  # FR-VC-3 target is 0.5 m


def test_fr_vc_4_structure_ends_from_data(model, truth):
    st = model.structures[0]
    ts = truth.structures[0]
    assert len(model.structures) == 1
    assert abs(st.s_start - ts.s_start) < 1.0
    assert abs(st.s_end - ts.s_end) < 1.0
    assert any(f.startswith("FR-VC-4:") for f in st.prov.flags)


def test_fr_cr_1_crossings_order_dz_and_skew(model, truth):
    over = by_feature(model, "over")
    assert len(model.crossings) == 2
    truth_dz = {c.lower_road: c.dz_m for c in truth.crossings}
    for c in model.crossings:
        assert c.upper_road == over.id
        lower = model.road(c.lower_road)
        t_lower = by_feature(truth, lower.vector_refs[0].feature_id)
        assert abs(c.dz_m - truth_dz[t_lower.id]) < 0.1
        assert abs(abs(c.skew_rad) - 0.17) < 0.01
        assert c.tags_agree is True


def test_fr_cr_2_tag_disagreement_is_logged_with_both_values(points, vectors, truth):
    flipped = [VectorEdge(v.source, v.feature_id, v.coords, {**v.tags, "layer": "2"} if v.feature_id == "eb" else v.tags) for v in vectors]
    res = run_pipeline(points, flipped, truth.frame, truth.unit)
    eb = by_feature(res.model, "eb")
    c = next(c for c in res.model.crossings if c.lower_road == eb.id)
    assert c.upper_road != eb.id  # order is measured, not taken from tags
    assert c.tags_agree is False
    assert "layer=2" in c.tag_lower
    flags = [f for f in res.model.flags if f.code == "FR-CR-2:tags_disagree"]
    assert len(flags) == 1 and "layer=2" in flags[0].message


def test_fr_cr_3_chained_crossings_share_one_structure(model):
    st = model.structures[0]
    assert {c.structure_id for c in model.crossings} == {st.id}
    assert sorted(st.crossed_roads) == sorted(c.lower_road for c in model.crossings)
