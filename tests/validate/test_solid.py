"""FR-SV-1 to FR-SV-6: native solid checks on hand-built meshes."""

from __future__ import annotations

import numpy as np

from roader.validate.solid import check_solid

from .meshes import box, box_arrays, merged, mesh, tetra


def reqs(violations) -> set[str]:
    return {v.req for v in violations}


def test_fr_sv_closed_box_passes_everything():
    assert check_solid(box()) == []


def test_fr_sv_box_at_unit_frame_magnitude_passes():
    assert check_solid(box(lo=(5123.25, -8871.5, 101.0), hi=(5135.0, -8860.0, 101.3))) == []


def test_fr_sv_3_inverted_box_fails_orientation():
    v, f = box_arrays()
    found = check_solid(mesh("inv", v, f[:, ::-1]))
    assert reqs(found) == {"FR-SV-3"}
    assert "volume" in found[0].message


def test_fr_sv_3_one_flipped_face_fails_winding():
    v, f = box_arrays()
    f[0] = f[0, ::-1]
    assert "FR-SV-3" in reqs(check_solid(mesh("flip", v, f)))


def test_fr_sv_1_open_box_fails_closed_shell():
    v, f = box_arrays()
    found = check_solid(mesh("open", v, f[1:]))
    assert "FR-SV-1" in reqs(found)
    assert "FR-SV-3" not in reqs(found)


def test_fr_sv_open_terrain_skips_closed_checks():
    v, f = box_arrays()
    found = check_solid(mesh("terrain", v, f[:2], closed=False))
    assert found == []


def test_fr_sv_5_duplicate_vertex():
    v, f = box_arrays()
    v = np.vstack([v, v[0]])
    f[0, 0] = 8
    found = check_solid(mesh("dup", v, f))
    assert "FR-SV-5" in reqs(found)
    assert any("duplicate" in x.message for x in found)


def test_fr_sv_5_zero_area_sliver_and_t_junction():
    v, f = box_arrays()
    v = np.vstack([v, 0.5 * (v[4] + v[5])])  # 8: midpoint of top edge 4-5
    top = np.array([[4, 8, 7], [8, 5, 6], [8, 6, 7], [4, 5, 8]])  # last one is the sliver
    f = np.vstack([f[:2], top, f[4:]])
    found = check_solid(mesh("sliver", v, f))
    assert "FR-SV-5" in reqs(found)
    text = " ".join(x.message for x in found if x.req == "FR-SV-5")
    assert "zero-area" in text
    assert "T-junction" in text


def test_fr_sv_5_short_edge():
    found = check_solid(box("thin", hi=(0.0005, 1.0, 1.0)))
    assert "FR-SV-5" in reqs(found)
    assert any("1 mm" in x.message for x in found)


def test_fr_sv_2_bow_tie_vertex_is_non_manifold():
    v1, f1 = tetra(True)
    v2, f2 = tetra(False)
    v = np.vstack([v1, v2[1:]])
    f2 = np.where(f2 == 0, 0, f2 + 3)
    found = check_solid(mesh("bowtie", v, np.vstack([f1, f2])))
    assert "FR-SV-2" in reqs(found)
    assert "FR-SV-4" not in reqs(found)


def test_fr_sv_2_edge_with_three_faces():
    v, f = box_arrays()
    v = np.vstack([v, [[0.5, 2.0, 0.5]]])
    f = np.vstack([f, [[2, 6, 8]]])  # a fin on edge 2-6
    assert "FR-SV-2" in reqs(check_solid(mesh("fin", v, f)))


def test_fr_sv_4_overlapping_boxes_in_one_mesh_self_intersect():
    m = merged("overlap", box_arrays(), box_arrays((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)))
    assert "FR-SV-4" in reqs(check_solid(m))


def test_fr_sv_4_vertex_pushed_through_opposite_face():
    v, f = box_arrays()
    v[6] = [0.3, 0.3, -0.5]  # top corner pushed below the bottom face
    assert "FR-SV-4" in reqs(check_solid(mesh("dent", v, f)))


def test_fr_sv_6_two_components():
    m = merged("two", box_arrays(), box_arrays((3, 0, 0), (4, 1, 1)))
    found = check_solid(m)
    assert reqs(found) == {"FR-SV-6"}
