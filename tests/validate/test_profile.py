"""FR-SV-7 (analytic profile, solid-to-solid) and FR-SV-8 (tile seams)."""

from __future__ import annotations

import numpy as np

from roader.validate.profile import check_profile, check_seam, seam_plane_from_line

from .meshes import box, mesh, tetra


def test_fr_sv_7_overlapping_solids_fail():
    found = check_profile([box("a"), box("b", (0.5, 0.5, 0.5), (1.5, 1.5, 1.5))])
    assert found
    assert {v.req for v in found} == {"FR-SV-7"}


def test_fr_sv_7_contained_solid_fails():
    found = check_profile([box("a", (0, 0, 0), (4, 4, 4)), box("b", (1, 1, 1), (2, 2, 2))])
    assert any("inside" in v.message for v in found)


def test_fr_sv_7_same_side_coplanar_overlap_fails():
    # Boundaries coincide but interiors overlap over [1, 2] x [0, 2] x [0, 2].
    found = check_profile([box("a", (0, 0, 0), (2, 2, 2)), box("b", (1, 0, 0), (3, 2, 2))])
    assert found


def test_fr_sv_7_face_sharing_conforming_boxes_pass():
    # Shared face x = 1 is triangulated with different diagonals; vertices coincide.
    assert check_profile([box("a"), box("b", (1, 0, 0), (2, 1, 1))]) == []


def test_fr_sv_7_separate_solids_pass():
    assert check_profile([box("a"), box("b", (3, 0, 0), (4, 1, 1))]) == []


def test_fr_sv_7_edge_contact_passes():
    assert check_profile([box("a"), box("b", (1, 1, 0), (2, 2, 1))]) == []


def test_fr_sv_7_t_junction_contact_is_non_conforming():
    found = check_profile([box("a"), box("b", (1, 0, 0), (2, 0.5, 1))])
    assert found
    assert all("non-conforming" in v.message for v in found)


def test_fr_sv_7_tetra_piercing_box_fails():
    v, f = tetra(True, at=(0.5, 0.5, 0.5))
    assert check_profile([box("a"), mesh("t", v + [0.0, 0.0, 0.2], f)])


def test_fr_sv_8_bit_identical_seam_passes():
    origin, normal = seam_plane_from_line((1.0, -5.0), (1.0, 5.0))
    assert check_seam([box("a")], [box("b", (1, 0, 0), (2, 1, 1))], origin=origin, normal=normal) == []


def test_fr_sv_8_seam_off_by_a_bit_fails():
    origin, normal = seam_plane_from_line((1.0, -5.0), (1.0, 5.0))
    right = box("b", (1, 0, 0), (2, 1, 1))
    right.vertices[0, 1] = np.nextafter(right.vertices[0, 1], 1.0)
    found = check_seam([box("a")], [right], origin=origin, normal=normal)
    assert found and {v.req for v in found} == {"FR-SV-8"}
    assert any("bit-identical" in v.message for v in found)


def test_fr_sv_8_missing_partner_fails():
    origin, normal = seam_plane_from_line((1.0, -5.0), (1.0, 5.0))
    found = check_seam([box("a")], [box("b", (1, 0, 0), (2, 0.5, 1))], origin=origin, normal=normal)
    assert any("no partner" in v.message for v in found)
