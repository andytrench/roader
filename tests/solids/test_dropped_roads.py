"""Roads Bellevue dropped from the solids (FR-SG-1, FR-SG-2, FR-SG-3, FR-EX-1, D-156, D-157)."""

from __future__ import annotations

import dataclasses

from roader.model.schema import ELEMENT_FAILED, JunctionArea, JunctionSeam
from roader.model.synthetic import TERRAIN_Z, overpass_model
from roader.solids import generate
from roader.solids.junctions import zones

from . import meshcheck as mc


def _gentle_fill_beside_a_skewed_deck(skew: float = 0.6, ratio: float = 8.0):
    """Route 1 on a deck skewed ``skew``, its right fill at 1:``ratio`` (out of rule, as measured)
    over ground 8 m down, with a berm beside the road that ends 30 m before the deck: the toe
    jumps from the berm to the low ground there (Bellevue I-405 NB and link 4744441)."""
    m = overpass_model()
    over = m.roads[2]
    st = m.structures[0]
    st = dataclasses.replace(st, supports=[dataclasses.replace(sp, skew_rad=skew) for sp in st.supports])
    slopes = [dataclasses.replace(ss, ratio=ratio) if ss.road_id == over.id and ss.side == "right" else ss for ss in m.side_slopes]
    t = m.terrain
    z = []
    for j in range(t.ny):
        for i in range(t.nx):
            s, u = over.alignment.project(t.origin_x + i * t.cell_m, t.origin_y + j * t.cell_m)
            zz = TERRAIN_Z
            if u < -12.0:
                zz = TERRAIN_Z + 4.0 if -20.0 < u < -14.0 and s < st.s_start - 30.0 else TERRAIN_Z - 8.0
            z.append(zz)
    return dataclasses.replace(m, structures=[st], side_slopes=slopes, terrain=dataclasses.replace(t, z=z)), over.id


def test_d157_a_shared_region_without_seams_covers_only_its_sections():
    """Bellevue link 103757285: its shared region with SR 520 WB starts where the link stops riding
    a deck (a carried edge, no seam) and ends at the road end (no seam). The whole road was taken as
    the area's interval, which reached the link's own deck 200 m back and dropped the road."""
    m = overpass_model()
    over = m.roads[2]
    st = m.structures[0]
    lo = st.s_end + 100.0
    sections = [_section_at(over, lo + k * 10.0) for k in range(int((over.length - lo) // 10.0) + 1)]
    area = JunctionArea("jn_shared_tail", [over.id], [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], st.prov, kind="shared", sections=sections)
    assert zones(dataclasses.replace(m, junctions=[area]))[over.id] == [(lo, sections[-1].s)]
    contained = dataclasses.replace(area, kind="merge", sections=[])
    assert zones(dataclasses.replace(m, junctions=[contained]))[over.id] == [(0.0, over.length)]


def _section_at(road, s: float) -> JunctionSeam:
    x, y, heading, k = road.alignment.evaluate(s)
    return JunctionSeam(road.id, s, -5.0, 5.0, 0, x, y, heading, k)


def test_d158_a_deck_edge_a_fraction_of_a_millimetre_from_a_parapet_breakline_takes_it():
    """Bellevue SR 520: the twin-joint overhang (-0.174 m) put both decks' facing edges 0.6-0.7 mm
    from their parapet's outer top line; the strip between gave sub-millimetre edges (val3dity 102)."""
    m = overpass_model()
    st = m.structures[0]
    top_outer = 5.0 + 0.1 + 0.25  # the parapet at 5.0 m, base 0.45 m, top 0.25 m: outer top line at 5.35 m
    deck = dataclasses.replace(st.deck, overhang_left_m=top_outer + 0.0006 - 5.45)
    m = dataclasses.replace(m, structures=[dataclasses.replace(st, deck=deck)])
    meshes = generate(m)
    for mesh in meshes:
        if mesh.closed and mesh.id.startswith(st.id):
            assert not mc.solid_problems(mesh), (mesh.id, mc.solid_problems(mesh))


def test_d156_a_jumping_toe_beside_a_skewed_deck_keeps_the_skew():
    """The toe met a skewed station line nowhere (no fixed point) or out of order (lines crossed):
    the road was swept with its structures squared, which left the junction seams off the outline."""
    model, over = _gentle_fill_beside_a_skewed_deck()
    failures = []
    meshes = generate(model, failures=failures)
    assert not [f for f in failures if f.code == ELEMENT_FAILED], [f.message for f in failures]
    assert any(f.code == "FR-SG-2:toe_limited" for f in failures)
    for mesh in meshes:
        if mesh.closed:
            assert not mc.solid_problems(mesh), (mesh.id, mc.solid_problems(mesh))
    assert any(m.id.startswith(f"{over}:pavement") for m in meshes)
    assert any(m.id.startswith("ss_") and m.closed for m in meshes)
