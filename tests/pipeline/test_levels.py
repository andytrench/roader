"""Roads at different levels in plan contact: links and tracking keep each road on its own level (FR-AF-3, FR-VC-2, FR-CR-1, AC-H4)."""

from __future__ import annotations

import functools

from roader.model.schema import ELEMENT_FAILED
from tests.pipeline import scenes as S


@functools.cache
def _dead_end():
    return S.run_scene(S.viaduct(dead_end=True))


def test_fr_af_3_street_ending_under_a_deck_is_not_linked_to_the_road_on_it():
    """A street ending at grade under the viaduct lies within the topology distance of the deck road's line (D-143)."""
    out = _dead_end()
    assert out.error is None, out.error
    model = out.model
    dead = next(r for r in model.roads if r.name == "Dead End Street")
    assert not [lk for lk in model.links if dead.id in (lk.from_road, lk.to_road)]
    flag = next(f for f in model.flags if f.code == "FR-AF-3:other_level")
    assert flag.element_id == dead.id and "I-99 westbound" in flag.message
    assert out.gates["AC-H4"].passed, [v.message for v in out.gates["AC-H4"].violations]
    assert not [f for f in model.flags if f.code == ELEMENT_FAILED]


@functools.cache
def _under_viaduct():
    """A short street crossing under both decks at 25 degrees: its longest visible surface is the deck above it."""
    return S.run_scene(S.viaduct(n_streets=1, skew=1.13, street_length=70.0))


def test_fr_vc_2_street_under_a_viaduct_is_not_fitted_onto_its_deck():
    """No at-grade crossing with a limited-access road: the street is refit below the deck it took (D-144, FR-CR-1)."""
    out = _under_viaduct()
    assert out.error is None, out.error
    model = out.model
    street = next(r for r in model.roads if r.name == "1st Street")
    assert any(f.code == "FR-VC-2:off_deck" and f.element_id == street.id for f in model.flags)
    for c in model.crossings:
        assert model.road(c.lower_road).id == street.id and c.dz_m > 7.0, (c.id, c.dz_m)
    for gate in ("AC-H2", "AC-H3", "AC-H4"):
        assert out.gates[gate].passed, (gate, [v.message for v in out.gates[gate].violations][:3])
