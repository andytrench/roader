"""Terrain re-triangulation around footprints that cross or leave the patch (FR-SG-6, D-072)."""

from __future__ import annotations

import numpy as np

from roader.model.provenance import Provenance, Source
from roader.model.schema import TerrainPatch
from roader.solids.terrain import TerrainSampler, terrain_meshes


def _sampler() -> TerrainSampler:
    nx = ny = 21
    z = [10.0 + 0.1 * i + 0.05 * j for j in range(ny) for i in range(nx)]
    return TerrainSampler(TerrainPatch(0.0, 0.0, 1.0, nx, ny, z, Provenance(Source.MEASURED, 1.0)))


def _ring(sampler, loops):
    seen, ring = set(), []
    for loop in loops:
        for x, y in loop:
            if (x, y) not in seen:
                seen.add((x, y))
                ring.append((x, y, sampler(x, y)))
    return ring


def _boundary_z_ok(meshes, sampler):
    for m in meshes:
        v = m.vertices
        assert np.all(np.isfinite(v))
        # Every terrain vertex lies on the (planar) ground surface.
        assert np.allclose(v[:, 2], [sampler(x, y) for x, y in v[:, :2]], atol=1e-9)


def test_fr_sg_6_crossing_footprints_do_not_stop_the_terrain():
    """Two ground solids whose footprints cross: the crossing points become terrain vertices on the ground."""
    s = _sampler()
    a = [(5.3, 5.3), (11.7, 5.3), (11.7, 9.7), (5.3, 9.7)]
    b = [(9.2, 7.4), (15.6, 7.4), (15.6, 13.1), (9.2, 13.1)]
    meshes = terrain_meshes(s, [a, b], _ring(s, [a, b]), "terrain", "measured", 0.01)
    assert meshes
    _boundary_z_ok(meshes, s)
    xy = np.vstack([m.vertices[:, :2] for m in meshes])
    for p in ((11.7, 7.4), (9.2, 9.7)):
        assert np.min(np.hypot(*(xy - p).T)) < 1e-6


def test_fr_sg_6_footprint_leaving_the_patch():
    s = _sampler()
    a = [(15.3, 4.3), (25.0, 4.3), (25.0, 8.6), (15.3, 8.6)]
    meshes = terrain_meshes(s, [a], _ring(s, [a]), "terrain", "measured", 0.01)
    assert meshes
    assert all(np.all(m.vertices[:, 0] <= 20.0 + 1e-9) for m in meshes)


def test_fr_sg_6_ground_line_crossing_a_body_top_keeps_the_wall_closed():
    """Terrain above a support's top at one corner and below it at the next: no crossing strip, the shell closes."""
    from roader.model.provenance import Source as Src
    from roader.solids.assemble import Body, assemble
    from roader.solids.grid import Cell, PlanVertices

    from . import meshcheck as mc

    nx = ny = 11
    z = [9.5 + 0.1 * i for j in range(ny) for i in range(nx)]  # rises 0.1 m per metre in x
    s = TerrainSampler(TerrainPatch(0.0, 0.0, 1.0, nx, ny, z, Provenance(Source.MEASURED, 1.0)))
    pv = PlanVertices()
    loop = [pv.add(2.5, 2.5), pv.add(7.5, 2.5), pv.add(7.5, 7.5), pv.add(2.5, 7.5)]
    cell = Cell(loop, [(loop[0], loop[1], loop[2]), (loop[0], loop[2], loop[3])])
    body = Body("sp:abutment", "sp", "L5", "abutment", "structural_concrete", Src.RULE, [cell],
                lambda _p: 8.0, lambda _p: 10.0, ground=True)
    result = assemble([body], pv, s)
    (raw,) = result.raw
    mesh = result.meshes[0]
    assert mc.solid_problems(mesh) == [], mc.solid_problems(mesh)
    assert raw is not None
