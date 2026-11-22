# Roader model (spec section 5)

The parametric model is the product; every mesh is regenerated from it. The
schema lives in `src/roader/model/` and follows the spec's layers.

| Layer | Implementation | Notes |
|---|---|---|
| L0 Network | `Road`, `Link`, `Crossing` in `schema.py` | Content-derived ids (`stable_id`). Crossing order is measured; vector layer tags are logged beside it. |
| L1 Horizontal alignment | `Alignment` of `AlignElement` (line, arc, spiral) in `geometry.py` | Spirals have linearly varying curvature, integrated with fixed Gauss-Legendre nodes so evaluation is deterministic. |
| L2 Vertical profile | `Profile` of `VerticalElement` | Constant grades (`g0 == g1`) joined by parabolic curves. Beyond the ends the end grade continues. |
| L3 Cross-section | `CrossSection` with `Piecewise` widths, slopes, thickness; `LaneGroup`, `Median`, `Barrier` | Piecewise-linear values: every change is a taper of nonzero length. Two knots at one station are rejected (AC-P2). |
| L4 Earthwork | `SideSlope` | Fill, cut or wall per side and station range; the tie-in line is computed against L7 by `roader.solids`. |
| L5 Structures | `Structure`, `DeckSpec`, `Support` | Parts reference the carried road's L1 to L3. |
| L6 Junction areas | `JunctionArea` | Built by planar partition in milestone 2. |
| L7 Terrain | `TerrainPatch` grid | Re-triangulated with solid footprints as constraints by `roader.solids`. |
| L8 Provenance | `Provenance` on every element | Source (measured, inventory, vector, rule), residuals, point count, confidence (D-010). |

## Conventions

- Unit frame: transverse Mercator centered on the unit, heights NAVD88 (D-007). The
  anchor and PROJ string are in `Frame`.
- Station `s`: meters along the reference line. Offset `u`: meters, positive to the
  left of travel. Heading: radians counter-clockwise from east. Curvature positive
  for a left turn.
- Cross slope: `slope_left` and `slope_right` are dz/du on each side, so a normal
  crown is `-0.02` / `+0.02`.
- Serialization: canonical JSON from `roader.model.dumps`, hashed by `model_hash`
  (D-011, AC-H5).

## Mesh contract

`roader.model.Mesh` is one closed body generated from one element: vertices in the
unit frame, faces counter-clockwise seen from outside, plus `element_id`, `layer`,
`part`, `material` and the provenance `source` (FR-DM-3, AC-H6). Only the L7
terrain surface may be open.

## Synthetic scene

`roader.model.synthetic` builds a reference overpass model and the lidar it would
produce (nothing sampled under the deck). Every lane tests against it.
