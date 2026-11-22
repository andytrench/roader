# Roader exports (spec stage 12)

`roader.export.write_unit(model, meshes, report, out_dir, *, points=None, lods=None)` writes
every export of one unit with shared ids (FR-EX-3) and routes the unit through the
publish gate (FR-EX-1). Writers use the standard library plus numpy; the report
figures use matplotlib's Agg canvas. No timestamps, sorted iteration and fixed float
formatting make two runs on the same inputs byte-identical (AC-H5).

## Files

A unit whose hard gates all pass (`GateReport.publishable`) is written to
`out_dir/published/<unit_id>/`. Otherwise it goes to `out_dir/held/<unit_id>/` with
`HELD.json` listing the failed hard gates and their first violations. Each call
rebuilds the destination and deletes the unit's copy in the other state, so a unit
that starts failing withdraws its published copy. Reduced-LOD publication of a
failing unit (the spec's alternative to holding) is not implemented yet; the LOD
chain itself is published (below).

`roader.export.run.export_model(..., profile="analytic", lods=True)` renders the
solids in one export profile (section 7: `analytic` or `realtime`), renders LOD
levels 1 to 3 (FR-AS-1) in both AC-H5 runs, gates them (`FR-AS-1`, hard) together
with the profile's `AC-H1` (`AC-H1:realtime` for the real-time profile), and writes
them into the glTF and USD files with the main rendering as level 0.

| File | Format | Content | Requirements |
|---|---|---|---|
| `model.json` | canonical JSON (`roader.model.dumps`) | the native parametric model | FR-EX-3, D-011 |
| `meshes.glb` | glTF 2.0 binary | one node per mesh, PBR material per semantic slot; NORMAL, TEXCOORD_0/1; coarser LODs through `MSFT_lod` | FR-EX-3, FR-DM-3, FR-AS-1, FR-AS-3..6, FR-AS-10 |
| `meshes_provenance.glb` | glTF 2.0 binary | same geometry, materials colored by provenance source | FR-RH-1 |
| `meshes.usda` | USD text | one Mesh prim per mesh, UsdPreviewSurface per slot; primvars `normals`, `st`, `st1`; LODs as a `lod` variant set | FR-EX-3, FR-DM-3, FR-AS-1, FR-AS-3..5 |
| `model.city.json` | CityJSON 2.0 | solids and semantics, Transportation, Bridge, Relief | FR-DM-2, FR-RH-2 |
| `roads.xodr` | ASAM OpenDRIVE 1.8 | L0 to L3 and L6 road description | FR-DM-1 |
| `report.json`, `report.md` | JSON, Markdown | inputs and versions, coverage, flags, gate table, triangle counts, residuals | FR-EX-4, FR-AS-2 |
| `figures/plan.png`, `figures/profile_<road>.png` | PNG | plan view; profile per road with points overlaid when given | FR-EX-4 |
| `HELD.json` | JSON | failed hard gates (held units only) | FR-EX-1 |

Shared ids: model element ids (`rd_...`, `st_...`, `sp_...`, `br_...`, `ss_...`,
`jn_...`) and mesh ids (`<element_id>:<part>[:<n>]`) are the same in every file.
The glTF node name, USD `customData.mesh_id`, CityJSON `attributes.mesh_ids` and
`report.json` `triangles.per_mesh` all use the mesh id. The OpenDRIVE road, bridge and
junction ids and the CityJSON object ids all use the element id.

## Frames and axes

The model frame is the unit's transverse Mercator frame (D-007): x east, y north,
z up (NAVD88 orthometric heights), meters, right-handed. Its origin is the geodetic
anchor `(Frame.anchor_lon, Frame.anchor_lat)`. In milestone 1 a unit is one tile, so
the tile-local origin equals the unit frame origin, `[0, 0, 0]` (FR-AS-10). Every
format records the PROJ string and the anchor.

| Format | Axes | Precision | Where the anchor is recorded |
|---|---|---|---|
| glTF | Y-up right-handed: `gltf(X, Y, Z) = unit(x, z, -y)`, so X east, Y up, Z south. This is a proper rotation, so face winding is unchanged. | float32 (FR-DM-5 allows single precision in meshes) | `asset.extras.frame`, `asset.extras.axis_conversion` |
| USD | `upAxis = "Z"`, `metersPerUnit = 1`, the unit frame unchanged | text with 6 decimals | `customLayerData.roader.frame` |
| CityJSON | the unit frame; integer vertices, `transform.scale` 0.001, integer-meter `translate` | 1 mm | `metadata.roaderFrame` |
| OpenDRIVE | the unit frame (OpenDRIVE inertial x/y/z, heading counter-clockwise from x) | text with 10 decimals | `header/geoReference` (PROJ string), header `userData` |

## Normals, UVs and LODs

Generated solids carry three per-face-corner attributes (`Mesh.normals`, `uv0`,
`uv1`; see `roader.solids.attributes`). A closed solid keeps one vertex where a
crease or UV seam passes, so the attributes are stored per corner and the glTF
writer unwelds them (one glTF vertex per distinct position, normal and UV pair, in
first-use order).

| Attribute | Content | glTF | USD |
|---|---|---|---|
| normals (FR-AS-3) | smooth across model-continuous edges, hard at slab edges, barrier arrises, embankment slope breaks and end-wall corners | `NORMAL`, rotated to Y-up | `primvars:normals`, faceVarying |
| uv0 (FR-AS-4) | metres: road and deck tops (distance along the offset line, arc length across); walls (distance along the outline, elevation); other faces an isometric face projection | `TEXCOORD_0` | `primvars:st` |
| uv1 (FR-AS-5) | non-overlapping atlas in [0, 1], 1024 px with 2 px padding, one texel density per mesh | `TEXCOORD_1`; density in node `extras.uv1_texel_px_per_m`, atlas in `asset.extras.uv1_atlas` | `primvars:st1`; same keys in customData / customLayerData |

UVs are written as stored (v is not flipped to image space). The terrain carries
normals and uv1 but no uv0: it is not a road surface and engines texture it in world
space. LOD levels (chord 1 cm, 5 cm, 25 cm, 1 m) keep the mesh ids of level 0; glTF
coarser nodes are named `<mesh id>#lod<k>` and sit outside the scene roots, as
`MSFT_lod` requires. CityJSON and the native model carry level 0 only: the coarser
levels are tessellations of the same semantic LOD, not different CityGML LODs.

## Model to OpenDRIVE

| Model | OpenDRIVE 1.8 | Exactness |
|---|---|---|
| `Road` | `road` (`id` = road id, `length`, `rule="RHT"`, `junction="-1"`) | exact |
| `Road.road_class` | `type@type`: `motorway` for `motorway*`, else `unknown` | the raw class is in userData |
| `Link` | `road/link/predecessor` or `successor` (`elementType="road"`, `contactPoint`) | only the first link per road end is written there; all links are in header userData |
| `AlignElement` line / arc / spiral | `planView/geometry` with `line`, `arc@curvature`, `spiral@curvStart,curvEnd` | exact; read back with `read_xodr_geometry` to within 1e-6 m (tested) |
| `VerticalElement` | `elevation s a b c d`: a = z0, b = g0, c = (g1 - g0) / (2 L), d = 0 | exact (tested). Beyond the last element the cubic continues, where the model continues the end grade; profiles cover the full road. |
| `slope_left`, `slope_right` | banking g = (left + right) / 2 as `superelevation` with roll = atan(g), linear between slope knots; crown c = (right - left) / 2 as two `shape` records per knot: t = -R: a = -cR, b = c, and t = 0: a = 0, b = -c | approximation: the roll is linear between knots, while atan of a linear slope is not quite linear, and OpenDRIVE measures t in the rolled plane, not horizontally. The error is second order in the slope (about 0.2% of the crown height at 6% banking). |
| `width_left`, `width_right`, `LaneGroup` | a `laneSection` at every width knot and lane-group boundary. Per side, from the reference line outward: inner shoulder, `count` driving lanes, outer shoulder, each a linear `width` polynomial. Nominal widths are scaled so they sum to the pavement width, which keeps the pavement edge exact. | pavement edge exact (tested). Without a lane group, one `driving` lane per side covers the full width (FR-IN-5: lane counts not required). |
| `Road.oneway` | left-side lanes get `direction="reversed"` (they travel with the reference line) | |
| `Barrier` | `objects/object type="barrier"` with a continuous `repeat` (distance 0) at t = ±(offset + base/2) | position and height |
| `Structure` (bridge, culvert) | `objects/bridge s length id name type` on the carried road. `type` is `steel` for truss, else `concrete`; `name` is the inventory id when matched. | span exact; the full record is in the bridge's userData |
| `Structure` (tunnel portal) | `objects/tunnel type="standard"` | |
| `JunctionArea` (L6, milestone 2) | `junction type="default"` with a `connection` for each model link between member roads; the outline is in userData | connections only; no lane links or boundary yet |
| `Crossing`, `UnitRecord`, `Frame` | header `userData` | |
| everything | `userData code="roader"` holding the canonical JSON of each road, structure and junction (D-011 rounding) | lossless for L0 to L3 and L6 |

## Model to CityJSON

| Model element | CityObject | Geometry | Semantics / attributes |
|---|---|---|---|
| `Road` | `Road` | one `CompositeSurface` per pavement mesh (its closed shell) | `TrafficArea` on upward faces; name, class, kind, oneway, length |
| `SideSlope` (L4) | `Road` child of its road | `CompositeSurface` | `function="earthwork"`, side, kind, ratio, stations |
| `JunctionArea` (L6) | `Road` | `CompositeSurface` | `TrafficArea`; `function="junction"`, road ids |
| `Structure` bridge / culvert | `Bridge` | none (its parts carry the geometry) | kind, family, carried and crossed roads, stations, inventory id, review flag |
| structure meshes (deck, girders) | `BridgeConstructiveElement`, id = mesh id | `Solid` | `function` = part |
| `Support` | `BridgeConstructiveElement` | `Solid`, or `MultiSolid` when columns and cap are separate meshes | abutment or pier, station, skew, column count |
| parapet `Barrier` on a structure | `BridgeConstructiveElement` | `Solid` | kind, side, stations, height |
| other `Barrier` | `CityFurniture` | `Solid` | kind, road id |
| `Structure` tunnel portal | `Tunnel` with `TunnelConstructiveElement` parts | as for bridges | |
| L7 terrain mesh | `TINRelief` | `CompositeSurface` of triangles | |
| mesh with no model element | `GenericCityObject`, id = mesh id | `Solid` or `CompositeSurface` | `unmatched=true` |

Every object carries `element_id`, `source` (measured, inventory, vector, rule),
`confidence`, `layer` and `mesh_ids` (FR-RH-2, AC-H6). For roads, the confidence is
the lowest confidence among the road's alignment and profile elements.

## Known limits

- **CRS (CityJSON).** The CityJSON 2.0 schema requires `metadata.referenceSystem` to
  be an OGC URL (`https://www.opengis.net/def/crs/...`). The unit frame is a custom
  transverse Mercator frame with no EPSG code, so `referenceSystem` is left out and
  the PROJ string and anchor are written to `metadata.roaderFrame`. Tools that need
  an EPSG CRS have to reproject. A later option is to publish in a state-plane or UTM
  EPSG CRS plus NAVD88 (a compound EPSG code).
- **Road solids (CityJSON).** CityJSON 2.0 allows only MultiLineString, MultiSurface
  and CompositeSurface on Transportation objects. Closed pavement and earthwork
  solids are therefore written as the CompositeSurface of their closed shell
  (`attributes.closed_shell = true`). Bridge parts are true `Solid`s.
- **Validation.** CityJSON is checked structurally by `validate_cityjson`: vertex
  indices, allowed types and geometry per type, parents and children, semantic
  surface types. cjio or the official JSON-schema validator could be added later as
  a test oracle (it would need a THIRD_PARTY entry). USD is written as text and is
  not opened with usd-core; usd-core (`pxr`) could be added as a test-only oracle.
- **glTF.** One primitive per mesh. No instancing yet (FR-AS-8). No textures: UV
  sets are written, materials are still flat colors.
- **USD size.** Per-corner primvars and four LOD levels make `meshes.usda` large
  (about 55 MB for the M1 site); a binary `.usdc` would need usd-core.
  The tile-local origin is the unit origin, which keeps float32 error under about 1 mm
  within roughly 8 km of the anchor.
- **OpenDRIVE.** Lateral profile is approximate (see above). The median (`Median`) is
  not modelled as a lane; it is in userData. One predecessor/successor per road end;
  splits and merges need the milestone-2 junction work (connecting roads, lane links).
- **Report.** The point overlay projects points onto each road with a nearest-sample
  search and keeps points within 1 m vertically of the fitted surface, which
  separates stacked sheets for display only. It is not the AC-A1 held-out metric,
  which comes from the validate lane's GateReport. The FR-AS-2 freeway budget (stated
  for four-lane freeway) is applied to every mainline road.
- **PNG determinism** holds for a fixed matplotlib and freetype build; the versions
  are recorded in `report.json`.
