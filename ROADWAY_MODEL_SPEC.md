# Roadway Infrastructure Model (RIM): Build Specification

You are building a system that turns public airborne lidar plus 2D road vectors into a parametric, engineering-plausible model of highways, ramps, interchanges and bridges, and then generates closed solid meshes from that model.

Read this whole document before writing code. It tells you what to build, how well it has to work, and in what order. Where it gives a number, the number is a starting value you may tune unless section 9 lists it as an acceptance threshold.

## 0. How to work on this

**The core idea.** The parametric model is the product. Meshes are a deterministic rendering of it. Never build a mesh directly from points, and never repair a mesh into validity after the fact. If a solid comes out invalid, fix the generator.

**Three principles that decide most design questions:**

1. Vectors first, lidar second. Start from 2D road centerlines and use lidar to measure, correct and lift them into 3D. Do not try to discover the road network from points alone.
2. Keep a parametric model in the middle. Store alignment, profile, cross-section and structure parameters; generate geometry from those.
3. Everything carries provenance. Every parameter is tagged as measured, taken from an inventory record, taken from a vector hint, or filled by rule, with a confidence score.

**Build order.** Work through the milestones in section 10 in order. Do not start a milestone until the previous one passes its gates. Within a milestone, write the validity and accuracy tests from section 9 first, then the code that passes them.

**Requirement language.** "Shall" is mandatory. "Should" is a target. "May" is optional. Requirement IDs (for example FR-SG-3) are stable; use them in test names, commit messages and code comments so coverage can be traced.

**When something is unclear.** Choose the option that keeps the parametric model authoritative and the output deterministic, note the decision in `DECISIONS.md` with the requirement ID it touches, and continue. Stop and ask only if a choice would change an acceptance threshold or the data model in section 5.

**Suggested stack.** This is a default, not a requirement; change it if you have a reason and record why.

- Python for the pipeline and model; a compiled extension only where profiling demands it.
- PDAL or laspy for point I/O; pyproj for reference frames; numpy and scipy for fitting; shapely for 2D geometry.
- A robust mesh kernel with exact or guaranteed-manifold booleans for the few boolean operations.
- val3dity as an external test oracle for solid validity, alongside your own native checks.
- pytest; a small set of real lidar tiles checked into a fixtures cache, not the repository.

**Suggested repository layout.**

```
rim/
  io/          point, vector, inventory readers; reference frames
  evidence/    road-surface labelling, layer separation, barrier detection
  conflate/    vector merge, registration, structure extents, crossings
  fit/         alignment, profile, cross-section, joint solver
  complete/    occlusion completion, under-bridge ground
  structures/  inventory matching, structure inference
  rules/       design-convention rule library (data files + loader)
  model/       the section 5 schema, serialization, provenance
  solids/      sweep, junction partition, terrain conformance
  assets/      LOD, UVs, materials, instancing, collision
  validate/    solid checks, accuracy metrics, reports
  export/      USD, glTF, OpenDRIVE, CityJSON
tests/
DECISIONS.md
```

## 1. Scope

**In scope for v1:** United States limited-access highways, their ramps and interchanges, every grade separation on them, and the crossing surface streets within 150 m of those structures.

### Goals

| ID | Goal | Measure |
| --- | --- | --- |
| G1 | Every structure is a closed solid | 100% of exported solids pass the validity tests in section 7 |
| G2 | Geometry follows the real road | Deck and pavement surfaces within the section 9 tolerances of held-out lidar returns |
| G3 | Voids are real | Space under every bridge deck is open; no draped geometry from deck edge to ground |
| G4 | It looks engineered, not scanned | Smooth parametric sweeps with crisp edges; no lidar noise in geometry |
| G5 | It is a game-ready asset | LODs, UVs, PBR materials, collision proxies, instanced repeated parts, stable IDs |
| G6 | It runs unattended at state scale | A full state's highway network processes with no manual edits; every element has a confidence score |
| G7 | It is reproducible | Same inputs and version produce byte-identical output |

### Non-goals for v1

- Lane-level HD-map accuracy (centimeter lane boundaries need mobile lidar).
- Signals, signs and street furniture placed from observation. They may be placed by rule.
- Tunnel interiors. Model portals only; lidar cannot see inside.
- Structural-engineering fidelity (rebar, as-built girder sizes).
- Local residential streets, except where they cross in-scope roads.

### Definitions

- **Solid:** a closed, orientable, 2-manifold mesh with outward normals and positive volume, representing one physical body.
- **Watertight:** every edge is shared by exactly two faces of the same solid, and no face intersects another face of that solid except along shared edges.
- **Game-asset-like:** usable in a real-time engine without cleanup: controlled triangle budgets, an LOD chain, non-overlapping UVs, tiling PBR materials, collision meshes, instancing.
- **Ultra-realistic:** measured fidelity to the real site, plus plausibility of everything that was not measured, achieved by obeying highway and bridge design conventions.
- **Observed vs. inferred:** the provenance tag on every parameter. It travels to every export.

## 2. Why this is hard

Keep these in mind; most requirements exist because of one of them.

- **Nothing is measured under a deck.** Airborne lidar sees the top of a bridge and the ground beside it. The road beneath, the underside of the deck, and most piers are unobserved and must be completed from the model and from rules.
- **Roads overlap in plan.** An interchange is several surfaces stacked at one map location. Any 2.5D height-field approach collapses them.
- **Bridge decks are not planes.** They curve, bank and change grade. Plane-fitting methods from building reconstruction fail; fit along the centerline.
- **Inputs disagree.** Vectors are several meters off, bridge tags start and end in the wrong place, classification codes are unreliable, and lidar collections differ in year and density.
- **Junctions break swept meshes.** Where ramps split or merge, independently swept roads overlap and self-intersect.

## 3. Inputs

Each input has known defects. The defects column is a list of things your code must survive.

| Input | Role | Key facts | Defects to design for |
| --- | --- | --- | --- |
| USGS 3DEP lidar point clouds (LAS 1.4 / LAZ, cloud-optimized variants) | Primary geometry measurement | Quality level QL2: at least 2 pulses per m². QL1 and QL0: at least 8. Non-vegetated vertical RMSE at most 0.10 m for QL1 and QL2. Minimum class codes: 1 unclassified, 2 bare earth, 7 low noise, 9 water, 17 bridge deck. | Older collections lack the bridge class or use code 17 for something else. Deck points are sometimes left as ground. Vehicles sit in class 1 on the road. Adjacent projects differ in year, density and vertical datum realization. |
| 3DEP bare-earth DEM | Terrain away from roads | Bridges are deliberately removed; the surface under a bridge is interpolated from terrain beside it. | Embankments near abutments are cut inconsistently between collections. |
| Road vectors: Overture transportation segments, OpenStreetMap, Census TIGER, state DOT centerlines | Topology, classification, hints | Segments carry class, bridge and tunnel flags, a relative stacking level, width and surface hints. | Stacking level is a rendering hint, not an elevation. Horizontal error of several meters. Bridge extents wrong. Lane counts missing or stale. |
| FHWA National Bridge Inventory | Per-structure priors | Structure length, out-to-out deck width, curb-to-curb width, skew, number of main and approach spans, maximum span length, clearances, structure type and material, year built. | Point locations are approximate. Two record schemas exist (the older coding guide and the newer SNBI); parse both. |
| Public aerial imagery (NAIP or state orthophotos) | Optional: markings, pavement type, material color | | Lean, shadows, vehicles; date mismatch with lidar. |

### Input requirements

- FR-IN-1. Read LAS 1.4, LAZ and cloud-optimized point formats, by tile and by spatial query.
- FR-IN-2. Record per work unit: collection year, quality level, measured point density on pavement, and horizontal and vertical reference frames. Convert everything to one 3D reference frame before fusion.
- FR-IN-3. Treat every classification code other than ground as untrusted evidence. Re-derive road and deck membership yourself (stage 3).
- FR-IN-4. Where two collections overlap, prefer the newer for geometry, and flag structures that appear in only one.
- FR-IN-5. Work with any single vector source; improve when several agree. Do not require lane counts.
- FR-IN-6. Imagery is optional. Nothing in sections 6 or 7 may depend on it.
- FR-IN-7. Keep each vector source's attributes separable by source, so any source can be switched off by configuration.

## 4. Pipeline overview

Twelve stages. Stages 1 to 8 build the parametric model from data. Stages 9 to 12 turn it into assets. Every stage reads and writes the section 5 model, so any stage can be rerun alone.

1. Work planning
2. Point normalization
3. Road-surface evidence
4. Vector conflation and registration
5. Crossing resolution
6. Alignment and profile fitting
7. Cross-section estimation
8. Completing what was not observed
9. Structure inference
10. Solid generation
11. Surface detail and materials
12. Validation, confidence and export

## 5. Data model

Implement this as your own schema. Every mesh must be regenerable from it. The layering mirrors how highway engineers describe a road.

| Layer | Contents | Continuity and constraints |
| --- | --- | --- |
| L0 Network | Directed graph of carriageways, ramps and connectors. Stable IDs. Per-crossing vertical order (who is above whom), measured, not taken from tags. | Input topology is preserved, or changes are logged. |
| L1 Horizontal alignment | Per road: a reference line in plan as an ordered chain of straight, circular-arc and spiral (linearly varying curvature) elements. Parametric cubics allowed where spirals do not fit. | Position and heading continuous everywhere. Curvature continuous on freeway mainlines. |
| L2 Vertical profile | Elevation as a function of distance along the reference line: constant grades joined by parabolic vertical curves. | Elevation and grade continuous. Within the design envelope for the road class unless measured data proves otherwise. |
| L3 Cross-section | As a function of distance along the line: lane count and widths, shoulders, median type and width, barriers, curbs, cross slope per lane, banking (superelevation) and its transitions. | Widths change only through tapers. Banking changes only through transition lengths. |
| L4 Earthwork | Side slopes, ditches, embankment and cut faces tying pavement edge to terrain. Retaining walls where slope limits are exceeded. | The tie-in line lies on the terrain surface. |
| L5 Structures | Bridge: deck slab, girders or box, soffit, parapets and railings, expansion joints, piers or bents with caps and columns, abutments, wingwalls, approach slabs, slope paving. Also culverts, sign gantries (optional), tunnel portals. | Each part is one solid. Parts reference the L1 to L3 geometry of the road they carry. |
| L6 Junction areas | Gore areas at ramp splits and merges, at-grade intersections, lane connectivity through them. | Joining road surfaces meet with matching elevation and slope along the seam. |
| L7 Terrain patch | Bare-earth surface re-conformed to meet road and structure solids exactly, including the true ground under bridges. | No terrain inside any solid; no gap between terrain and solids. |
| L8 Provenance | For every parameter: source (measured, inventory, vector hint, rule), fit residual statistics, point count and density used, confidence 0 to 1. | Mandatory. Exported with every format. |

### Model requirements

- FR-DM-1. Layers L0 to L3 and L6 shall export losslessly to ASAM OpenDRIVE 1.8: plan-view geometry, elevation profile, superelevation, lateral shape, lanes, junctions, and bridges and tunnels as objects on the carrying road.
- FR-DM-2. Solids and semantics shall export to CityJSON (CityGML 3.0 conceptual model) using the Transportation and Bridge modules.
- FR-DM-3. Every exported mesh primitive carries the ID of the model element and layer that produced it.
- FR-DM-4. Editing a parameter and regenerating changes only the affected solids and their neighbors' seams.
- FR-DM-5. Meters and radians. One right-handed 3D frame per tile with a documented geodetic anchor. Double precision in the model; tile-local single precision permitted in meshes.

## 6. Functional requirements by stage

### Stage 1. Work planning

- FR-PL-1. Partition work by road corridor, not by lidar tile. A corridor is the vector network buffered by 150 m.
- FR-PL-2. A structure (bridge, interchange) is atomic and is never split across work units. Units overlap, and seams fall on plain road segments.
- FR-PL-3. Processing is deterministic and idempotent per unit, so units run in parallel and reruns reproduce bytes.

### Stage 2. Point normalization

- FR-PN-1. Transform points to the working frame. Remove flagged noise and withheld points. Keep return number, intensity, scan angle, time and source ID.
- FR-PN-2. Measure flight-strip vertical disagreement on pavement. Above 5 cm, correct per strip or lower the confidence of everything derived there.
- FR-PN-3. Compute per-cell density on candidate pavement. Mark cells under 2 points per m² sparse, and under 0.5 unobserved.

### Stage 3. Road-surface evidence

- FR-RS-1. Label each point with the probability that it lies on a paved travel surface, from local geometry: smoothness, low roughness, slope within road limits, lateral extent, intensity consistency. Vendor class codes are one feature among others.
- FR-RS-2. **Separate vertical layers.** Within a corridor, group points into continuous surface sheets such that two sheets may overlap in plan at different heights. This turns one 3D problem into several single-layer problems. Everything downstream depends on it.
- FR-RS-3. Remove transient objects. Returns roughly 0.3 m to 4.5 m above a locally fitted road sheet with a vehicle-sized footprint are excluded from surface fitting and never meshed.
- FR-RS-4. Detect linear raised features beside the sheet (barriers, parapets, guardrail, curbs) as their own evidence class, with height and offset.
- FR-RS-5. A learned point classifier may be used. Start with geometric features; add learning only if the geometric approach misses a section 9 target. Evaluate any classifier per lidar vintage and density class.

### Stage 4. Vector conflation and registration

- FR-VC-1. Merge vector sources into one graph. Keep each source's attributes with attribution.
- FR-VC-2. Assign each graph edge to exactly one surface sheet per location. An edge with no supporting sheet is flagged (unbuilt, demolished, tunnel or occluded), not dropped.
- FR-VC-3. Shift and bend each centerline to the measured pavement. Estimate lateral offset from cross-profiles taken every 5 to 10 m, then apply it as a smooth correction. Residual lateral error target: under 0.5 m.
- FR-VC-4. Ignore tagged bridge extents. Locate each structure's true start and end from data: the stations where the carrying sheet separates from the surrounding ground and a void or lower sheet appears beneath or beside it.
- FR-VC-5. Report surface sheets with no matching vector (new ramps, frontage roads) as candidates with geometry.

### Stage 5. Crossing resolution

- FR-CR-1. For every place two edges cross in plan, record which is above, the measured surface-to-surface height difference, and the crossing angle (skew).
- FR-CR-2. Vertical order comes from measurement. Log disagreement with vector tags, with both values.
- FR-CR-3. Group chains of adjacent crossings on one road (a viaduct over several roads) into one candidate structure when the carrying sheet never rejoins the ground between them.

### Stage 6. Alignment and profile fitting

- FR-AF-1. Fit the horizontal alignment (L1) to the registered centerline as the fewest straight, arc and spiral elements that keep deviation under 0.25 m, with heading continuity.
- FR-AF-2. Fit the vertical profile (L2) to road-sheet elevations sampled along the alignment, as grades joined by parabolic curves, using a robust estimator that ignores residual vehicles and noise.
- FR-AF-3. Solve fits jointly over each connected component, not road by road, subject to: equal elevation and slope where roads join; measured clearance preserved at every crossing; profile smoothness.
- FR-AF-4. Design-convention limits (section 8) enter as soft penalties. Dense, consistent measurements override them. Record the override.
- FR-AF-5. Report per element: residual RMS, 95th-percentile residual, and the fraction of length supported by observed (not interpolated) points.

### Stage 7. Cross-section estimation

- FR-XS-1. Find pavement edges at each sampled station from elevation breaks, roughness change, intensity change and barrier evidence. Fit edge lines along the road, not station by station, so widths vary smoothly.
- FR-XS-2. Estimate cross slope and banking from the sheet at each station. Fit transitions as continuous functions of distance.
- FR-XS-3. Infer lane count and shoulder widths from total paved width, road class and any vector hints. Tag as inferred.
- FR-XS-4. Classify the median: none, painted, raised, depressed, barrier-separated. Model twin carriageways with independent profiles when their elevations differ.
- FR-XS-5. On structures, measure deck outline and out-to-out width directly from the deck sheet and parapet evidence, and reconcile with the inventory width. Flag disagreement over 1 m.

### Stage 8. Completing what was not observed

- FR-OC-1. Where a road passes under a deck or dense canopy, carry its fitted alignment, profile and cross-section through the gap from the observed sections on both sides. Never invent a sag, bump or width change in a gap.
- FR-OC-2. Check every completed underpass against the structure above. The clearance from the lower road surface to the inferred soffit must be physically possible: positive, and normally at least the class minimum. On failure, refit, then flag.
- FR-OC-3. Reconstruct ground under bridges as engineered shapes: abutment face, sloped embankment or wall, lower roadway and its side slopes. Never a draped interpolation.
- FR-OC-4. Tunnels: model portals and the approach cut from data. Connect portals with a rule-based bore. Tag the interior as inferred.

### Stage 9. Structure inference

- FR-ST-1. Match each detected structure to an inventory record by location, carried and crossed route, and length. Unmatched structures proceed on rules alone.
- FR-ST-2. Choose a structure family (slab, multi-girder, box girder, truss, arch, culvert) from the inventory type and material codes. Default by span length and era when absent.
- FR-ST-3. Place abutments at the measured deck ends. Place intermediate supports using, in priority order: directly observed pier or cap returns; inventory span count and maximum span; the layout below (supports only in medians, gores and beyond shoulders, never in a travel lane or waterway channel unless observed); even spacing.
- FR-ST-4. Align supports with the skew of what they cross.
- FR-ST-5. Size superstructure depth, pier dimensions, parapet type and bearings by rule from span, width, family and era. Fit above-deck elements that lidar does observe (truss members, arches, towers, cables) from points, with rule-based fill for the rest.
- FR-ST-6. Route signature structures above a size threshold (long-span, movable, cable-supported) to a review queue instead of auto-publishing.

### Stage 10. Solid generation

See section 7.

### Stage 11. Surface detail and materials

See section 8.

### Stage 12. Validation, confidence and export

- FR-EX-1. Run every section 9 test on every unit. A unit publishes only if all hard gates pass. Otherwise it publishes at reduced LOD with a flag, or is held.
- FR-EX-2. Roll parameter confidences up to per-structure and per-kilometer scores with a documented formula.
- FR-EX-3. Export meshes as USD and glTF (tiled, streamable), the road description as OpenDRIVE, semantics as CityJSON, and the native model. All exports of one unit share IDs.
- FR-EX-4. Produce a per-unit report: inputs and versions, coverage, flags, metric table, and overlays of fitted surfaces against points.

### Non-functional requirements

- NFR-1. Throughput target: one large state's limited-access network within 24 hours on commodity cloud compute. Report cost per kilometer.
- NFR-2. Incremental update: when a new lidar collection arrives, only affected units recompute, and unchanged solids keep their IDs.
- NFR-3. No manual step on the main path. Human review is a queue fed by flags.

## 7. Solid geometry and assets

Construct solids; never recover them. Each body is generated as a closed shape from model parameters (a profile swept along an alignment, an extruded footprint, a lofted pier), so it is watertight by construction.

### Construction rules

- FR-SG-1. **Swept bodies.** Pavement, deck, girders, parapets, barriers, curbs and retaining walls are closed 2D cross-section profiles swept along the L1/L2 curve with L3 banking applied, capped at both ends. The profile may vary along the sweep (tapers, widening, changing banking) by interpolating between compatible profiles with the same vertex count.
- FR-SG-2. **Sampling.** Place sweep stations adaptively so chord deviation from the true curve stays under the LOD tolerance, with extra stations at every parameter change. Neighboring bodies that share a boundary use the same stations.
- FR-SG-3. **Junction areas.** Build gores, merges and intersections from a 2D planar partition of the lane and edge lines, then lift it to 3D and thicken it. A planar partition cannot overlap itself, which removes the usual junction failure of swept roads.
- FR-SG-4. **Finite thickness everywhere.** Pavement is a slab with a real underside, not a ribbon. Decks have a soffit. Barriers have a base. No single-sided surfaces in any solid.
- FR-SG-5. **Booleans are a last resort.** Where one is unavoidable (pier into sloped ground, skewed abutment), use exact or provably robust arithmetic, and run the FR-SV tests on the result before use.
- FR-SG-6. **Terrain.** The L7 terrain patch is a surface re-triangulated with road and structure footprints as constraint edges, so its boundary vertices coincide with solid vertices. On request it exports as a closed block with skirt and base.

### Validity tests (hard gates)

These follow the ISO 19107 solid definition. Implement them natively and also run val3dity as an independent check.

| ID | Test | Pass condition |
| --- | --- | --- |
| FR-SV-1 | Closed shell | Every edge has exactly two incident faces |
| FR-SV-2 | Manifold | Faces around every vertex form a single fan; no edge with more than two faces |
| FR-SV-3 | Orientation | Consistent winding; normals outward; signed volume positive |
| FR-SV-4 | No self-intersection | No two faces of a solid intersect except at shared edges or vertices |
| FR-SV-5 | No degeneracy | No zero-area face, no duplicate vertex, no edge under 1 mm, no T-junction |
| FR-SV-6 | Single component | One connected shell per solid unless inner voids are declared |
| FR-SV-7 | Solid-to-solid | Per export profile below |
| FR-SV-8 | Tile seams | Vertices on a shared tile boundary are bit-identical on both sides |

### Two export profiles for how solids meet

| Profile | Rule | Use |
| --- | --- | --- |
| Analytic | Interiors of different solids never overlap. Touching solids share conforming faces. The union of a structure's parts is itself one valid solid. | Simulation, line-of-sight and obstruction analysis, volume queries, CAD and GIS exchange |
| Real-time | Parts may embed into neighbors by a declared minimum (default 0.2 m) so no seam is visible. No coplanar overlapping faces anywhere. | Game engines, visualization |

Generate both from the same model. Build the analytic profile first; it is the reference for acceptance.

### Asset requirements

- FR-AS-1. **Levels of detail.** Four levels per solid, each independently watertight. Chord tolerance 1 cm, 5 cm, 25 cm, 1 m. Voids under decks survive until the structure is replaced by its coarsest proxy. Silhouette and deck elevation are preserved at every level.
- FR-AS-2. **Budgets.** At the finest level: under 150,000 triangles per kilometer of four-lane freeway on grade, and under 60,000 per ordinary overpass including substructure. Report actuals. Treat a threefold overrun as a defect.
- FR-AS-3. **Shading normals.** Hard edges where the model has a crease (slab edge, barrier arris); smooth along sweeps. Derive them from the model, not from an angle threshold.
- FR-AS-4. **UV set 0, tiling.** Parameterize by distance along the road and offset across it, in meters, so pavement, barrier and concrete textures tile without stretch through curves and tapers. Continuous across segment and tile seams.
- FR-AS-5. **UV set 1, unique.** Non-overlapping atlas per asset for baked lighting and wear masks, with stated texel density and padding.
- FR-AS-6. **Materials.** Metallic-roughness PBR. One material slot per semantic surface type (travel-lane asphalt, concrete pavement, shoulder, deck, structural concrete, steel, soil slope, slope paving). No per-asset unique materials on the main path.
- FR-AS-7. **Markings and stains are not geometry.** Lane lines, joints, patches and weathering are decals or mask layers above the solid, so the solid stays simple and valid.
- FR-AS-8. **Instancing.** Repeated parts (barrier units, railing posts, bearings, identical columns) export as instances of shared prototypes.
- FR-AS-9. **Collision.** A drivable-surface collision mesh equal to the smooth model surface, plus convex proxies for structures. Collision surfaces carry the surface type.
- FR-AS-10. **Hygiene.** Tile-local origin with documented geodetic anchor; meters; consistent up-axis per format; stable human-readable names built from model IDs; no orphan nodes or empty meshes.

## 8. Realism

Realism comes from being right about shape and from obeying the conventions real roads are built to. It does not come from adding noise. Geometry is clean and deterministic; all irregularity lives in materials.

### Design-convention rule library

- FR-RL-1. All priors live in a data-driven rule library keyed by road class, jurisdiction and construction era. Do not hard-code conventions in pipeline logic.
- FR-RL-2. Each rule states a typical value, a permitted range, and where the value came from.
- FR-RL-3. Rules act as soft priors during fitting (stage 6) and as fill values where nothing was observed (stages 8 and 9). A measured value outside a rule's range is kept and flagged, never clamped.

Two values are confirmed and may seed the library: Interstate travel lanes are at least 12 ft (3.66 m) wide, and Interstate vertical clearance is at least 16 ft (4.9 m) in rural areas and on one urban routing, 14 ft (4.3 m) on other urban routes.

The rule families below are required. The magnitudes are approximate and unverified; they show scale only. Treat them as placeholders in the rule data files, marked as such, until confirmed against the published design standards.

| Family | What it governs | Indicative scale (placeholder) |
| --- | --- | --- |
| Lanes and shoulders | Lane width; right and left shoulder width by lane count | Lanes about 3.6 m; right shoulder about 3 m |
| Cross slope and banking | Normal crown, maximum banking, transition length by speed | Crown about 2%; banking up to about 6 to 8% |
| Grades and vertical curves | Maximum grade by class and terrain; minimum curve length by speed | Freeway grades mostly under 6% |
| Horizontal curvature | Minimum radius by speed and banking; spiral use | By design-speed table |
| Ramps | Lane width, taper and gore geometry, merge and diverge lengths | By ramp type |
| Earthwork | Fill and cut slope ratios; where walls replace slopes | Slopes about 1:2 to 1:6 |
| Clearances | Vertical and lateral clearance under structures | Confirmed values above |
| Superstructure | Depth as a ratio of span by family; girder spacing; deck overhang | Depth roughly span / 20 to span / 25 for girder bridges |
| Substructure | Column count and size by deck width; cap shape; pier type by era | Columns roughly 1 to 2 m across |
| Barriers and rails | Profile family and height by era and jurisdiction | Roughly 0.8 to 1.1 m tall |
| Pavement | Asphalt vs. concrete; joint spacing for concrete slabs | Joints every few meters |
| Markings | Line widths, dash and gap lengths, colors by side | Per the federal marking manual (MUTCD) |

### Structure detail

- FR-RD-1. At the finest LOD every bridge includes: deck with overhangs and drip edge, parapets or rails, expansion joints at deck ends and at inventory-indicated intermediate joints, girders or box with the correct count for the width, bearings, pier caps, columns, abutment seat and backwall, wingwalls, approach slabs, and slope protection under end spans.
- FR-RD-2. All exposed edges carry a small bevel or chamfer consistent with formed concrete or rolled steel. No zero-radius silhouette edges at the finest LOD.
- FR-RD-3. Part proportions vary by era. A 1960s overpass and a 2015 overpass of the same span shall not look alike.
- FR-RD-4. Parts the lidar did observe (deck outline, parapet height, truss and arch members, sign gantries) take priority over library parts.

### Surface detail

- FR-RM-1. Choose materials per surface from evidence where possible: lidar intensity statistics, vector surface hints, inventory deck surface codes, optional imagery.
- FR-RM-2. Wear is procedural and structured: wheel-path polishing and darkening per lane, center-of-lane staining, joint and crack sealing, patching, edge raveling, tire marking at ramps, water staining below deck joints and drains, efflorescence on old concrete. Intensity scales with age from the inventory year built.
- FR-RM-3. Fine relief (texture, tining, rumble strips, joints) is carried by normal and height maps, not triangles.
- FR-RM-4. Generate markings by rule from lane topology: edge lines, lane lines, gore chevrons, merge and exit patterns. Their wear follows FR-RM-2.
- FR-RM-5. Imagery, when enabled, drives masks and color statistics only. Never apply it directly as surface color; it contains shadows, vehicles and lean.
- FR-RM-6. Seed all procedural variation from stable model IDs so output is reproducible.

### Honest presentation

- FR-RH-1. Provide a debug material mode that colors every surface by provenance: measured, inventory, vector hint, rule.
- FR-RH-2. Exports carry the provenance layer so downstream users can filter on it.

## 9. Acceptance criteria

A milestone is done when every hard gate passes on every test site in its scope and every target is met on the pooled test set.

### Hard gates (100% required)

| ID | Gate | Method |
| --- | --- | --- |
| AC-H1 | Every solid passes FR-SV-1 to FR-SV-8 in the analytic profile | Native checker plus val3dity |
| AC-H2 | Vertical order is correct at every crossing | Compared with a hand-labeled crossing list per site |
| AC-H3 | Voids are open | Rays cast along every lower road beneath every deck hit no geometry between road surface and soffit |
| AC-H4 | No surface discontinuity | Elevation and grade continuous along every road; matching elevation and slope across every junction seam |
| AC-H5 | Determinism | Two runs on the same inputs produce identical hashes for all exports |
| AC-H6 | Provenance complete | Every parameter and every mesh primitive carries a source tag |

### Accuracy targets

Measure against lidar returns withheld from fitting in spatial blocks (20% of pavement area), and against independent survey or mobile-lidar data where a public set exists.

| ID | Quantity | Target |
| --- | --- | --- |
| AC-A1 | Pavement and deck surface, vertical, observed zones, QL1 and QL2 data | RMS at most 0.10 m; 95th percentile at most 0.20 m |
| AC-A2 | Same, QL3 or pre-2015 data | RMS at most 0.20 m |
| AC-A3 | Pavement edge and deck outline, horizontal | RMS at most 0.5 m |
| AC-A4 | Surface-to-surface height difference at crossings | Within 0.15 m of measured |
| AC-A5 | Structure start and end stations | Within 2 m |
| AC-A6 | Deck out-to-out width | Within 0.5 m of measured points |
| AC-A7 | Structure detection against in-scope inventory records | Recall and precision each at least 98% |
| AC-A8 | Span count where the inventory provides it | Matches in at least 95% of structures |
| AC-A9 | Occluded underpass profile, checked where a second lidar vintage or mobile data shows it | RMS at most 0.25 m |

### Plausibility and asset targets

- AC-P1. Zero violations of confirmed clearance minimums that are not backed by a measurement flag.
- AC-P2. No lane-width, banking or grade step changes; all transitions have nonzero length.
- AC-P3. Triangle budgets in FR-AS-2 met at the median, with no asset over three times budget.
- AC-P4. UV stretch along and across the road under 5% at the finest LOD.
- AC-P5. Automated visual-defect checks pass on renders from driver, low-aerial and under-bridge viewpoints: no floating parts, buried parts, z-fighting, kinked sweeps, razor edges, texture swim or supports in travel lanes. Save the renders in the unit report for human review.
- AC-P6. Confidence is calibrated: among elements scored above 0.9, at least 95% meet AC-A1 and AC-A3.

### Test-site classes

Pick at least one real site per class and record its location and lidar project in `tests/sites.yaml`. Confirm lidar coverage and vintage before committing to a site.

| Class | Why it is in the set |
| --- | --- |
| Simple overpass on a rural freeway | Baseline; first milestone |
| Diamond and cloverleaf interchanges | Ramps, gores, loop curvature and banking |
| Four- or five-level stack interchange | Many overlapping layers, heavy occlusion |
| Long urban viaduct | Chained crossings, repeated piers, instancing |
| Twin carriageways at different elevations | Independent profiles, median slopes |
| Major river crossing with truss or arch | Above-deck members, review-queue routing |
| Mountain section with deep cuts, fills and walls | Earthwork and retaining structures |
| Dense tree canopy over the road | Non-structural occlusion |
| QL3 or older lidar area | Sparse-data behavior |
| Construction mismatch (road newer than lidar, or two vintages disagree) | Change detection and flagging |
| Tunnel portal | Unobserved interior |
| Area with wrong or missing vector tags | Robustness to hints |

## 10. Milestones

Complete these in order. Each ends with a passing test run and a unit report.

1. **One overpass, end to end.** A single rural freeway overpass: stages 1 to 10 and 12, analytic profile, finest LOD only, flat-color materials. All hard gates pass. AC-A1, A3, A4 and A5 met on that site.
2. **One full interchange.** Adds layer separation under heavy overlap, joint fitting, junction-area construction, occlusion completion, both export profiles, the LOD chain and UVs. Test on a diamond or cloverleaf, then a stack.
3. **Structure and surface realism.** Inventory matching, the rule library, full bridge part set, materials, wear and markings. Sections 8 and FR-AS fully covered.
4. **A 100 km corridor, unattended.** Work planning, parallel units, tile seams, confidence roll-up, review queue, full metric report. All accuracy targets on the pooled set.
5. **One state network.** Throughput, cost per kilometer, incremental update.

## 11. Assumed defaults

These decisions were made so you can proceed. Record any change in `DECISIONS.md`.

- The analytic export profile is tuned first; real-time follows in milestone 2.
- v1 is lidar-and-vector only. Imagery support is built behind a switch and off by default.
- Surface streets beyond the 150 m crossing buffer are out of scope.
- Export targets are USD, glTF, OpenDRIVE 1.8 and CityJSON.
- The accuracy thresholds in section 9 stand as written.
