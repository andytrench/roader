# Roader rule library (spec section 8)

Design conventions live in data, not in pipeline code (FR-RL-1). Each rule gives a
typical value, a permitted range and where the value came from (FR-RL-2). Fitting
uses rules as soft priors, and later stages use them as fill values where nothing
was observed. A measured value outside the range is kept and flagged, never
clamped (FR-RL-3). Milestone 1 ships only the rules it needs; milestone 3 fills
in the full library (D-006).

Code: `src/roader/rules/book.py`. Data: `src/roader/rules/data/*.toml`, one file
per rule family.

## API

```python
from roader.rules import RuleBook, era_of

book = RuleBook.load()                      # every *.toml in roader/rules/data/
book = RuleBook.load(path)                  # a directory of *.toml, or one file
rule = book.get("lane.width", road_class="motorway", era="*", jurisdiction="*")
ok, rule = book.evaluate("grade", 0.071, road_class="motorway")   # never clamps
p = book.penalty("grade", 0.071, road_class="motorway")            # FR-AF-4
value, prov = book.fill("deck.overhang", era=era_of(1968))         # stages 8 and 9
```

- `get` returns the `Rule` row that matches most specifically (see the resolution
  order below). It raises `KeyError` for an unknown id or when no row matches.
- `evaluate` returns `(in_range, rule)`. The caller keeps its measured value. If
  the value is out of range, the caller adds a flag (for example `FR-AF-4:override`).
- `penalty` is 0 inside `[low, high]`. Outside the range it is
  `(distance_outside / (high - low))^2`. A zero-width range uses `max(|typical|, 1)`
  as the scale instead. The penalty is smooth at the range edges, so a fitting
  solver can add it to its objective with a weight of its choosing.
- `fill` returns `(rule.typical, Provenance(source=RULE, rule_id=..., confidence=rule.confidence))`.
  Placeholder rules also carry the flag `FR-RL-2:placeholder`.
- `era_of(year_built)` maps a construction year to an era key: `pre1970`,
  `1970_1994` or `1995_plus`. It returns `*` when the year is unknown.
- `RuleBook.ids()` and `RuleBook.rules()` list every id and every row.

## Resolution order

A row matches a query when each of its three keys either equals the query value
or is `*`. The road class can also match through its family:

- `class_chain("motorway_link")` is `motorway_link`, then `motorway`, then `*`.
- Any `X_link` class falls back to `X`, then to `*`.
- Every other class falls back straight to `*`.
- A query with class `*` matches only `*` rows.

If several rows match, the winner is decided in this order:

1. The most specific road class, by position in the class chain.
2. Then the jurisdiction: an exact match beats `*`.
3. Then the era: an exact match beats `*`.

The loader rejects two rows that share an id and all three keys. Because of that,
the winner is always unique and the result is deterministic. Every rule id has a
row with all keys set to `*`, so `get(rule_id)` with default keys always succeeds.
A test enforces this.

## Adding or changing a rule

1. Pick the family file in `src/roader/rules/data/`, or add a new `<family>.toml`
   with a `family = "..."` line and `[[rule]]` tables.
2. Each `[[rule]]` table needs these fields:
   - `id`
   - `typical`, `low`, `high`, which must satisfy `low <= typical <= high` and be finite
   - `unit`
   - `origin`
   - `placeholder`
3. These fields are optional:
   - `road_class`, `era`, `jurisdiction` (each defaults to `*`)
   - `confidence` (from 0 to 1; defaults to 0.3 for a placeholder and 0.5 for a
     confirmed rule)
   - `note`
4. The loader rejects any of the following:
   - unknown fields or missing required fields
   - non-numeric values, or `low > high`
   - `typical` outside the range
   - an era that is not one of `ERAS`
   - duplicate id and key combinations, including across files
   - an `origin` that disagrees with `placeholder` (see the next section)
5. Use dotted ids. The first part names the quantity
   (`lane.width`, `superstructure.depth_ratio.slab`). Put a qualifier that is not
   class, era or jurisdiction (structure family, pavement material, rural or urban)
   in the id.
6. Add the rule to the table below. A test checks that every id appears in this file.

Units: `m`; `ratio` (dimensionless, such as a slope magnitude, a grade or a
depth-to-span ratio); `h_per_v` (horizontal over vertical, so 3.0 means 3H:1V);
`m_per_pct` (the vertical-curve K value, meaning curve length per percent of
algebraic grade change).

How to read a range:

- Most rules describe a quantity. For example, `grade` is the absolute longitudinal
  grade, and `high` is the class maximum.
- `structure.*` and `superelevation.radius_*` are thresholds. Their `typical` is
  the threshold the pipeline uses, and the range bounds how far it can be tuned.

## Placeholder policy

- `placeholder = true` exactly when `origin` starts with `placeholder:`. The loader
  enforces this. Two origins are in use:
  - `placeholder: indicative scale from ROADWAY_MODEL_SPEC.md section 8`, where the
    spec's table gives a scale.
  - `placeholder: engineering judgement, unverified`, where it does not.
- Only the two confirmed values in spec section 8 have `placeholder = false`.
  Their origin is `FHWA, as confirmed in ROADWAY_MODEL_SPEC.md section 8`.
  - Their `low` and `typical` are the confirmed minimum.
  - Their `high` is a plausibility bound, not taken from the standard. Each row's
    `note` says so.
- `motorway` is used as a proxy for Interstate. Non-Interstate motorways resolve
  to the same row until a jurisdiction key separates them.
- Never write a publication into `origin` unless someone has read that publication
  and checked the value. To confirm a value, change its `origin` and `placeholder`
  in one commit that cites the source.
- Every value filled from a placeholder rule carries the flag `FR-RL-2:placeholder`
  in its provenance, so reports and the provenance debug view (FR-RH-1) can find it.

## Rule table

Origin key:

- **FHWA**: `FHWA, as confirmed in ROADWAY_MODEL_SPEC.md section 8`.
- **spec scale**: `placeholder: indicative scale from ROADWAY_MODEL_SPEC.md section 8`.
- **judgement**: `placeholder: engineering judgement, unverified`.

| id | class | era | typical | range | unit | placeholder | origin |
|---|---|---|---|---|---|---|---|
| `lane.width` | motorway | * | 3.658 | 3.658 to 4 | m | **no** | FHWA |
| `lane.width` | motorway_link | * | 4 | 3.6 to 5.5 | m | yes | judgement |
| `lane.width` | primary | * | 3.6 | 3 to 3.9 | m | yes | spec scale |
| `lane.width` | * | * | 3.6 | 3 to 4 | m | yes | spec scale |
| `shoulder.right.width` | motorway | * | 3 | 1.8 to 3.7 | m | yes | spec scale |
| `shoulder.right.width` | motorway_link | * | 2.4 | 1.2 to 3 | m | yes | judgement |
| `shoulder.right.width` | * | * | 2.4 | 0 to 3.7 | m | yes | judgement |
| `shoulder.left.width` | motorway | * | 1.2 | 0.6 to 3.7 | m | yes | judgement |
| `shoulder.left.width` | motorway_link | * | 1 | 0.6 to 1.8 | m | yes | judgement |
| `shoulder.left.width` | * | * | 0.6 | 0 to 3 | m | yes | judgement |
| `cross_slope.crown` | * | * | 0.02 | 0.015 to 0.025 | ratio | yes | spec scale |
| `superelevation.rate` | * | * | 0.06 | 0 to 0.08 | ratio | yes | spec scale |
| `superelevation.radius_crown` | * | * | 600 | 300 to 3000 | m | yes | judgement |
| `superelevation.radius_full` | * | * | 70 | 30 to 1000 | m | yes | judgement |
| `superelevation.relative_gradient` | * | * | 0.004 | 0 to 0.0065 | ratio | yes | judgement |
| `superelevation.runoff_length` | * | * | 45 | 20 to 200 | m | yes | judgement |
| `grade` | motorway | * | 0.02 | 0 to 0.06 | ratio | yes | spec scale |
| `grade` | motorway_link | * | 0.03 | 0 to 0.08 | ratio | yes | judgement |
| `grade` | primary | * | 0.03 | 0 to 0.06 | ratio | yes | judgement |
| `grade` | * | * | 0.03 | 0 to 0.12 | ratio | yes | judgement |
| `vcurve.k_crest` | motorway | * | 100 | 50 to 10000 | m_per_pct | yes | judgement |
| `vcurve.k_crest` | motorway_link | * | 30 | 8 to 10000 | m_per_pct | yes | judgement |
| `vcurve.k_crest` | * | * | 50 | 8 to 10000 | m_per_pct | yes | judgement |
| `vcurve.k_sag` | motorway | * | 70 | 40 to 10000 | m_per_pct | yes | judgement |
| `vcurve.k_sag` | motorway_link | * | 25 | 10 to 10000 | m_per_pct | yes | judgement |
| `vcurve.k_sag` | * | * | 40 | 10 to 10000 | m_per_pct | yes | judgement |
| `curve.radius` | motorway | * | 900 | 500 to 100000 | m | yes | judgement |
| `curve.radius` | motorway_link | * | 150 | 40 to 100000 | m | yes | judgement |
| `curve.radius` | * | * | 300 | 30 to 100000 | m | yes | judgement |
| `slope.fill_h_per_v` | * | * | 3 | 2 to 6 | h_per_v | yes | spec scale |
| `slope.cut_h_per_v` | * | * | 2 | 2 to 6 | h_per_v | yes | spec scale |
| `slope.max_height` | * | * | 6 | 3 to 15 | m | yes | judgement |
| `clearance.vertical.rural` | motorway | * | 4.877 | 4.877 to 6 | m | **no** | FHWA |
| `clearance.vertical.rural` | * | * | 4.877 | 4.3 to 6 | m | yes | judgement |
| `clearance.vertical.urban` | motorway | * | 4.267 | 4.267 to 6 | m | **no** | FHWA |
| `clearance.vertical.urban` | * | * | 4.3 | 4.1 to 6 | m | yes | judgement |
| `clearance.lateral.support` | * | * | 1.2 | 0.6 to 9 | m | yes | judgement |
| `superstructure.depth_ratio.slab` | * | * | 0.05 | 0.035 to 0.07 | ratio | yes | judgement |
| `superstructure.depth_ratio.multi_girder` | * | * | 0.045 | 0.04 to 0.05 | ratio | yes | spec scale |
| `superstructure.depth_ratio.box_girder` | * | * | 0.04 | 0.03 to 0.05 | ratio | yes | judgement |
| `deck.slab_thickness` | * | * | 0.2 | 0.18 to 0.3 | m | yes | judgement |
| `deck.overhang` | * | * | 1 | 0.5 to 1.8 | m | yes | judgement |
| `girder.spacing` | * | * | 2.7 | 1.5 to 3.7 | m | yes | judgement |
| `column.width` | * | * | 1.2 | 1 to 2 | m | yes | spec scale |
| `bent.width_per_column` | * | * | 6 | 4 to 12 | m | yes | judgement |
| `cap.depth` | * | * | 1.2 | 0.9 to 2 | m | yes | judgement |
| `cap.overhang` | * | * | 0.15 | 0.1 to 0.6 | m | yes | judgement |
| `abutment.backwall_height` | * | * | 1.8 | 1.2 to 3.5 | m | yes | judgement |
| `footing.depth` | * | * | 1.5 | 1 to 3 | m | yes | judgement |
| `parapet.height` | * | pre1970 | 0.8 | 0.6 to 1 | m | yes | judgement |
| `parapet.height` | * | 1970_1994 | 0.81 | 0.7 to 1.1 | m | yes | judgement |
| `parapet.height` | * | 1995_plus | 1.07 | 0.8 to 1.2 | m | yes | judgement |
| `parapet.height` | * | * | 0.9 | 0.8 to 1.1 | m | yes | spec scale |
| `parapet.base_width` | * | pre1970 | 0.45 | 0.3 to 0.7 | m | yes | judgement |
| `parapet.base_width` | * | * | 0.55 | 0.38 to 0.9 | m | yes | judgement |
| `parapet.top_width` | * | * | 0.2 | 0.15 to 0.3 | m | yes | judgement |
| `pavement.thickness.asphalt` | * | * | 0.3 | 0.15 to 0.6 | m | yes | judgement |
| `pavement.thickness.concrete` | * | * | 0.28 | 0.2 to 0.4 | m | yes | judgement |
| `structure.slab_max_span` | * | * | 12 | 9 to 15 | m | yes | judgement |
| `structure.review_span` | * | * | 150 | 100 to 300 | m | yes | judgement |

All jurisdiction keys are `*` in milestone 1.

Meaning of the less obvious ids:

- `clearance.vertical.rural` applies in rural areas and on the one urban routing
  that requires 16 ft. `clearance.vertical.urban` applies on all other urban routes.
- `bent.width_per_column` is the deck width carried by each column, so the number
  of columns per bent is `ceil(deck_width / typical)`.
- `cap.overhang` is how far a pier cap or abutment seat extends beyond the column
  face on each side, along the carried road, so the cap width is
  `column.width + 2 * cap.overhang`.
- `structure.slab_max_span`: when no inventory record matches, a span below this
  defaults to the slab family, and a longer one to `multi_girder` (FR-ST-2).
- `structure.review_span`: a maximum span at or above this sends the structure to
  the review queue (FR-ST-6).
- `superelevation.radius_crown` and `superelevation.radius_full` set how banking is
  filled by rule from curvature. A curve with a radius at or above `radius_crown`
  keeps the normal crown. A curve with a radius at or below `radius_full` takes
  `superelevation.rate` typical.

## Legacy constant map

The pre-spec code (formerly `src/roader/legacy/`, deleted after milestone 1) had
36 tuning constants:

- 27 module constants in `surface.py`
- 6 module constants in `assign.py`
- 3 default arguments of `decks.extract_decks` (`decks.py` has no module constants)

Each one is classified below:

- **Rule** means a design convention. It now lives in the rule library under the
  id given, and new code reads it from there.
- **Tolerance** means an algorithm setting. It stays in code as a named constant of
  the new stage that needs it.

The legacy code was deleted when milestone 1 passed its gates (D-004).

| Legacy constant | Value | Kind | Rule id, or new home |
|---|---|---|---|
| `surface.GAP_FILL_M` | 18.0 | Tolerance | Stage 8 `roader.complete`: longest gap carried through (FR-OC-1) |
| `surface.GAP_FAIL_M` | 25.0 | Tolerance | Stage 8 `roader.complete`: longer gaps are flagged, not invented (FR-OC-1) |
| `surface.LOW_PERCENTILE` | 15.0 | Tolerance | Stage 6 `roader.fit`: robust profile estimator (FR-AF-2) |
| `surface.RESIDUAL_M` | 0.20 | Tolerance | Stage 6 `roader.fit`: inlier residual for profile fitting (FR-AF-2) |
| `surface.PROFILE_TIE_M` | 0.12 | Tolerance | Stage 6 `roader.fit`: data-versus-prior override threshold (FR-AF-4) |
| `surface.CROWN` | 0.02 | Rule | `cross_slope.crown` |
| `surface.SE_MAX` | 0.06 | Rule | `superelevation.rate` (typical) |
| `surface.R_TANGENT_M` | 600.0 | Rule | `superelevation.radius_crown` |
| `surface.R_FULL_M` | 70.0 | Rule | `superelevation.radius_full` |
| `surface.RUNOFF_M` | 45.0 | Rule | `superelevation.runoff_length` |
| `surface.MAX_CROSS_SLOPE` | 0.08 | Rule | `superelevation.rate` (high) |
| `surface.NDVI_MAX` | 0.35 | Tolerance | Stage 3 `roader.evidence`, imagery feature behind its switch (FR-IN-6, D-009) |
| `surface.INTENSITY_KEEP_PCT` | 65.0 | Tolerance | Stage 3 `roader.evidence`: intensity-consistency feature (FR-RS-1) |
| `surface.DEAD_END_M` | 40.0 | Tolerance | Stage 4 `roader.conflate`: short dead-end vector filter (FR-VC-2) |
| `surface.DEAD_END_TOL_M` | 8.0 | Tolerance | Stage 4 `roader.conflate`: endpoint snap for dead-end test |
| `surface.CORRIDOR_MARGIN_M` | 3.0 | Tolerance | Stage 4 `roader.conflate`: search margin beyond the class width (FR-VC-3) |
| `surface.STATION_STEP_M` | 5.0 | Tolerance | Stage 4 `roader.conflate`: cross-profile spacing, 5 to 10 m (FR-VC-3) |
| `surface.MIN_DECK_RUN_M` | 18.0 | Tolerance | Stage 4 `roader.conflate`: shortest structure run (FR-VC-4) |
| `surface.FILL_RISE_M` | 1.0 | Tolerance | Stage 7 `roader.fit`: fill-versus-cut classification of side slopes (L4) |
| `surface.LIP_CLOSE_M` | 1.2 | Tolerance | Stage 7 `roader.fit`: pavement-lip attachment test |
| `surface.FILL_MAX_M` | 15.0 | Tolerance | Stage 4 `roader.conflate`: deeper drop is a missed span, not an embankment (FR-VC-4) |
| `surface.BANK_LIP_INSET_M` | 0.5 | Tolerance | Stage 7 `roader.fit`: side-slope sampling window |
| `surface.BANK_LIP_WIDTH_M` | 2.5 | Tolerance | Stage 7 `roader.fit`: side-slope sampling window |
| `surface.BANK_TOE_INSET_M` | 8.0 | Tolerance | Stage 7 `roader.fit`: side-slope sampling window |
| `surface.BANK_TOE_REACH_M` | 18.0 | Tolerance | Stage 7 `roader.fit`: side-slope sampling window |
| `surface.CLASS_HALF_WIDTH_M` | dict | Rule | Replaced by lane count × `lane.width` + `shoulder.right.width` + `shoulder.left.width` (FR-XS-3); the search band in stage 4 derives from those |
| `surface.max_grade_for` (function, also listed) | 0.06 / 0.08 / 0.12 | Rule | `grade` (high) per class |
| `assign._JOIN_M` | 8.0 | Tolerance | Stage 4 `roader.conflate`: endpoint join distance |
| `assign._HEADING_MIN` | 0.57 | Tolerance | Stage 4 `roader.conflate`: heading agreement for joins |
| `assign._RAMP_REACH_M` | 20.0 | Tolerance | Stage 4 `roader.conflate`: ramp end extension reach |
| `assign._COPY_OVERLAP` | 0.6 | Tolerance | Stage 4 `roader.conflate`: duplicate footprint overlap |
| `assign._COPY_Z_M` | 1.5 | Tolerance | Stage 4 `roader.conflate`: duplicate height match |
| `assign._CLASS_RANK` | dict | Tolerance | Stage 4 `roader.conflate`: class precedence when merging vectors (not a design value) |
| `decks.extract_decks(height_above_ground_m)` | 2.5 | Tolerance | Stage 3 `roader.evidence`: elevated-sheet seed height (FR-RS-2) |
| `decks.extract_decks(cell_m)` | 1.0 | Tolerance | Stage 3 `roader.evidence`: outline raster cell |
| `decks.extract_decks(min_points)` | 40 | Tolerance | Stage 3 `roader.evidence`: minimum deck cluster size |

`max_grade_for` is a function, not a constant. It is listed because its literal
grades are a design convention.

`class_half_limits` uses half of the class width as its lower bound. It goes away
together with `CLASS_HALF_WIDTH_M`.
