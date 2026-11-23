"""One unit end to end: pinned inputs, model, gates, published exports (stages 1 to 12).

``build_unit`` reads a unit config (``configs/units/<site>.toml``), checks the pinned
input hashes, fits the model with a spatial holdout (D-018), measures the accuracy
references, then renders, gates and publishes through ``export_model`` (FR-EX-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from roader.export.gltf import write_points_glb
from roader.export.run import export_model
from roader.io import (
    build_unit_record,
    load_unit_config,
    read_points,
    read_vectors,
    verify_inputs,
)
from roader.io.frame import FrameTransformer
from roader.model.schema import Frame, Road, RoadModel
from roader.pipeline import holdout_for, measure_edges, pavement_points, run_pipeline
from roader.validate.report import GateReport

POINTS_OVERLAY = "points_sample.glb"


@dataclass
class SiteReference:
    """Hand-labelled (or draft) references for one site (AC-H2, AC-A4, AC-A5)."""

    crossings: list[dict[str, str]] = field(default_factory=list)
    crossing_dz: dict[str, float] = field(default_factory=dict)
    crossing_points: dict[str, tuple[float, float]] = field(default_factory=dict)  # D-096
    structure_stations: dict[str, tuple[float, float]] = field(default_factory=dict)
    status: str = ""


def _road_for_way(model: RoadModel, way: int) -> Road | None:
    return next(
        (
            r
            for r in model.roads
            if any(ref.feature_id == str(way) for ref in r.vector_refs)
        ),
        None,
    )


def load_site_reference(
    path: str | Path, model: RoadModel, frame: Frame
) -> SiteReference:
    """Map a site crossing list (D-044 layout) onto the model's road, crossing and structure ids.

    Entries whose OSM ways are not modelled are skipped here; AC-H2 then reports the
    model crossings that the reference does not cover.
    """
    doc = yaml.safe_load(Path(path).read_text())
    ref = SiteReference(status=str(doc.get("status", "")))
    to_unit = FrameTransformer(frame)
    for entry in doc.get("crossings", []):
        upper = _road_for_way(model, entry["upper"]["osm_way"])
        lower = _road_for_way(model, entry["lower"]["osm_way"])
        if upper is None or lower is None:
            continue
        ref.crossings.append({"upper_road": upper.id, "lower_road": lower.id})
        crossing = next(
            (
                c
                for c in model.crossings
                if {c.upper_road, c.lower_road} == {upper.id, lower.id}
            ),
            None,
        )
        if crossing is not None and "dz_m" in entry:
            ref.crossing_dz[crossing.id] = float(entry["dz_m"])
            if "lonlat" in entry:
                x, y = to_unit.lonlat_to_unit([entry["lonlat"][0]], [entry["lonlat"][1]])
                ref.crossing_points[crossing.id] = (float(x[0]), float(y[0]))
    for structure in (doc.get("structure_ends") or {}).get("structures", []):
        for end in structure.get("ends", []):
            road = _road_for_way(model, end["along"])
            st = next(
                (
                    s
                    for s in model.structures
                    if road is not None and s.carried_road == road.id
                ),
                None,
            )
            if st is None:
                continue
            stations = []
            for key in ("start_lonlat", "end_lonlat"):
                x, y = to_unit.lonlat_to_unit([end[key][0]], [end[key][1]])
                stations.append(road.alignment.project(float(x[0]), float(y[0]))[0])
            ref.structure_stations[st.id] = (min(stations), max(stations))
    return ref


def build_unit(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    crossings_path: str | Path | None = None,
    val3dity_exe: str | None = None,
    points_overlay: bool = True,
    profile: str = "analytic",
    lods: bool = True,
) -> tuple[RoadModel, GateReport, dict[str, Path]]:
    """Build, gate and publish one unit. Returns the model, the gate report and export paths.

    ``profile`` is the export profile of the solids ("analytic" or "realtime", section 7).
    """
    cfg = load_unit_config(config_path)
    verify_inputs(cfg)
    frame = cfg.frame()
    parts, records = [], []
    for i, src in enumerate(cfg.lidar):
        pts, rec = read_points(
            src.path,
            cfg.aoi_lonlat,
            frame,
            i,
            name=src.name,
            year=src.year,
            quality_level=src.quality_level,
        )
        parts.append(pts)
        records.append(rec)
    if len(parts) != 1:
        raise NotImplementedError(
            "several lidar collections per unit arrive with FR-IN-4"
        )
    points = parts[0]
    vectors = read_vectors(cfg.vectors, frame)
    unit = build_unit_record(cfg, records)
    pavement = pavement_points(points, vectors)
    holdout = holdout_for(points, pavement)
    model = run_pipeline(points, vectors, frame, unit, holdout=holdout).model

    measured: dict[str, Any] = {"edges": measure_edges(model, points, holdout=holdout)}
    crossings_ref = None
    if crossings_path is not None and Path(crossings_path).exists():
        ref = load_site_reference(crossings_path, model, frame)
        crossings_ref = ref.crossings
        measured["crossing_dz"] = ref.crossing_dz
        measured["crossing_points"] = ref.crossing_points
        measured["structure_stations"] = ref.structure_stations
    report, paths = export_model(
        model,
        out_dir,
        crossings_ref=crossings_ref,
        points=pavement,
        measured=measured,
        val3dity_exe=val3dity_exe,
        profile=profile,
        lods=lods,
    )
    if points_overlay:
        paths["points"] = write_points_glb(
            model,
            points.xyz,
            points.classification,
            paths["model"].parent / POINTS_OVERLAY,
        )
    return model, report, paths
