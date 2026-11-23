"""`roader build`: pinned inputs to a gated, published unit (FR-EX-1, AC-H2, AC-A4, AC-A5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from roader.build import POINTS_OVERLAY, load_site_reference
from roader.cli import main
from roader.io import load_unit_config
from roader.io.frame import FrameTransformer
from roader.model.synthetic import overpass_model

ROOT = Path(__file__).resolve().parents[1]
M1_UNIT = ROOT / "configs" / "units" / "wboylston_laurel_i190.toml"


def test_site_reference_maps_ways_to_model_ids(tmp_path):
    model = overpass_model()
    over = next(r for r in model.roads if r.vector_refs[0].feature_id == "over")
    st = model.structures[0]
    to_lonlat = FrameTransformer(model.frame)
    x0, y0, _h, _k = over.alignment.evaluate(st.s_start)
    x1, y1, _h, _k = over.alignment.evaluate(st.s_end)
    lon, lat = (a.tolist() for a in to_lonlat.unit_to_lonlat([x0, x1], [y0, y1]))
    doc = {
        "status": "draft, needs human review",
        "crossings": [
            {
                "upper": {"osm_way": "over"},
                "lower": {"osm_way": "eb"},
                "lonlat": [lon[0], lat[0]],
                "dz_m": 7.0,
                "structure": "s",
            },
            {
                "upper": {"osm_way": "over"},
                "lower": {"osm_way": "missing"},
                "dz_m": 7.0,
            },
        ],
        "structure_ends": {
            "status": "draft",
            "structures": [
                {
                    "id": "s",
                    "ends": [
                        {
                            "along": "over",
                            "start_lonlat": [lon[0], lat[0]],
                            "end_lonlat": [lon[1], lat[1]],
                            "length_m": 1.0,
                        }
                    ],
                }
            ],
        },
    }
    path = tmp_path / "crossings.yaml"
    path.write_text(yaml.safe_dump(doc))
    ref = load_site_reference(path, model, model.frame)
    eb = next(r for r in model.roads if r.vector_refs[0].feature_id == "eb")
    assert ref.crossings == [{"upper_road": over.id, "lower_road": eb.id}]
    assert list(ref.crossing_dz.values()) == [7.0]
    (px, py), = ref.crossing_points.values()  # D-096
    assert (px, py) == pytest.approx((x0, y0), abs=1e-3)
    s0, s1 = ref.structure_stations[st.id]
    assert s0 == pytest.approx(st.s_start, abs=1e-3) and s1 == pytest.approx(
        st.s_end, abs=1e-3
    )


_cfg = load_unit_config(M1_UNIT)


@pytest.mark.skipif(
    not (_cfg.lidar[0].path.exists() and _cfg.vectors["osm"].path.exists()),
    reason="fixture not fetched; run scripts/fetch_fixtures.py --site wboylston_laurel_i190",
)
def test_cli_build_m1_site(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    result = CliRunner().invoke(
        main, ["build", "--unit", str(M1_UNIT), "--out", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    for gate in (
        "AC-H2",
        "AC-H3",
        "AC-H4",
        "AC-H5",
        "AC-H6",
        "AC-A1",
        "AC-A3",
        "AC-A4",
        "AC-A5",
    ):
        assert f"PASS {gate}" in result.output, result.output
    state = "published" if "Published:" in result.output else "held"
    unit = tmp_path / state / "wboylston_laurel_i190"
    assert (unit / POINTS_OVERLAY).stat().st_size > 0
    report = json.loads((unit / "report.json").read_text())
    assert report["unit_id"] == "wboylston_laurel_i190"
