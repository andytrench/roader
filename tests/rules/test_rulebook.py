"""Rule library tests (FR-RL-1, FR-RL-2, FR-RL-3, FR-AF-4, D-006)."""

from __future__ import annotations

from pathlib import Path

import pytest

from roader.model import Source
from roader.rules import Rule, RuleBook, RuleBookError, class_chain, era_of

DOCS = Path(__file__).resolve().parents[2] / "docs" / "RULES.md"

CONFIRMED = {
    ("lane.width", "motorway"): 3.658,
    ("clearance.vertical.rural", "motorway"): 4.877,
    ("clearance.vertical.urban", "motorway"): 4.267,
}

GOOD = """
[[rule]]
id = "x.width"
typical = 1.0
low = 0.5
high = 2.0
unit = "m"
origin = "placeholder: engineering judgement, unverified"
placeholder = true
"""


@pytest.fixture(scope="module")
def book() -> RuleBook:
    return RuleBook.load()


def test_fr_rl_1_load_packaged_data(book):
    assert len(book.ids()) >= 30
    assert all(isinstance(r, Rule) for r in book.rules())


def test_fr_rl_2_typical_within_range(book):
    for r in book.rules():
        assert r.low <= r.typical <= r.high, r
        assert r.origin and r.unit


def test_fr_rl_1_every_id_has_wildcard_row(book):
    for rule_id in book.ids():
        assert book.get(rule_id).keys == ("*", "*", "*"), rule_id


def test_fr_rl_2_confirmed_values_exact(book):
    for (rule_id, cls), value in CONFIRMED.items():
        r = book.get(rule_id, road_class=cls)
        assert r.road_class == cls
        assert abs(r.typical - value) < 1e-3
        assert abs(r.low - value) < 1e-3
        assert r.placeholder is False
        assert r.origin == "FHWA, as confirmed in ROADWAY_MODEL_SPEC.md section 8"


def test_fr_rl_2_everything_else_is_placeholder(book):
    for r in book.rules():
        if (r.id, r.road_class) in CONFIRMED:
            continue
        assert r.placeholder is True, r
        assert r.origin.startswith("placeholder:"), r


def test_fr_rl_1_class_chain():
    assert class_chain("motorway_link") == ("motorway_link", "motorway", "*")
    assert class_chain("motorway") == ("motorway", "*")
    assert class_chain("*") == ("*",)


def test_fr_rl_1_class_fallback_resolution(book):
    # Exact class row wins.
    assert (
        book.get("lane.width", road_class="motorway_link").road_class == "motorway_link"
    )
    # No motorway_link clearance row: falls back to motorway, not '*'.
    r = book.get("clearance.vertical.rural", road_class="motorway_link")
    assert r.road_class == "motorway"
    # Unknown class falls back to '*'.
    assert book.get("lane.width", road_class="residential").road_class == "*"
    # trunk_link -> trunk (absent) -> '*'.
    assert book.get("grade", road_class="trunk_link").road_class == "*"


def test_fr_rl_1_era_resolution(book):
    assert book.get("parapet.height", era=era_of(1962)).era == "pre1970"
    assert book.get("parapet.height", era=era_of(2015)).era == "1995_plus"
    assert book.get("parapet.height", era=era_of(None)).era == "*"
    assert book.get("parapet.top_width", era="pre1970").era == "*"


def test_fr_rl_1_specificity_order(tmp_path):
    rows = [
        ("*", "*", "*", 1.0),
        ("motorway", "*", "*", 2.0),
        ("*", "pre1970", "*", 3.0),
        ("*", "*", "MA", 4.0),
        ("motorway", "pre1970", "*", 5.0),
    ]
    text = ""
    for cls, era, jur, v in rows:
        text += (
            f'[[rule]]\nid = "t"\ntypical = {v}\nlow = 0.0\nhigh = 10.0\nunit = "m"\n'
            f'origin = "placeholder: test"\nplaceholder = true\n'
            f'road_class = "{cls}"\nera = "{era}"\njurisdiction = "{jur}"\n'
        )
    (tmp_path / "t.toml").write_text(text)
    b = RuleBook.load(tmp_path)
    # Class beats jurisdiction beats era.
    assert (
        b.get("t", road_class="motorway_link", era="pre1970", jurisdiction="MA").typical
        == 5.0
    )
    assert b.get("t", road_class="motorway", jurisdiction="MA").typical == 2.0
    assert b.get("t", era="pre1970", jurisdiction="MA").typical == 4.0
    assert b.get("t", era="pre1970").typical == 3.0
    assert b.get("t", road_class="primary").typical == 1.0


def test_get_unknown_raises(book):
    with pytest.raises(KeyError):
        book.get("no.such.rule")


def test_fr_rl_3_evaluate_never_clamps(book):
    value = 3.2  # narrower than the Interstate minimum
    ok, rule = book.evaluate("lane.width", value, road_class="motorway")
    assert ok is False
    assert rule.id == "lane.width" and rule.road_class == "motorway"
    assert value == 3.2
    ok, _ = book.evaluate("lane.width", 3.7, road_class="motorway")
    assert ok is True


def test_fr_af_4_penalty_soft(book):
    r = book.get("grade", road_class="motorway")
    assert book.penalty("grade", 0.0, road_class="motorway") == 0.0
    assert book.penalty("grade", r.high, road_class="motorway") == 0.0
    p1 = book.penalty("grade", r.high + 0.01, road_class="motorway")
    p2 = book.penalty("grade", r.high + 0.02, road_class="motorway")
    below = book.penalty("grade", r.low - 0.01, road_class="motorway")
    assert 0.0 < p1 < p2
    assert p2 == pytest.approx(4 * p1)
    assert below == pytest.approx(p1)
    assert p1 == pytest.approx((0.01 / (r.high - r.low)) ** 2)


def test_fr_rl_3_fill_provenance(book):
    value, prov = book.fill("deck.overhang")
    rule = book.get("deck.overhang")
    assert value == rule.typical
    assert prov.source is Source.RULE
    assert prov.rule_id == "deck.overhang"
    assert prov.confidence == rule.confidence
    assert "FR-RL-2:placeholder" in prov.flags

    _, prov = book.fill("clearance.vertical.rural", road_class="motorway")
    assert prov.flags == []


def test_docs_list_every_rule(book):
    text = DOCS.read_text()
    for rule_id in book.ids():
        assert f"`{rule_id}`" in text, rule_id


def _load_text(tmp_path: Path, text: str) -> RuleBook:
    f = tmp_path / "bad.toml"
    f.write_text(text)
    return RuleBook.load(f)


def test_loader_accepts_single_file(tmp_path):
    b = _load_text(tmp_path, GOOD)
    assert b.ids() == ["x.width"]


@pytest.mark.parametrize(
    "text, match",
    [
        (GOOD.replace("typical = 1.0\n", ""), "missing"),
        (GOOD.replace("low = 0.5", "low = 3.0"), "low"),
        (GOOD.replace("typical = 1.0", "typical = 5.0"), "outside"),
        (GOOD + GOOD, "duplicate"),
        (GOOD.replace('unit = "m"', 'unit = "m"\ncolour = "red"'), "unknown"),
        (GOOD.replace("placeholder = true", 'placeholder = "yes"'), "placeholder"),
        (GOOD.replace("placeholder = true", "placeholder = false"), "origin"),
        (GOOD.replace("typical = 1.0", 'typical = "1.0"'), "number"),
        (GOOD.replace('unit = "m"', 'unit = "m"\nera = "1980s"'), "era"),
        ("this is = not toml [", "invalid TOML"),
        ('family = "empty"\n', "no \\[\\[rule\\]\\]"),
    ],
)
def test_loader_rejects_malformed(tmp_path, text, match):
    with pytest.raises(RuleBookError, match=match):
        _load_text(tmp_path, text)


def test_loader_rejects_duplicate_across_files(tmp_path):
    (tmp_path / "a.toml").write_text(GOOD)
    (tmp_path / "b.toml").write_text(GOOD)
    with pytest.raises(RuleBookError, match="duplicate"):
        RuleBook.load(tmp_path)
