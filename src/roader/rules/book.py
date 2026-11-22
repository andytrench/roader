"""Rule library contract and loader (FR-RL-1 to 3, D-006). Owned by lane `rules`.

Rules live in TOML files under ``roader/rules/data/``, one file per rule family
(spec section 8). Each file holds ``[[rule]]`` tables. See ``docs/RULES.md``.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from roader.model.provenance import Provenance, Source

DATA_DIR = Path(__file__).parent / "data"

WILDCARD = "*"
PLACEHOLDER_PREFIX = "placeholder:"
# Fill confidence when a rule does not state one: rules are priors, not observations.
DEFAULT_CONFIDENCE_PLACEHOLDER = 0.3
DEFAULT_CONFIDENCE_CONFIRMED = 0.5

_REQUIRED = ("id", "typical", "low", "high", "unit", "origin", "placeholder")
_OPTIONAL = ("road_class", "era", "jurisdiction", "confidence", "note")

# Construction-era keys used by the data files. ``era_of`` maps a year built to one.
ERAS = ("pre1970", "1970_1994", "1995_plus")


class RuleBookError(ValueError):
    """A rule data file is malformed or inconsistent."""


@dataclass(frozen=True)
class Rule:
    """One design convention. ``low``/``high`` bound the permitted range (FR-RL-2).

    A measured value outside the range is kept and flagged, never clamped (FR-RL-3).
    """

    id: str  # e.g. "lane.width"
    typical: float
    low: float
    high: float
    unit: str  # "m" | "ratio" | "h_per_v" | "m_per_pct" | ...
    origin: str  # publication or "placeholder: indicative scale from spec section 8"
    placeholder: bool
    road_class: str = WILDCARD
    era: str = WILDCARD
    jurisdiction: str = WILDCARD
    confidence: float = DEFAULT_CONFIDENCE_PLACEHOLDER  # used by fill()
    note: str = ""

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    @property
    def keys(self) -> tuple[str, str, str]:
        return (self.road_class, self.era, self.jurisdiction)


def class_chain(road_class: str) -> tuple[str, ...]:
    """Classes tried for ``road_class``, most specific first.

    ``motorway_link`` -> ``motorway`` -> ``*``. A ``*`` query matches only ``*`` rows.
    """
    if road_class == WILDCARD:
        return (WILDCARD,)
    chain = [road_class]
    if road_class.endswith("_link") and len(road_class) > len("_link"):
        chain.append(road_class[: -len("_link")])
    chain.append(WILDCARD)
    return tuple(chain)


def era_of(year_built: int | None) -> str:
    """Era key for a construction year; ``*`` when unknown."""
    if year_built is None:
        return WILDCARD
    if year_built < 1970:
        return "pre1970"
    if year_built < 1995:
        return "1970_1994"
    return "1995_plus"


def _level(query: str, value: str) -> int | None:
    if value == query:
        return 0
    if value == WILDCARD:
        return 1
    return None


class RuleBook:
    """Rules indexed by id, resolved by most-specific match on class, jurisdiction, era."""

    def __init__(self, rules: list[Rule]):
        by_id: dict[str, list[Rule]] = {}
        seen: set[tuple[str, str, str, str]] = set()
        for rule in rules:
            key = (rule.id, *rule.keys)
            if key in seen:
                raise RuleBookError(f"duplicate rule {rule.id} for keys {rule.keys}")
            seen.add(key)
            by_id.setdefault(rule.id, []).append(rule)
        self._by_id = {
            k: sorted(v, key=lambda r: r.keys) for k, v in sorted(by_id.items())
        }

    @classmethod
    def load(cls, path: Path | None = None) -> RuleBook:
        """Load every ``*.toml`` in ``path`` (a directory or one file); default: packaged data."""
        path = DATA_DIR if path is None else Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.toml"))
        elif path.is_file():
            files = [path]
        else:
            raise RuleBookError(f"rule data path not found: {path}")
        if not files:
            raise RuleBookError(f"no rule files in {path}")
        rules: list[Rule] = []
        for f in files:
            rules.extend(_read_file(f))
        try:
            return cls(rules)
        except RuleBookError as exc:
            raise RuleBookError(f"{path}: {exc}") from None

    def ids(self) -> list[str]:
        return list(self._by_id)

    def rules(self) -> list[Rule]:
        """Every row, sorted by id then keys."""
        return [r for rows in self._by_id.values() for r in rows]

    def get(
        self,
        rule_id: str,
        road_class: str = WILDCARD,
        era: str = WILDCARD,
        jurisdiction: str = WILDCARD,
    ) -> Rule:
        """Most specific row for the keys (FR-RL-1).

        A row matches when each of its keys equals the query or is ``*`` (class also
        matches its family, see ``class_chain``). Among matches the winner has the most
        specific class, then jurisdiction, then era. Rows with identical keys are
        rejected at load, so the winner is unique.
        """
        rows = self._by_id.get(rule_id)
        if rows is None:
            raise KeyError(f"unknown rule id {rule_id!r}")
        chain = class_chain(road_class)
        best: tuple[tuple[int, int, int], Rule] | None = None
        for rule in rows:
            if rule.road_class not in chain:
                continue
            j = _level(jurisdiction, rule.jurisdiction)
            e = _level(era, rule.era)
            if j is None or e is None:
                continue
            score = (chain.index(rule.road_class), j, e)
            if best is None or score < best[0]:
                best = (score, rule)
        if best is None:
            raise KeyError(
                f"no {rule_id!r} rule for road_class={road_class!r} era={era!r} "
                f"jurisdiction={jurisdiction!r}"
            )
        return best[1]

    def evaluate(
        self,
        rule_id: str,
        value: float,
        *,
        road_class: str = WILDCARD,
        era: str = WILDCARD,
        jurisdiction: str = WILDCARD,
    ) -> tuple[bool, Rule]:
        """Whether ``value`` lies in the rule's range. The value is never altered (FR-RL-3)."""
        rule = self.get(rule_id, road_class, era, jurisdiction)
        return rule.contains(value), rule

    def penalty(
        self,
        rule_id: str,
        value: float,
        *,
        road_class: str = WILDCARD,
        era: str = WILDCARD,
        jurisdiction: str = WILDCARD,
    ) -> float:
        """Soft prior for fitting (FR-AF-4): 0 inside the range, ((distance / width))^2 outside.

        ``width`` is ``high - low``; for a zero-width range it is ``max(|typical|, 1)``.
        """
        rule = self.get(rule_id, road_class, era, jurisdiction)
        excess = max(rule.low - value, value - rule.high, 0.0)
        if excess == 0.0:
            return 0.0
        width = rule.high - rule.low
        if width <= 0.0:
            width = max(abs(rule.typical), 1.0)
        return (excess / width) ** 2

    def fill(
        self,
        rule_id: str,
        *,
        road_class: str = WILDCARD,
        era: str = WILDCARD,
        jurisdiction: str = WILDCARD,
    ) -> tuple[float, Provenance]:
        """Typical value and RULE provenance for an unobserved parameter (stages 8 and 9)."""
        rule = self.get(rule_id, road_class, era, jurisdiction)
        flags = ["FR-RL-2:placeholder"] if rule.placeholder else []
        prov = Provenance(Source.RULE, rule.confidence, rule_id=rule.id, flags=flags)
        return rule.typical, prov


def _read_file(path: Path) -> list[Rule]:
    try:
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise RuleBookError(f"{path.name}: invalid TOML: {exc}") from None
    extra = set(doc) - {"family", "rule"}
    if extra:
        raise RuleBookError(f"{path.name}: unknown top-level keys {sorted(extra)}")
    entries = doc.get("rule")
    if not isinstance(entries, list) or not entries:
        raise RuleBookError(f"{path.name}: no [[rule]] entries")
    return [
        _parse_rule(entry, f"{path.name} rule {i}") for i, entry in enumerate(entries)
    ]


def _number(entry: dict, name: str, where: str) -> float:
    v = entry[name]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RuleBookError(f"{where}: {name} must be a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise RuleBookError(f"{where}: {name} must be finite")
    return v


def _string(entry: dict, name: str, where: str) -> str:
    v = entry[name]
    if not isinstance(v, str) or not v.strip():
        raise RuleBookError(f"{where}: {name} must be a non-empty string")
    return v


def _parse_rule(entry: object, where: str) -> Rule:
    if not isinstance(entry, dict):
        raise RuleBookError(f"{where}: not a table")
    missing = [k for k in _REQUIRED if k not in entry]
    if missing:
        raise RuleBookError(f"{where}: missing fields {missing}")
    unknown = sorted(set(entry) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise RuleBookError(f"{where}: unknown fields {unknown}")
    where = f"{where} ({entry['id']})"

    rule_id = _string(entry, "id", where)
    typical = _number(entry, "typical", where)
    low = _number(entry, "low", where)
    high = _number(entry, "high", where)
    if low > high:
        raise RuleBookError(f"{where}: low {low} > high {high}")
    if not low <= typical <= high:
        raise RuleBookError(f"{where}: typical {typical} outside [{low}, {high}]")

    placeholder = entry["placeholder"]
    if not isinstance(placeholder, bool):
        raise RuleBookError(f"{where}: placeholder must be true or false")
    origin = _string(entry, "origin", where)
    if placeholder != origin.startswith(PLACEHOLDER_PREFIX):
        raise RuleBookError(
            f"{where}: origin must start with {PLACEHOLDER_PREFIX!r} exactly when placeholder"
        )

    keys = {}
    for name in ("road_class", "era", "jurisdiction"):
        keys[name] = _string(entry, name, where) if name in entry else WILDCARD
    if keys["era"] not in (WILDCARD, *ERAS):
        raise RuleBookError(f"{where}: era {keys['era']!r} not in {ERAS}")

    if "confidence" in entry:
        confidence = _number(entry, "confidence", where)
        if not 0.0 <= confidence <= 1.0:
            raise RuleBookError(f"{where}: confidence {confidence} outside [0, 1]")
    else:
        confidence = (
            DEFAULT_CONFIDENCE_PLACEHOLDER
            if placeholder
            else DEFAULT_CONFIDENCE_CONFIRMED
        )
    note = _string(entry, "note", where) if "note" in entry else ""

    return Rule(
        id=rule_id,
        typical=typical,
        low=low,
        high=high,
        unit=_string(entry, "unit", where),
        origin=origin,
        placeholder=placeholder,
        confidence=confidence,
        note=note,
        **keys,
    )
