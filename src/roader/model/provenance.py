"""L8 provenance. Every model element carries one record (D-010, AC-H6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Source(str, Enum):
    MEASURED = "measured"
    INVENTORY = "inventory"
    VECTOR = "vector"
    RULE = "rule"


@dataclass
class Provenance:
    source: Source
    confidence: float
    rms_m: float | None = None
    p95_m: float | None = None
    n_points: int = 0
    density_pts_m2: float | None = None
    observed_fraction: float | None = None
    rule_id: str | None = None
    # Requirement IDs or short codes, e.g. "FR-AF-4:override" when data overrode a rule.
    flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source = Source(self.source)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


def measured(
    confidence: float,
    *,
    rms_m: float | None = None,
    p95_m: float | None = None,
    n_points: int = 0,
    observed_fraction: float | None = None,
) -> Provenance:
    return Provenance(
        Source.MEASURED,
        confidence,
        rms_m=rms_m,
        p95_m=p95_m,
        n_points=n_points,
        observed_fraction=observed_fraction,
    )


def by_rule(rule_id: str, confidence: float = 0.5) -> Provenance:
    return Provenance(Source.RULE, confidence, rule_id=rule_id)


def from_vector(confidence: float = 0.5) -> Provenance:
    return Provenance(Source.VECTOR, confidence)
