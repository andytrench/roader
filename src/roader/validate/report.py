"""Gate result contract (stage 12). Owned by lane `validate`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Violation:
    req: str  # e.g. "FR-SV-1"
    element_id: str
    message: str


@dataclass
class GateResult:
    gate: str  # e.g. "AC-H1", "AC-A1"
    passed: bool
    hard: bool
    metrics: dict[str, float] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    skipped_reason: str = ""  # set when the gate could not run; a skipped hard gate fails


@dataclass
class GateReport:
    unit_id: str
    results: list[GateResult] = field(default_factory=list)

    @property
    def publishable(self) -> bool:
        """FR-EX-1: publish only when every hard gate passed."""
        return all(result.passed for result in self.results if result.hard)
