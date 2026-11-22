"""Soft rule priors for fitting (FR-AF-4).

``RuleBook.penalty`` scales the excess by the width of the permitted range. For
rules whose range is a one-sided bound with a far plausibility limit (minimum
radius 500 to 100000 m, minimum K 50 to 10000) that makes the penalty vanish, so
fitting scales the excess by the bound itself instead. The rule values are used
as they are; only the normalization differs.
"""

from __future__ import annotations

from roader.rules import RuleBook


def bound_penalty(rules: RuleBook, rule_id: str, value: float, road_class: str = "*") -> float:
    """0 inside [low, high]; ((excess) / |bound|)^2 outside, bound being the violated limit."""
    rule = rules.get(rule_id, road_class)
    if value < rule.low:
        return ((rule.low - value) / max(abs(rule.low), 1e-9)) ** 2
    if value > rule.high:
        return ((value - rule.high) / max(abs(rule.high), 1e-9)) ** 2
    return 0.0


def curvature_penalty(rules: RuleBook, kappa: float, road_class: str = "*") -> float:
    """Minimum-radius prior expressed in curvature: ((|k| - 1/R_min) * R_min)^2 beyond the limit."""
    rule = rules.get("curve.radius", road_class)
    k_max = 1.0 / rule.low
    excess = abs(kappa) - k_max
    return (excess / k_max) ** 2 if excess > 0 else 0.0
