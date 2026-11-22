"""Design-convention rule library (section 8). Lane `rules`.

Public API: Rule, RuleBook.load(path=None), RuleBook.get/evaluate/penalty/fill,
class_chain, era_of, RuleBookError. See docs/RULES.md.
"""

from roader.rules.book import ERAS, Rule, RuleBook, RuleBookError, class_chain, era_of

__all__ = ["ERAS", "Rule", "RuleBook", "RuleBookError", "class_chain", "era_of"]
