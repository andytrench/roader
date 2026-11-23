"""Element-level solid failures (FR-EX-1, D-119).

A stage that builds one element's bodies raises ``ElementFailure`` naming that element,
so ``generate(..., failures=[...])`` can replace or drop just that element and keep the
unit going. The original exception type and message stay in the flag text.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class ElementFailure(ValueError):
    """Building the bodies of one model element failed."""

    def __init__(self, element_id: str, cause: BaseException) -> None:
        super().__init__(f"{element_id}: {type(cause).__name__}: {cause}")
        self.element_id = element_id
        self.cause = cause


def describe(exc: BaseException) -> str:
    cause = exc.cause if isinstance(exc, ElementFailure) else exc
    return f"{type(cause).__name__}: {cause}"


@contextmanager
def element(element_id: str) -> Iterator[None]:
    """Re-raise any exception inside as an ElementFailure for ``element_id``."""
    try:
        yield
    except ElementFailure:
        raise
    except Exception as exc:
        raise ElementFailure(element_id, exc) from exc
