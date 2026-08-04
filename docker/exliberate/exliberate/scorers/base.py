"""Base types for all exliberate scorers.

Implements SPEC section 3 (scorers/base.py) exactly:

    @dataclass
    class Score:
        value: float    # scalar, LOWER-is-better convention for ALL scorers
        display: str

    class Scorer(Protocol):
        name: str
        def score(self, model: ModelBackend) -> Score: ...

This module deliberately avoids importing ``exliberate.model`` or
``exliberate.prompts`` at runtime: scorers only rely on the structural
interfaces described in the SPEC, so they can be exercised against a
``MockBackend`` and against any object with ``user``/``system`` attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime dependency
    from exliberate.model import ModelBackend  # noqa: F401


@dataclass
class Score:
    """Scalar score with a human-readable rendering.

    Convention (SPEC, binding): ``value`` is LOWER-is-better for every
    scorer, regardless of what the underlying metric measures.
    """

    value: float
    display: str


@runtime_checkable
class Scorer(Protocol):
    """Protocol every scorer implements (SPEC section 3)."""

    name: str

    def score(self, model: "ModelBackend") -> Score:
        """Score ``model`` and return a lower-is-better :class:`Score`."""
        ...


class PromptLike(Protocol):
    """Structural twin of ``exliberate.prompts.Prompt``.

    SPEC's ``Prompt`` is a frozen dataclass with ``user`` and an optional
    ``system``; any object with those attributes is accepted by the scorers.
    Keeping this structural avoids a hard runtime dependency on core files.
    """

    user: str
    system: str | None


def prompt_texts(prompts: Sequence[PromptLike]) -> list[str]:
    """Extract the user text of each prompt (utility for display/logging)."""
    return [p.user for p in prompts]


@dataclass(frozen=True)
class SimplePrompt:
    """Fallback prompt object mirroring ``exliberate.prompts.Prompt``.

    Used only when the core package is not importable (e.g. scorer unit
    tests against a MockBackend); see :func:`make_prompt`.
    """

    user: str
    system: str | None = None


def make_prompt(user: str, system: str | None = None) -> PromptLike:
    """Build a prompt, preferring the real ``exliberate.prompts.Prompt``.

    Soft import: if the core package is available the canonical dataclass is
    returned (so backends that type-check see the real type); otherwise a
    structurally identical :class:`SimplePrompt` is returned.
    """
    try:
        from exliberate.prompts import Prompt

        return Prompt(user=user, system=system)
    except ImportError:  # pragma: no cover - exercised in scorer-only envs
        return SimplePrompt(user=user, system=system)
