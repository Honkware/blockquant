"""Capability probe scorer (SPEC section 4, ``capability.CapabilityProbe``).

Measures capability damage from abliteration via forward-only multiple-choice
loglikelihood probing (no generation): for every probe item, each choice is
teacher-forced after the context, the argmax-logprob choice is the model's
answer, and accuracy is the fraction matching the gold label.

``value = base_accuracy - current_accuracy`` (SPEC: "value = -delta accuracy,
lower better = less damage"). Sign convention:

* ``value > 0`` — the current model is *worse* than base (capability damage;
  the quantity being minimized during search),
* ``value = 0`` — no measurable damage,
* ``value < 0`` — the current model out-scores base on the probe (improvement
  on this metric, i.e. negative damage).

Base results are cached at init when ``base_model`` is provided; otherwise
captured lazily on the first :meth:`score` call or explicitly via
:meth:`capture_base`.

Probe file format (SPEC section 2): JSONL, one
``{"context": str, "choices": [str, ...], "label": int}`` per line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from exliberate.scorers.base import Score, make_prompt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from exliberate.model import ModelBackend


@dataclass(frozen=True)
class ProbeItem:
    """One multiple-choice capability probe item."""

    context: str
    choices: tuple[str, ...]
    label: int

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("probe item must have at least one choice")
        if not 0 <= self.label < len(self.choices):
            raise ValueError(
                f"label {self.label} out of range for {len(self.choices)} choices"
            )


def load_probe(path: str | Path) -> list[ProbeItem]:
    """Load a capability probe JSONL file (SPEC section 2 schema)."""
    items: list[ProbeItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                items.append(
                    ProbeItem(
                        context=str(record["context"]),
                        choices=tuple(str(c) for c in record["choices"]),
                        label=int(record["label"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: malformed probe item: {exc}") from exc
    if not items:
        raise ValueError(f"{path}: probe file is empty")
    return items


class CapabilityProbe:
    """Multiple-choice loglik capability scorer (delta vs base; SPEC 4).

    For each item, ``teacher_forced_logprobs`` is called once with the
    context repeated per choice; ``argmax`` is the predicted answer. Batching
    is item-wise (small probe files; keeps memory bounded on CPU).
    """

    name = "capability_delta"

    def __init__(
        self,
        probe_path: str | Path,
        base_model: "ModelBackend | None" = None,
    ) -> None:
        self.probe_path = Path(probe_path)
        self.items = load_probe(self.probe_path)
        self._base_accuracy: float | None = None
        if base_model is not None:
            self.capture_base(base_model)

    def capture_base(self, model: "ModelBackend") -> None:
        """Cache the base model's probe accuracy."""
        self._base_accuracy = self.accuracy(model)

    def accuracy(self, model: "ModelBackend") -> float:
        """Raw multiple-choice accuracy of ``model`` on the probe items."""
        correct = 0
        for item in self.items:
            prompts = [make_prompt(user=item.context)] * len(item.choices)
            logprobs = model.teacher_forced_logprobs(
                prompts, list(item.choices)
            )
            if not isinstance(logprobs, torch.Tensor):
                logprobs = torch.as_tensor(logprobs, dtype=torch.float32)
            pred = int(logprobs.argmax().item())
            if pred == item.label:
                correct += 1
        return correct / len(self.items)

    def score(self, model: "ModelBackend") -> Score:
        if self._base_accuracy is None:
            # Lazy capture: first scored model becomes the reference (delta 0).
            self.capture_base(model)
        current_accuracy = self.accuracy(model)
        assert self._base_accuracy is not None  # for type checkers
        value = self._base_accuracy - current_accuracy
        display = (
            f"capability delta {value:+.3f} "
            f"(base {self._base_accuracy:.3f} -> current {current_accuracy:.3f}, "
            f"{len(self.items)} items)"
        )
        return Score(value=value, display=display)
