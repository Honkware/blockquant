"""Multi-token teacher-forced KL scorer (SPEC section 4, ``kl.MultiTokenKL``).

Rationale: heretic measures distribution drift only at the *first* generated
token, which misses drift accumulated over the rest of the completion. The
SPEC's ``ModelBackend.teacher_forced_logprobs`` returns the *summed* logprob
of each continuation, so a full per-token KL is not directly available.
This scorer therefore implements the standard KL-consistent logprob-delta
approximation:

    For continuation c (drawn from / typical of the base model on prompt p),
    reverse-KL on the continuation is

        KL(p_base || p_cur) over the continuation
            = E_base[ log p_base(c|p) - log p_cur(c|p) ]
           ~= -( log p_cur(c|p) - log p_base(c|p) )        (teacher-forced)

    i.e. ``kl_i = -(mean_logprob_current_i - mean_logprob_base_i)`` and the
    score is the mean over prompt/continuation pairs.

This is **reverse-KL-on-continuation** — the practical multi-token drift
metric: it is exactly 0 against the base model, grows as the current model's
teacher-forced probability of base-typical continuations degrades, and is
computed from the SPEC's summed-logprob interface alone. Because the backend
returns summed (not per-token-mean) logprobs, the per-pair deltas are
length-weighted; this is documented and acceptable for relative comparisons
on a fixed prompt/continuation set.

Also provides :func:`first_token_kl`, a reusable helper built on
``get_logits`` (full-vocabulary KL of the first generated token), used by
``heretic_compat.FirstTokenKL`` semantics and available to the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from exliberate.scorers.base import PromptLike, Score
from exliberate.scorers.heretic_compat import first_token_kl_distributions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from exliberate.model import ModelBackend


def first_token_kl(
    base_logits: torch.Tensor,
    current_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(base || current) of first-token full-vocab distributions.

    Thin reusable wrapper over the heretic-exact computation
    (``log_target=True``, ``batchmean`` reduction).
    """
    return first_token_kl_distributions(base_logits, current_logits)


class MultiTokenKL:
    """Multi-token teacher-forced KL against cached base logprobs (SPEC 4).

    Base-model teacher-forced logprobs on ``(prompts, continuations)`` are
    captured at init when ``base_model`` is given; otherwise they are
    captured lazily on the first :meth:`score` call (that model then defines
    the reference and scores 0), or explicitly via :meth:`capture_base`.

    ``value`` = mean over pairs of ``-(logprob_current - logprob_base)``
    (reverse-KL-on-continuation approximation, see module docstring). Lower
    is better; 0.0 means no drift from base on these continuations. The value
    is not clamped: a negative value means the current model assigns *higher*
    probability to the continuations than base did.
    """

    name = "multi_token_kl"

    def __init__(
        self,
        prompts: Sequence[PromptLike],
        continuations: Sequence[str],
        max_tokens: int = 64,
        base_model: "ModelBackend | None" = None,
    ) -> None:
        if len(prompts) != len(continuations):
            raise ValueError(
                f"prompts/continuations length mismatch: "
                f"{len(prompts)} != {len(continuations)}"
            )
        self.prompts = list(prompts)
        self.continuations = list(continuations)
        # Kept for interface parity with SPEC (kl.MultiTokenKL(...,
        # max_tokens=64)); teacher-forced scoring does not generate, so the
        # continuation length is the only truncation bound.
        self.max_tokens = max_tokens
        self._base_logprobs: torch.Tensor | None = None
        if base_model is not None:
            self.capture_base(base_model)

    def capture_base(self, model: "ModelBackend") -> None:
        """Capture and cache base-model summed logprobs of the continuations."""
        self._base_logprobs = (
            model.teacher_forced_logprobs(self.prompts, self.continuations)
            .detach()
            .cpu()
            .float()
        )

    def score(self, model: "ModelBackend") -> Score:
        if self._base_logprobs is None:
            # Lazy capture: first scored model becomes the reference (KL 0).
            self.capture_base(model)
        current = (
            model.teacher_forced_logprobs(self.prompts, self.continuations)
            .detach()
            .cpu()
            .float()
        )
        if current.shape != self._base_logprobs.shape:
            raise RuntimeError(
                f"teacher_forced_logprobs shape mismatch: {current.shape} vs "
                f"base {self._base_logprobs.shape}"
            )
        # Reverse-KL-on-continuation approximation per pair.
        per_pair = -(current - self._base_logprobs)
        value = float(per_pair.mean().item())
        return Score(
            value=value,
            display=(
                f"multi-token KL ~{value:.4f} nats "
                f"({len(per_pair)} continuation pairs)"
            ),
        )
