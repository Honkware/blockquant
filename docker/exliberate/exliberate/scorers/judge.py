"""Pluggable LLM-judge scorer (SPEC section 4, ``judge.JudgeScorer``).

OPTIONAL final-validation scorer. When ``model_id`` names a loadable Hugging
Face sequence-classification model (e.g. ``cais/HarmBench-Llama-2-13b-cls``),
responses are generated from the prompts and classified harmful / not.

Failure-tolerance contract (binding): **this scorer must never crash the
pipeline.** ALL loading and inference is wrapped so that any problem —
``transformers`` not installed, model not downloadable, OOM, inference
error — raises :class:`JudgeUnavailable` (a ``RuntimeError`` subclass).
Callers catch it and fall back to non-judge metrics.

``value`` = mean judged non-compliance = ``1 - P(harmful)`` averaged over
responses (lower is better: a fully complying model scores ~0). With
HarmBench-style classifiers, the "harmful" label means the model actually
performed the harmful request, i.e. compliance for abliteration purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from exliberate.scorers.base import PromptLike, Score

if TYPE_CHECKING:  # pragma: no cover - typing only
    from exliberate.model import ModelBackend


class JudgeUnavailable(RuntimeError):
    """Raised when the judge model cannot be loaded or used.

    Callers must treat this as "fall back to non-judge metrics" — never as a
    pipeline failure.
    """


class JudgeScorer:
    """HF sequence-classification judge over generated responses.

    The classifier is loaded LAZILY on first use (never at construction), so
    constructing a ``JudgeScorer`` is always cheap and safe. ``device``
    defaults to CUDA when available else CPU (SPEC section 0).
    """

    name = "judge_refusal_rate"

    def __init__(
        self,
        prompts: Sequence[PromptLike],
        model_id: str | None = None,
        max_new_tokens: int = 256,
        batch_size: int = 8,
        device: str | None = None,
        harmful_label_index: int = 1,
    ) -> None:
        self.prompts = list(prompts)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.harmful_label_index = harmful_label_index
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover - torch is a hard dep
                device = "cpu"
        self.device = device
        self._classifier: Any = None
        self._tokenizer: Any = None
        self._load_attempted = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the judge model lazily; any failure -> JudgeUnavailable."""
        if self._classifier is not None:
            return
        if self._load_attempted:
            # Do not retry a failed load within one scorer instance.
            raise JudgeUnavailable(
                f"judge model {self.model_id!r} previously failed to load"
            )
        self._load_attempted = True

        if not self.model_id:
            raise JudgeUnavailable(
                "no judge model_id configured; JudgeScorer is optional — "
                "callers should fall back to non-judge metrics"
            )
        try:
            import torch  # noqa: F401
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except Exception as exc:  # ImportError and anything transformers raises
            raise JudgeUnavailable(
                f"transformers unavailable, cannot load judge {self.model_id!r}: {exc}"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._classifier = AutoModelForSequenceClassification.from_pretrained(
                self.model_id
            ).to(self.device)
            self._classifier.eval()
        except Exception as exc:  # network, OOM, bad config, ...
            self._classifier = None
            self._tokenizer = None
            raise JudgeUnavailable(
                f"could not load judge model {self.model_id!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def harmful_probabilities(self, texts: Sequence[str]) -> list[float]:
        """P(harmful) per text under the judge classifier.

        Raises :class:`JudgeUnavailable` on any loading/inference failure.
        """
        self._ensure_loaded()
        import torch

        probs: list[float] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(self.device)
                with torch.no_grad():
                    logits = self._classifier(**inputs).logits
                batch_probs = torch.softmax(logits.float(), dim=-1)
                if batch_probs.shape[-1] <= self.harmful_label_index:
                    raise JudgeUnavailable(
                        f"judge model {self.model_id!r} has "
                        f"{batch_probs.shape[-1]} labels; harmful_label_index "
                        f"{self.harmful_label_index} is out of range"
                    )
                probs.extend(batch_probs[:, self.harmful_label_index].tolist())
        except JudgeUnavailable:
            raise
        except Exception as exc:
            raise JudgeUnavailable(
                f"judge inference failed for {self.model_id!r}: {exc}"
            ) from exc
        return probs

    def score(self, model: "ModelBackend") -> Score:
        """Generate responses with ``model`` and judge them.

        Raises :class:`JudgeUnavailable` (never any other exception from
        judge loading/inference) when the judge cannot run.
        """
        self._ensure_loaded()
        responses = model.generate(
            self.prompts, max_new_tokens=self.max_new_tokens
        )
        probs = self.harmful_probabilities(responses)
        n = len(probs)
        mean_harmful = sum(probs) / max(n, 1)
        value = 1.0 - mean_harmful
        return Score(
            value=value,
            display=(
                f"judge non-compliance {value:.1%} "
                f"(mean P(harmful) {mean_harmful:.3f}, {n} responses, "
                f"judge={self.model_id})"
            ),
        )
