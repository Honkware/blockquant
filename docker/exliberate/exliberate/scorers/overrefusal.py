"""Over-refusal scorer (SPEC section 4, ``overrefusal.OverRefusalScorer``).

Runs the improved keyword classifier over an XSTest-style
seemingly-harmful-but-benign prompt file (``overrefusal.jsonl``; path
injected). These prompts SHOULD be answered, so every refusal credit is a
false refusal. Report-only metric (not part of the search objectives).

``value`` = mean refusal credit on the over-refusal set (lower is better).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from exliberate.scorers.base import PromptLike, Score, make_prompt
from exliberate.scorers.keyword import KeywordScorer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from exliberate.model import ModelBackend


def load_text_prompts(path: str | Path, limit: int | None = None) -> list[PromptLike]:
    """Load a ``{"text": ...}`` JSONL prompt file (SPEC section 2 schema).

    Local minimal loader so the scorers have no hard dependency on core
    ``prompts.py``; produces real ``Prompt`` objects when the core package is
    importable (see :func:`exliberate.scorers.base.make_prompt`).
    """
    prompts: list[PromptLike] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                text = str(record["text"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"{path}:{lineno}: malformed prompt record: {exc}"
                ) from exc
            prompts.append(make_prompt(user=text))
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"{path}: prompt file is empty")
    return prompts


class OverRefusalScorer:
    """Keyword-based false-refusal rate on an over-refusal prompt file.

    Accepts either a path to a ``{"text": ...}`` JSONL file or a pre-built
    prompt sequence. Delegates classification to :class:`KeywordScorer`
    (lecture partial credit applies).
    """

    name = "overrefusal_rate"

    def __init__(
        self,
        prompts_or_path: str | Path | Sequence[PromptLike],
        max_new_tokens: int = 64,
        limit: int | None = None,
    ) -> None:
        if isinstance(prompts_or_path, (str, Path)):
            self.prompts = load_text_prompts(prompts_or_path, limit=limit)
        else:
            self.prompts = list(prompts_or_path)
            if limit is not None:
                self.prompts = self.prompts[:limit]
        self.max_new_tokens = max_new_tokens
        self._keyword = KeywordScorer(self.prompts, max_new_tokens=max_new_tokens)

    def score(self, model: "ModelBackend") -> Score:
        keyword_score = self._keyword.score(model)
        return Score(
            value=keyword_score.value,
            display=f"over-refusal rate {keyword_score.value:.1%} "
            f"({len(self.prompts)} benign-but-sensitive prompts)",
        )
