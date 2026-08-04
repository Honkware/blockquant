"""exliberate scorers (SPEC section 4).

All scorers implement the :class:`~exliberate.scorers.base.Scorer` protocol
and return a lower-is-better :class:`~exliberate.scorers.base.Score`.
"""

from exliberate.scorers.base import (
    PromptLike,
    Score,
    Scorer,
    SimplePrompt,
    make_prompt,
    prompt_texts,
)
from exliberate.scorers.capability import CapabilityProbe, ProbeItem, load_probe
from exliberate.scorers.heretic_compat import (
    HERETIC_REFUSAL_MARKERS,
    FirstTokenKL,
    HereticKeywordRate,
    heretic_is_refusal,
    normalize_text,
)
from exliberate.scorers.judge import JudgeScorer, JudgeUnavailable
from exliberate.scorers.keyword import (
    KeywordScorer,
    ResponseVerdict,
    classify_response,
    refusal_credit,
)
from exliberate.scorers.kl import MultiTokenKL, first_token_kl
from exliberate.scorers.overrefusal import OverRefusalScorer, load_text_prompts

__all__ = [
    "CapabilityProbe",
    "FirstTokenKL",
    "HERETIC_REFUSAL_MARKERS",
    "HereticKeywordRate",
    "JudgeScorer",
    "JudgeUnavailable",
    "KeywordScorer",
    "MultiTokenKL",
    "OverRefusalScorer",
    "ProbeItem",
    "PromptLike",
    "ResponseVerdict",
    "Score",
    "Scorer",
    "SimplePrompt",
    "classify_response",
    "first_token_kl",
    "heretic_is_refusal",
    "load_probe",
    "load_text_prompts",
    "make_prompt",
    "normalize_text",
    "prompt_texts",
    "refusal_credit",
]
