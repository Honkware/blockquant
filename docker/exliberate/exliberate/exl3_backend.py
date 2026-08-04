"""EXL3 inference backend for post-quant validation (SPEC ADDENDUM §C).

``EXL3Backend`` wraps ExLlamaV3's Config/Model/Cache/Tokenizer/Generator
with the minimal surface exliberate's scorers need — ``generate(prompts,
max_new_tokens) -> list[str]`` and ``get_logits(prompts) -> Tensor`` — so
``KeywordScorer`` and ``heretic_compat.FirstTokenKL`` run unchanged against a
quantized EXL3 model.

``PostQuantValidator`` compares a pre-quant reference (the ablated fp16
model, any scorer-compatible backend) against the EXL3 artifact and produces
``{refusal_pre, refusal_post, kl_post_vs_pre, rebound_pp, passed}``.

ExLlamaV3 is lazy-imported; without it or without CUDA every constructor
raises :class:`exliberate.quant.QuantUnavailable` — importing this module is
always safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

from .quant import QuantUnavailable
from .scorers.base import PromptLike

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import ModelBackend


class EXL3Backend:
    """Scorer-compatible wrapper around a loaded EXL3 model (SPEC §C).

    CUDA-only (ExLlamaV3's EXL3 GEMM runs through its CUDA extension).
    ``adapter_dir`` optionally points at a PEFT-format LoRA directory (e.g.
    one written by ``quant.attach_ablation_lora``) applied at runtime
    post-dequant.
    """

    def __init__(
        self,
        model_dir: str,
        max_cache_tokens: int = 8192,
        adapter_dir: str | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise QuantUnavailable(
                "ExLlamaV3 inference requires a CUDA GPU (none detected).\n\n"
                + "Install:  pip install exllamav3   (or: pip install exliberate[quant])"
            )
        try:
            from exllamav3 import Cache, Config, Generator, Model, Tokenizer
            from exllamav3.generator.sampler import ComboSampler
        except ImportError as exc:
            raise QuantUnavailable() from exc

        self._ComboSampler = ComboSampler
        self.config = Config.from_directory(str(model_dir))
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=max_cache_tokens)
        self.model.load()  # pragma: no cover - GPU-only
        self.tokenizer = Tokenizer.from_config(self.config)
        self.generator = Generator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer
        )
        self._lora = None
        if adapter_dir is not None:  # pragma: no cover - GPU-only
            from exllamav3.model.lora import LoRA

            self._lora = LoRA.from_directory(
                self.model, str(adapter_dir), lora_scaling=1.0
            )

    # ------------------------------------------------------------------ api
    def unload_adapter(self) -> None:  # pragma: no cover - GPU-only
        """Detach the runtime LoRA, if one was loaded."""
        if self._lora is not None:
            self._lora.unload()
            self._lora = None

    def _chat_text(self, prompt: PromptLike) -> str:
        """Chat-template a prompt via the ExLlamaV3 tokenizer.

        ExLlamaV3's Tokenizer wraps the HF one as ``.tokenizer``; if no chat
        template is available, fall back to a minimal user/assistant format.
        """
        messages: list[dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        # The pre-quant reference formats with the model's real chat template.
        # ExLlamaV3's Tokenizer exposes hf_render_chat_template; the raw
        # tokenizers.Tokenizer underneath exposes neither, and falling back to
        # a plain transcript here would feed the two models DIFFERENT prompts,
        # making both the KL and the refusal comparison meaningless.
        for obj in (self.tokenizer, getattr(self.tokenizer, "tokenizer", None)):
            if obj is None:
                continue
            for attr in ("apply_chat_template", "hf_render_chat_template"):
                fn = getattr(obj, attr, None)
                if fn is None:
                    continue
                try:
                    out = fn(messages, tokenize=False, add_generation_prompt=True)
                except TypeError:
                    try:
                        out = fn(messages, add_generation_prompt=True)
                    except Exception:
                        continue
                except Exception:
                    continue
                if isinstance(out, str) and out:
                    return out
        text = f"{prompt.system}\n\n" if prompt.system else ""
        return text + f"user: {prompt.user}\nassistant:"  # pragma: no cover

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[PromptLike],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> list[str]:
        """Greedy (temperature 0) or sampled continuations, prompt stripped."""
        outputs: list[str] = []
        eos = self.tokenizer.eos_token_id  # pragma: no cover - GPU-only
        for prompt in prompts:  # pragma: no cover - GPU-only
            sampler = (
                self._ComboSampler(temperature=0.0, top_k=1)
                if temperature == 0.0
                else self._ComboSampler(temperature=float(temperature))
            )
            outputs.append(
                self.generator.generate(
                    prompt=self._chat_text(prompt),
                    max_new_tokens=max_new_tokens,
                    sampler=sampler,
                    stop_conditions=[eos],
                )
            )
        return outputs

    @torch.no_grad()
    def get_logits(self, prompts: Sequence[PromptLike]) -> torch.Tensor:
        """First-generated-token logits: (n_prompts, vocab), FP32 CPU."""
        rows: list[torch.Tensor] = []
        for prompt in prompts:  # pragma: no cover - GPU-only
            ids = self.tokenizer.encode(self._chat_text(prompt))
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor([list(ids)], dtype=torch.long)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            logits = self.model.forward(ids.cuda(), params=None)  # (1, n, vocab)
            rows.append(logits[0, -1].to(torch.float32).cpu())
        if not rows:
            return torch.zeros(0, 0)
        return torch.stack(rows)


class PostQuantValidator:
    """Post-quant validation: EXL3 artifact vs pre-quant ablated fp16 (§C).

    Captures the pre-quant reference metrics at init (keyword refusal rate on
    ``harmful_prompts``; first-token logits on ``benign_prompts`` for the KL
    baseline) via the unchanged exliberate scorers. ``validate()`` scores the
    post-quant model and returns the SPEC §C report dict.
    """

    def __init__(
        self,
        pre_model: "ModelBackend | Any",
        post_model: "EXL3Backend | Any",
        harmful_prompts: Sequence[PromptLike],
        benign_prompts: Sequence[PromptLike],
        max_new_tokens: int = 64,
        rebound_threshold_pp: float = 5.0,
    ) -> None:
        from .scorers.heretic_compat import FirstTokenKL
        from .scorers.keyword import KeywordScorer

        self.post_model = post_model
        self.rebound_threshold_pp = float(rebound_threshold_pp)
        self._refusal_pre = KeywordScorer(harmful_prompts, max_new_tokens).score(
            pre_model
        ).value
        self._keyword_post = KeywordScorer(harmful_prompts, max_new_tokens)
        self._kl = FirstTokenKL(benign_prompts, base_model=pre_model)

    def validate_metrics(self, post_model: Any | None = None) -> dict[str, float]:
        """Core metrics only (used per-λ inside the sweep)."""
        post = post_model if post_model is not None else self.post_model
        return {
            "refusal_pre": float(self._refusal_pre),
            "refusal_post": float(self._keyword_post.score(post).value),
            "kl_post_vs_pre": float(self._kl.score(post).value),
        }

    def validate(self, post_model: Any | None = None) -> dict[str, Any]:
        """Full SPEC §C report: metrics + rebound + pass/fail."""
        metrics = self.validate_metrics(post_model)
        rebound = (metrics["refusal_post"] - metrics["refusal_pre"]) * 100.0
        return {
            **metrics,
            "rebound_pp": float(rebound),
            "passed": bool(rebound <= self.rebound_threshold_pp),
        }
