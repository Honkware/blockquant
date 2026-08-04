"""ModelBackend: HF loading, LoRA adapter infra, residual/logit capture.

Implements SPEC §3 exactly. Device-agnostic (CPU dev sandbox, CUDA on user
hardware). Dense decoder-only models only (v1): Llama / Qwen2 / Qwen3 / Gemma /
SmolLM2-style layouts are discovered by suffix matching (``o_proj`` /
``out_proj`` for attention output, ``down_proj`` for the MLP down projection).

Residual capture follows the refusal-direction literature (Arditi et al. 2024,
arXiv:2406.11717): last-prompt-position hidden states per layer, including the
embedding output (``num_layers + 1`` entries), winsorized per layer/feature as
in heretic (github.com/p-e-w/heretic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .config import Settings
from .prompts import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Pinned CPU copy of the base weights is skipped above this many bytes
# (8 GiB). Dev-sandbox models are tiny; large models on user hardware simply
# lose `restore_base_weights` (exact export requires it) rather than OOMing.
_PIN_BUDGET_BYTES = 8 * 1024**3

# Suffixes recognised as abliterable components. "out_proj" covers
# architectures that name the attention output projection that way.
_ATTN_SUFFIXES = ("o_proj", "out_proj")
_MLP_SUFFIXES = ("down_proj",)


class ModelBackend:
    """SPEC §3 backend around a HF causal LM with optional PEFT LoRA adapters."""

    def __init__(self, settings: Settings) -> None:
        # Lazy heavy imports keep `import exliberate` cheap.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings = settings
        torch.manual_seed(settings.seed)

        self.device = settings.resolve_device()
        self.dtype = settings.resolve_dtype()

        trust = bool(getattr(settings, "trust_remote_code", False))
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model_id, trust_remote_code=trust
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            settings.model_id, dtype=self.dtype, trust_remote_code=trust
        )
        self._hf_model.to(self.device)
        self._hf_model.eval()
        if self._hf_model.generation_config is not None:
            self._hf_model.generation_config.pad_token_id = (
                self.tokenizer.pad_token_id
            )

        # PEFT wrapper, created by setup_adapters(). None until then.
        self._peft_model = None
        self._adapter_name = "default"

        # Pinned CPU copy of base weights for restore_base_weights().
        total_bytes = sum(
            p.numel() * p.element_size() for p in self._hf_model.parameters()
        )
        if total_bytes <= _PIN_BUDGET_BYTES:
            self._pinned: dict[str, torch.Tensor] | None = {
                k: v.detach().to("cpu", torch.float32).clone()
                for k, v in self._hf_model.state_dict().items()
            }
        else:  # pragma: no cover - only on large user models
            self._pinned = None

    # ------------------------------------------------------------------ spec
    @property
    def num_layers(self) -> int:
        return int(self._hf_model.config.num_hidden_layers)

    @property
    def _model(self) -> nn.Module:
        """The module to run forwards on (PEFT wrapper if present)."""
        return self._peft_model if self._peft_model is not None else self._hf_model

    def _layers(self) -> nn.ModuleList:
        """Locate the decoder layer stack across common HF layouts."""
        candidates = [
            ("model", "layers"),  # Llama/Qwen2/Qwen3/Gemma/SmolLM2
            ("transformer", "h"),  # GPT-2 style
            ("gpt_neox", "layers"),  # GPT-NeoX/Pythia
            ("model", "decoder", "layers"),  # OPT style
        ]
        for path in candidates:
            obj = self._hf_model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, nn.ModuleList):
                return obj
        # Fallback: first ModuleList matching num_hidden_layers.
        for module in self._hf_model.modules():
            if isinstance(module, nn.ModuleList) and len(module) == self.num_layers:
                return module
        raise ValueError(
            f"Could not locate decoder layers for {self.settings.model_id!r} "
            f"({type(self._hf_model).__name__})"
        )

    def get_layer_modules(self, layer_index: int) -> dict[str, list[nn.Module]]:
        """Abliterable modules of one decoder layer.

        Returns ``{"attn_o_proj": [...], "mlp_down_proj": [...]}``; lists to
        accommodate architectures with more than one match per layer.
        """
        layer = self._layers()[layer_index]
        result: dict[str, list[nn.Module]] = {
            "attn_o_proj": [],
            "mlp_down_proj": [],
        }
        for name, module in layer.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in _ATTN_SUFFIXES:
                result["attn_o_proj"].append(module)
            elif leaf in _MLP_SUFFIXES:
                result["mlp_down_proj"].append(module)
        if not result["attn_o_proj"] or not result["mlp_down_proj"]:
            raise ValueError(
                f"Layer {layer_index} of {type(self._hf_model).__name__} lacks "
                f"o_proj/down_proj modules: {result}"
            )
        return result

    # ------------------------------------------------------------- adapters
    def setup_adapters(self, rank: int) -> None:
        """Attach zeroed PEFT LoRA adapters (r=rank, alpha=1, no dropout, no
        bias) to every abliterable module.

        Idempotent when called again with the same rank (adapters are re-zeroed);
        a different rank raises, since PEFT cannot change r in place.
        """
        if self._peft_model is not None:
            existing = next(iter(self._iter_lora_params()), None)
            if existing is not None and existing[0].shape[0] == rank:
                self.zero_adapters()
                return
            raise RuntimeError(
                f"Adapters already set up with a different rank; "
                f"build a new ModelBackend to change r (requested {rank})."
            )
        from peft import LoraConfig, get_peft_model

        targets: list[str] = []
        for name, module in self._hf_model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in _ATTN_SUFFIXES + _MLP_SUFFIXES and isinstance(
                module, nn.Linear
            ):
                targets.append(name)
        if not targets:
            raise ValueError("No abliterable Linear modules found")

        config = LoraConfig(
            r=rank,
            lora_alpha=1,
            lora_dropout=0.0,
            bias="none",
            # lora_B starts at zero => adapter is initially the identity.
            init_lora_weights=True,
            target_modules=targets,
        )
        self._peft_model = get_peft_model(self._hf_model, config)
        self.zero_adapters()

    @property
    def adapter_rank(self) -> int | None:
        """LoRA rank r of the attached adapters, or None before setup.

        Additive introspection helper (post-merge integration): surgery needs
        the adapter rank to pad lower-k plans (rank-k LoRA with k < r is exact
        when the remaining rows/cols are zero).
        """
        if self._peft_model is None:
            return None
        first = next(iter(self._iter_lora_params()), None)
        return None if first is None else int(first[0].shape[0])

    def _iter_lora_params(self):
        """Yield (lora_A_weight, lora_B_weight) for every adapted module."""
        for module in self._model.modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None:
                continue
            yield (
                lora_a[self._adapter_name].weight,
                lora_b[self._adapter_name].weight,
            )

    def zero_adapters(self) -> None:
        """Zero every lora_A/lora_B — the fast per-trial reset (identity)."""
        with torch.no_grad():
            for a, b in self._iter_lora_params():
                a.zero_()
                b.zero_()

    def set_adapter(
        self,
        layer_index: int,
        component: str,
        module_idx: int,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> None:
        """Write LoRA factors for one module (A: (r, in), B: (out, r))."""
        module = self.get_layer_modules(layer_index)[component][module_idx]
        with torch.no_grad():
            module.lora_A[self._adapter_name].weight.copy_(
                lora_A.to(
                    device=module.lora_A[self._adapter_name].weight.device,
                    dtype=module.lora_A[self._adapter_name].weight.dtype,
                )
            )
            module.lora_B[self._adapter_name].weight.copy_(
                lora_B.to(
                    device=module.lora_B[self._adapter_name].weight.device,
                    dtype=module.lora_B[self._adapter_name].weight.dtype,
                )
            )

    # --------------------------------------------------------------- prompts
    def _chat_text(self, prompt: Prompt) -> str:
        messages = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        self.tokenizer.padding_side = "right"
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # chat template already carries specials
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @staticmethod
    def _last_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        """Index of the last non-pad token per sequence (right padding)."""
        return attention_mask.sum(dim=1) - 1

    def _batches(self, prompts: list[Prompt]):
        bs = self.settings.batch_size
        for start in range(0, len(prompts), bs):
            yield prompts[start : start + bs]

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        prompts: list[Prompt],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses with the chat template applied.

        ``temperature == 0.0`` decodes greedily; otherwise samples.
        Returns decoded continuations (prompt stripped).
        """
        outputs: list[str] = []
        self.tokenizer.padding_side = "left"
        try:
            for batch in self._batches(prompts):
                texts = [self._chat_text(p) for p in batch]
                enc = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.device)
                gen_kwargs: dict = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if temperature == 0.0:
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs.update(do_sample=True, temperature=temperature)
                generated = self._model.generate(**enc, **gen_kwargs)
                new_tokens = generated[:, enc["input_ids"].shape[1] :]
                outputs.extend(
                    self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                )
        finally:
            self.tokenizer.padding_side = "right"
        return outputs

    # ------------------------------------------------------------- residuals
    @torch.no_grad()
    def get_residuals(self, prompts: list[Prompt]) -> torch.Tensor:
        """Last-prompt-position hidden states, one entry per layer plus the
        embedding output: shape (n_prompts, num_layers+1, d_model), FP32 CPU.

        Equivalent to generating one token with ``output_hidden_states`` and
        taking the hidden states that produce the first generated token
        (Arditi et al. 2024 / heretic). Winsorized per layer/feature at
        ``settings.winsorization_quantile`` across the prompt dimension.
        """
        chunks: list[torch.Tensor] = []
        for batch in self._batches(prompts):
            texts = [self._chat_text(p) for p in batch]
            enc = self._encode(texts)
            out = self._model(**enc, output_hidden_states=True)
            last = self._last_indices(enc["attention_mask"])
            # hidden_states: (L+1) tensors of (batch, seq, d); [0] = embedding.
            per_layer = torch.stack(
                [
                    h[torch.arange(h.shape[0]), last].to(torch.float32)
                    for h in out.hidden_states
                ],
                dim=1,
            )  # (batch, L+1, d)
            chunks.append(per_layer.cpu())
        residuals = torch.cat(chunks, dim=0)

        q = self.settings.winsorization_quantile
        if 0.5 < q < 1.0 and residuals.shape[0] > 1:
            lo = residuals.quantile(1.0 - q, dim=0, keepdim=True)
            hi = residuals.quantile(q, dim=0, keepdim=True)
            residuals = torch.clamp(residuals, min=lo, max=hi)
        return residuals

    @torch.no_grad()
    def get_logits(self, prompts: list[Prompt]) -> torch.Tensor:
        """First-generated-token logits: shape (n_prompts, vocab), FP32 CPU."""
        chunks: list[torch.Tensor] = []
        for batch in self._batches(prompts):
            texts = [self._chat_text(p) for p in batch]
            enc = self._encode(texts)
            out = self._model(**enc)
            last = self._last_indices(enc["attention_mask"])
            logits = out.logits[torch.arange(out.logits.shape[0]), last]
            chunks.append(logits.to(torch.float32).cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def teacher_forced_logprobs(
        self, prompts: list[Prompt], continuations: list[str]
    ) -> torch.Tensor:
        """Summed logprob of each continuation under the chat template.

        Tokenizes prompt (template, generation prompt on) and continuation
        separately, concatenates, forwards once per example, and sums the
        logprobs of the continuation tokens. Returns shape (n,), FP32 CPU.
        """
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations must align")
        results: list[torch.Tensor] = []
        for prompt, continuation in zip(prompts, continuations):
            prompt_ids = self.tokenizer(
                self._chat_text(prompt), add_special_tokens=False
            )["input_ids"]
            cont_ids = self.tokenizer(
                continuation, add_special_tokens=False
            )["input_ids"]
            if not cont_ids:
                results.append(torch.tensor(0.0))
                continue
            ids = torch.tensor(
                [prompt_ids + cont_ids], dtype=torch.long, device=self.device
            )
            logits = self._model(ids).logits[0].to(torch.float32)
            n_prompt = len(prompt_ids)
            # Continuation token at position n_prompt + i is predicted by the
            # logits at position n_prompt + i - 1.
            pred_logits = logits[n_prompt - 1 : n_prompt - 1 + len(cont_ids)]
            logprobs = F.log_softmax(pred_logits, dim=-1)
            token_logprobs = logprobs.gather(
                1, torch.tensor(cont_ids, device=logprobs.device).unsqueeze(1)
            ).squeeze(1)
            results.append(token_logprobs.sum().cpu())
        return torch.stack(results)

    # ------------------------------------------------------------ checkpoint
    def save_merged(self, path: str) -> None:
        """Merge LoRA adapters into the base weights and save_pretrained.

        Filesystems safetensors' serializer cannot write (FUSE/portal mounts
        that reject its direct I/O) fall back to serializing into a local
        temp dir and copying the files over with plain writes; shards are
        capped at 90 MB since such mounts commonly reject large files.
        ``safetensors.SafetensorError`` does not subclass ``OSError``, so the
        fallback triggers on any failure of the direct save; a genuine
        serialization bug simply re-raises from the temp-dir save.
        """
        model = self._hf_model
        if self._peft_model is not None:
            model = self._peft_model.merge_and_unload()
            self._peft_model = None
            self._hf_model = model
        try:
            model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception:
            import shutil
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                model.save_pretrained(tmp, max_shard_size="90MB")
                self.tokenizer.save_pretrained(tmp)
                dest = Path(path)
                dest.mkdir(parents=True, exist_ok=True)
                for file in Path(tmp).iterdir():
                    # Plain chunked copy: sendfile(2)-based fast copies are
                    # rejected by the same mounts that reject safetensors.
                    with open(file, "rb") as src, open(
                        dest / file.name, "wb"
                    ) as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)

    def restore_base_weights(self) -> None:
        """Restore base weights from the CPU copy pinned at __init__.

        LoRA factors are untouched (call :meth:`zero_adapters` separately).
        Raises RuntimeError when pinning was skipped for a >8GiB model.
        """
        if self._pinned is None:  # pragma: no cover - large models only
            raise RuntimeError(
                "Base weights were not pinned at init (model > 8GiB); "
                "restore_base_weights() is unavailable for this model."
            )
        current = self._hf_model.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        for key, tensor in current.items():
            pinned_key = key.replace(".base_layer.", ".")  # undo PEFT wrapping
            if pinned_key in self._pinned:
                mapped[key] = self._pinned[pinned_key].to(
                    device=tensor.device, dtype=tensor.dtype
                )
        self._hf_model.load_state_dict(mapped, strict=False)
