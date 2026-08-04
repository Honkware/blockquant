"""Fused ablate+quantize via ExLlamaV3 (SPEC ADDENDUM v0.2.0 §A–§D).

Naive ablate-then-quant lets quantization silently rebalance refusal back in
(12–68pp swings). This module makes ablation and quantization ONE aware
pipeline:

* **baked** (default): merge the ablation into the weights
  (:func:`exliberate.surgery.export_exact`), quantize the merged model with a
  refusal-aware calibration mix (:func:`build_calibration` /
  :func:`install_calibration_hook`), and if post-quant rebound exceeds the
  threshold, bump ``max_weight`` ×1.15 and re-quant (max
  :data:`BAKED_MAX_ITERATIONS` iterations). Ships one self-contained EXL3
  model — nothing else to load.
* **residual** (optional): quantize the *base* model to EXL3 with the same
  hooked calibration, then attach the ablation as an fp16 runtime LoRA (our
  ablation IS a rank-k LoRA: ``lora_A = VᵀW``, ``lora_B = -w·V``).
  Quantization never touches the ablation, and the fp16 residual is sweepable
  post-quant (:func:`lambda_sweep`) with no re-quant.

ExLlamaV3 is lazy-imported everywhere; without it (or without CUDA) every
entry point raises :class:`QuantUnavailable` with install instructions — the
rest of the package never breaks. All calibration-mix and correction-loop
logic is pure and CPU-testable with fakes.

LoRA adapter directory format (residual mode output): standard PEFT layout
that ExLlamaV3's ``model/lora.py`` loads via ``LoRA.from_directory`` —
``adapter.safetensors`` with ``base_model.model.<dotted module
name>.lora_A.weight`` / ``.lora_B.weight`` tensors (PEFT conventions: A is
``(r, in_features)``, B is ``(out_features, r)``, fp16) plus
``adapter_config.json`` with ``lora_alpha == r`` (alpha/r == 1: our factors
are pre-scaled) and an ``exliberate`` metadata block recording λ.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

# Bound at module import time (not call time): the same tent kernel / basis
# machinery surgery uses, shared so the math cannot drift. Import-time binding
# also matches the suite-wide convention — test doubles that stub
# ``exliberate.surgery`` in sys.modules only affect call-time imports.
from .surgery import TrialPlan, _base_weight, _basis_rows, tent_weight

QUANT_INSTALL_HINT = (
    "fused quantization requires ExLlamaV3 and a CUDA GPU.\n"
    "Install:  pip install exllamav3        (or: pip install exliberate[quant])\n"
    "ExLlamaV3 needs torch with CUDA >= 12.4 and builds a CUDA extension;\n"
    "see https://github.com/turboderp-org/exllamav3 for wheel/build notes."
)


class QuantUnavailable(RuntimeError):
    """ExLlamaV3 and/or CUDA are not available on this machine.

    Carries install instructions; callers (CLI) must catch this and degrade
    gracefully — it never aborts the package.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or QUANT_INSTALL_HINT)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionSettings:
    """Configuration for :func:`run_fusion_pipeline` (SPEC §D).

    Kept separate from ``config.Settings`` (core-owned); the CLI builds one
    from its flags. Any duck-typed object with these attribute names is also
    accepted by :func:`run_fusion_pipeline`.
    """

    model_dir: str  # HF id / local path of the BASE model to quantize
    out_dir: str  # where the EXL3 model (+ adapter/ in residual mode) goes
    work_dir: str | None = None  # converter scratch dir (default: <out_dir>-work)
    bits: float = 4.25
    head_bits: int = 6
    devices: str = "0"
    extra_args: tuple[str, ...] | None = None  # extra convert.py CLI tokens
    cal_frac: float = 0.15  # fraction of calibration rows replaced by prompts
    cal_seed: int = 0
    lambda_sweep: tuple[float, ...] = (0.9, 1.0, 1.1, 1.25)
    rebound_threshold_pp: float = 5.0  # baked-mode trigger / pass criterion
    fusion: str = "baked"
    max_new_tokens: int = 64


def _fusion_settings(settings: Any) -> FusionSettings:
    """Coerce a duck-typed settings object into :class:`FusionSettings`."""
    if isinstance(settings, FusionSettings):
        return settings
    values: dict[str, Any] = {}
    for name in FusionSettings.__dataclass_fields__:
        if hasattr(settings, name):
            values[name] = getattr(settings, name)
    if "model_dir" not in values and hasattr(settings, "model_id"):
        values["model_dir"] = getattr(settings, "model_id")
    if "out_dir" not in values and hasattr(settings, "output_dir"):
        values["out_dir"] = str(getattr(settings, "output_dir"))
    missing = {"model_dir", "out_dir"} - values.keys()
    if missing:
        raise ValueError(f"fusion settings missing required fields: {sorted(missing)}")
    return FusionSettings(**values)


# --------------------------------------------------------------------------
# §B — refusal-aware calibration
# --------------------------------------------------------------------------


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize ``text`` to a flat list of ids across tokenizer flavors."""
    if hasattr(tokenizer, "__call__"):
        out = tokenizer(text, add_special_tokens=False)
        if isinstance(out, Mapping) and "input_ids" in out:
            ids = out["input_ids"]
            if ids and isinstance(ids[0], list):  # batched
                ids = ids[0]
            return list(ids)
    if hasattr(tokenizer, "encode"):  # pragma: no cover - exllamav3 path
        ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            return ids.flatten().tolist()
        # tokenizers.Tokenizer (the Rust fast tokenizer) returns an Encoding
        if hasattr(ids, "ids"):
            return list(ids.ids)
        return list(ids)
    raise TypeError(f"cannot tokenize with {type(tokenizer).__name__}")


def _render_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """Chat-format ``messages`` across tokenizer flavors.

    The converter hands the calibration hook whatever tokenizer it holds. That
    may be a transformers tokenizer (``apply_chat_template``), an ExLlamaV3
    Tokenizer (``hf_render_chat_template``) or the bare ``tokenizers.Tokenizer``
    underneath it, which has no template support at all. Falls back to a plain
    role-tagged transcript so calibration still sees both sides of the
    exchange rather than crashing.
    """
    for obj in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if obj is None:
            continue
        for attr in ("apply_chat_template", "hf_render_chat_template"):
            fn = getattr(obj, attr, None)
            if fn is None:
                continue
            try:
                out = fn(messages, tokenize=False,
                         add_generation_prompt=add_generation_prompt)
            except TypeError:
                try:
                    out = fn(messages, add_generation_prompt=add_generation_prompt)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(out, str) and out:
                return out
    parts = [f"{m['role']}: {m['content']}" for m in messages]
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)


def _prompt_row(tokenizer: Any, prompt: Any, cal_cols: int) -> torch.Tensor:
    """One (1, cal_cols) LongTensor row: chat-formatted prompt, tiled.

    Prompts carrying a ``response`` attribute (harmful/harmless prompts with
    short reference completions) are formatted as a full user+assistant
    exchange so the Hessian sees *both* sides of the refusal subspace;
    plain prompts get ``add_generation_prompt=True``.
    """
    hf_tok = getattr(tokenizer, "tokenizer", tokenizer)  # unwrap ExLlamaV3 Tokenizer
    messages: list[dict[str, str]] = []
    if getattr(prompt, "system", None):
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.user})
    response = getattr(prompt, "response", None)
    text = _render_chat(tokenizer, messages, add_generation_prompt=not response)
    if response:
        messages.append({"role": "assistant", "content": response})
        text = _render_chat(tokenizer, messages, add_generation_prompt=False)
    ids = _encode_ids(hf_tok, text)
    if not ids:
        ids = [0]
    reps = -(-cal_cols // len(ids))  # ceil: tile to cover cal_cols
    tiled = (ids * reps)[:cal_cols]
    return torch.tensor([tiled], dtype=torch.long)


def build_calibration(
    default_fn: Callable[[Any, Any], Sequence[torch.Tensor]],
    args: Any,
    tokenizer: Any,
    prompts: Sequence[Any],
    frac: float = 0.15,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Refusal-aware calibration mix (SPEC §B).

    Calls ``default_fn(args, tokenizer)`` for the base rows, then replaces
    ~``frac`` of them (seeded choice) with rows tokenized from the
    chat-formatted harmful+harmless extraction ``prompts`` (+ short responses
    where available). Row count and the (1, cal_cols) shape/dtype of every
    row are preserved; unreplaced rows are the original tensors untouched.

    Pure function; CPU-testable with a fake tokenizer.
    """
    rows = list(default_fn(args, tokenizer))
    if not rows or not prompts or frac <= 0.0:
        return rows
    cal_cols = int(getattr(args, "cal_cols", None) or rows[0].shape[-1])
    n_replace = min(len(rows), max(1, int(round(len(rows) * frac))))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n_replace))
    ordered = list(prompts)
    rng.shuffle(ordered)  # seeded shuffle: prompt order is deterministic
    for i, row_idx in enumerate(indices):
        rows[row_idx] = _prompt_row(tokenizer, ordered[i % len(ordered)], cal_cols)
    return rows


def install_calibration_hook(
    prompts: Sequence[Any],
    frac: float = 0.15,
    seed: int = 0,
) -> Callable[[], None]:
    """Monkeypatch the ExLlamaV3 converter's calibration source (SPEC §B).

    Replaces ``exllamav3.conversion.convert_model.get_default_calibration``
    (a by-name import inside the converter, so the *convert_model* namespace
    must be patched) with a wrapper that calls the original for the base rows
    and mixes in our prompt rows via :func:`build_calibration`.

    Returns a ``restore()`` callable that undoes the patch. Raises
    :class:`QuantUnavailable` when exllamav3 is not importable.
    """
    try:
        import exllamav3.conversion.convert_model as cm
    except ImportError as exc:
        raise QuantUnavailable() from exc
    original = cm.get_default_calibration

    def hooked(args: Any, tokenizer: Any) -> list[torch.Tensor]:
        return build_calibration(original, args, tokenizer, prompts, frac=frac, seed=seed)

    cm.get_default_calibration = hooked

    def restore() -> None:
        cm.get_default_calibration = original

    return restore


# --------------------------------------------------------------------------
# §A — programmatic ExLlamaV3 conversion
# --------------------------------------------------------------------------


def quantize_base_to_exl3(
    model_dir: str | Path,
    work_dir: str | Path,
    out_dir: str | Path,
    bits: float = 4.25,
    head_bits: int = 6,
    devices: str = "0",
    extra_args: Sequence[str] | None = None,
) -> Path:
    """Quantize ``model_dir`` (HF fp16) to EXL3 at ``out_dir``.

    Mirrors ``convert.py``: builds the argparse Namespace the converter
    expects by parsing CLI tokens through the converter's own module-level
    ``parser`` (so every parser default is inherited verbatim), then runs
    ``prepare()`` + ``main()``. Quantization is CUDA-only; raises
    :class:`QuantUnavailable` without CUDA or exllamav3.
    """
    if not torch.cuda.is_available():
        raise QuantUnavailable(
            "ExLlamaV3 quantization requires a CUDA GPU (none detected).\n\n"
            + QUANT_INSTALL_HINT
        )
    try:
        import exllamav3.conversion.convert_model as cm
    except ImportError as exc:
        raise QuantUnavailable() from exc

    argv = [
        "-i", str(model_dir),
        "-o", str(out_dir),
        "-w", str(work_dir),
        "-b", str(bits),
        "-hb", str(head_bits),
        "-d", str(devices),
    ]
    if extra_args:
        argv.extend(str(a) for a in extra_args)
    args = cm.parser.parse_args(argv)
    in_args, job_state, ok, err = cm.prepare(args)
    if not ok:
        raise RuntimeError(f"ExLlamaV3 conversion prepare() failed: {err}")
    cm.main(in_args, job_state)
    return Path(out_dir)


# --------------------------------------------------------------------------
# §A — the ablation as LoRA factors
# --------------------------------------------------------------------------


def plan_to_lora_factors(
    model: Any, subspace: Any, plan: Any
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute the plan's rank-k LoRA factors per module, keyed by name.

    Same math as :func:`exliberate.surgery.plan_to_adapters` (shared
    ``_basis_rows`` / tent kernel / factorization), but instead of writing
    PEFT adapters it returns ``{dotted_module_name: (lora_A, lora_B)}`` with
    ``lora_A`` (k, in_features) and ``lora_B`` (out_features, k) in fp32 on
    CPU — the exact ΔW = lora_B @ lora_A. Modules with tent weight 0 are
    omitted. Keys are HF dotted names (``model.layers.N.self_attn.o_proj`` /
    ``model.layers.N.mlp.down_proj``), which match both PEFT tensor naming
    and ExLlamaV3's module keys.
    """
    names: dict[int, str] = {}
    hf_model = getattr(model, "_hf_model", model)
    for name, module in hf_model.named_modules():
        names[id(module)] = name

    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_index in range(model.num_layers):
        rows_cache: torch.Tensor | None = None
        for component, params in plan.params.items():
            weight = tent_weight(layer_index, params)
            if weight == 0.0:
                continue
            if rows_cache is None:
                rows_cache = _basis_rows(subspace, plan, layer_index)
            for module in model.get_layer_modules(layer_index)[component]:
                key = names.get(id(module)) or names.get(
                    id(getattr(module, "base_layer", module))
                )
                if key is None:
                    raise ValueError(
                        f"module at layer {layer_index} ({component}) not found "
                        "in the model's named_modules()"
                    )
                w = _base_weight(module).detach().to(torch.float32).cpu()
                lora_A = rows_cache @ w  # (k, in_features)
                lora_B = -weight * rows_cache.T  # (out_features, k)
                factors[key] = (lora_A, lora_B)
    return factors


def _register_lora_tensors(
    exl3_model: Any,
    lora_A_by_module: Mapping[str, torch.Tensor],
    lora_B_by_module: Mapping[str, torch.Tensor],
    key: str = "exliberate",
) -> None:
    """Register factors directly on a loaded ExLlamaV3 model (runtime path).

    ExLlamaV3 applies LoRA post-dequant as ``x += x_in @ A @ B`` with
    A ``[in_features, r]`` and B ``[r, out_features]`` fp16, so our PEFT-layout
    factors are transposed on the way in. This mutates no weights and is
    reload-cheap — the λ-sweep knob.
    """
    matched = 0
    for module in getattr(exl3_model, "modules", []):
        module_key = getattr(module, "key", None)
        if module_key not in lora_A_by_module:
            continue
        A = lora_A_by_module[module_key]
        B = lora_B_by_module[module_key]
        device = next(module.parameters(), torch.tensor(0.0)).device \
            if hasattr(module, "parameters") else torch.device("cpu")
        module.lora_a_tensors[key] = A.T.to(device=device, dtype=torch.float16).contiguous()
        module.lora_b_tensors[key] = B.T.to(device=device, dtype=torch.float16).contiguous()
        matched += 1
    if matched != len(lora_A_by_module):  # pragma: no cover - GPU-only path
        raise RuntimeError(
            f"registered {matched}/{len(lora_A_by_module)} LoRA targets; "
            "the EXL3 model's module keys do not match the plan's module names."
        )


def attach_ablation_lora(
    exl3_model_dir_or_obj: Any,
    lora_A_by_module: Mapping[str, torch.Tensor],
    lora_B_by_module: Mapping[str, torch.Tensor],
    out_dir: str | Path,
    lambda_: float = 1.0,
) -> Path:
    """Write a standard ExLlamaV3-loadable LoRA adapter directory.

    Layout: ``adapter.safetensors`` (PEFT tensor names
    ``base_model.model.<module>.lora_A.weight`` / ``.lora_B.weight``, fp16)
    + ``adapter_config.json`` (PEFT LoraConfig fields, ``lora_alpha == r``
    so ExLlamaV3's α/r scaling is exactly 1 — our factors are pre-scaled —
    plus an ``exliberate`` metadata block with the chosen λ). Loadable via
    ``exllamav3.model.lora.LoRA.from_directory(model, out_dir,
    lora_scaling=1.0)``.

    When ``exl3_model_dir_or_obj`` is a loaded ExLlamaV3 model object (not a
    path), the factors are additionally registered directly on it
    (:func:`_register_lora_tensors`) for immediate in-process use.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not lora_A_by_module:
        raise ValueError("no LoRA factors given")
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    for module_key in lora_A_by_module:
        A = lora_A_by_module[module_key]
        B = lora_B_by_module[module_key]
        tensors[f"base_model.model.{module_key}.lora_A.weight"] = (
            A.to(torch.float16).contiguous()
        )
        tensors[f"base_model.model.{module_key}.lora_B.weight"] = (
            B.to(torch.float16).contiguous()
        )
    save_file(tensors, str(out_dir / "adapter.safetensors"))

    rank = int(next(iter(lora_A_by_module.values())).shape[0])
    base = (
        str(exl3_model_dir_or_obj)
        if isinstance(exl3_model_dir_or_obj, (str, Path))
        else None
    )
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": rank,  # alpha / r == 1: factors are pre-scaled
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": sorted(lora_A_by_module),
        "base_model_name_or_path": base,
        "exliberate": {
            "format": "exl3-ablation-residual",
            "lambda": float(lambda_),
            "note": "B already contains -lambda*w*V; load with lora_scaling=1.0",
        },
    }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    if not isinstance(exl3_model_dir_or_obj, (str, Path)):
        _register_lora_tensors(
            exl3_model_dir_or_obj, lora_A_by_module, lora_B_by_module
        )
    return out_dir


# --------------------------------------------------------------------------
# §C — validation helpers + correction loop
# --------------------------------------------------------------------------


def rebound_pp(refusal_pre: float, refusal_post: float) -> float:
    """Refusal rebound in percentage points: (post - pre) × 100."""
    return (float(refusal_post) - float(refusal_pre)) * 100.0


def validation_report(
    refusal_pre: float,
    refusal_post: float,
    kl_post_vs_pre: float,
    threshold_pp: float = 5.0,
    **extra: Any,
) -> dict[str, Any]:
    """The SPEC §C report dict; ``passed`` = rebound within threshold."""
    rb = rebound_pp(refusal_pre, refusal_post)
    report: dict[str, Any] = {
        "refusal_pre": float(refusal_pre),
        "refusal_post": float(refusal_post),
        "kl_post_vs_pre": float(kl_post_vs_pre),
        "rebound_pp": float(rb),
        "passed": bool(rb <= threshold_pp),
    }
    report.update(extra)
    return report


@dataclass
class LambdaSweepResult:
    """Outcome of :func:`lambda_sweep`: best λ and per-λ metrics."""

    best_lambda: float
    results: dict[float, dict[str, Any]]


def lambda_sweep(
    validator: Callable[[dict[str, tuple[torch.Tensor, torch.Tensor]]], Mapping[str, Any]],
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    lambdas: Sequence[float],
) -> LambdaSweepResult:
    """Sweep λ multipliers on the fp16 residual (SPEC §A step 6).

    For each λ, scales every ``lora_B`` by λ (λ acts on the ablation
    strength; A = VᵀW is unchanged), calls ``validator(scaled_factors)``,
    and returns the λ minimizing the documented scalarization
    ``refusal_post + kl_post_vs_pre``. Cheap: no re-quantization, just a
    LoRA reload per candidate.
    """
    if not lambdas:
        raise ValueError("lambda_sweep needs at least one lambda")
    results: dict[float, dict[str, Any]] = {}
    best_lambda: float | None = None
    best_scalar = float("inf")
    for lam in (float(x) for x in lambdas):
        scaled = {key: (A, lam * B) for key, (A, B) in factors.items()}
        metrics = dict(validator(scaled))
        metrics.setdefault("lambda", lam)
        results[lam] = metrics
        scalar = float(metrics["refusal_post"]) + float(metrics["kl_post_vs_pre"])
        if scalar < best_scalar:
            best_lambda, best_scalar = lam, scalar
    assert best_lambda is not None
    return LambdaSweepResult(best_lambda=best_lambda, results=results)


# --------------------------------------------------------------------------
# §A — baked-mode plan bump
# --------------------------------------------------------------------------

#: Max total quantize+validate passes in baked mode (initial + re-quants).
BAKED_MAX_ITERATIONS = 2
#: max_weight multiplier applied between baked iterations.
BAKED_WEIGHT_BUMP = 1.15
#: max_weight search-space cap (SPEC §5); bumps are clamped to it.
MAX_WEIGHT_CAP = 1.5


def bump_plan_weights(plan: Any, factor: float = BAKED_WEIGHT_BUMP) -> Any:
    """Return a copy of ``plan`` with every max_weight × ``factor`` (≤ cap)."""
    params = {
        name: replace(p, max_weight=min(MAX_WEIGHT_CAP, p.max_weight * factor))
        for name, p in plan.params.items()
    }
    return TrialPlan(
        direction_index=plan.direction_index,
        k=plan.k,
        whiten=plan.whiten,
        params=params,
    )


# --------------------------------------------------------------------------
# §A — orchestration
# --------------------------------------------------------------------------


def _default_residual_validator(
    fs: FusionSettings,
    backend: Any,
    subspace: Any,
    plan: Any,
    harmful: Sequence[Any],
    harmless: Sequence[Any],
) -> Callable[[dict[str, tuple[torch.Tensor, torch.Tensor]]], Mapping[str, Any]]:
    """Real (GPU) sweep validator: pre = ablated fp16, post = EXL3 + LoRA."""
    from .exl3_backend import EXL3Backend, PostQuantValidator
    from .surgery import plan_to_adapters

    # Pre-quant reference: the fp16 model with the plan applied as adapters.
    if getattr(backend, "adapter_rank", None) is None:
        backend.setup_adapters(max(int(plan.k), 1))
    backend.zero_adapters()
    plan_to_adapters(backend, subspace, plan)

    post = EXL3Backend(fs.out_dir)
    pv = PostQuantValidator(
        pre_model=backend,
        post_model=post,
        harmful_prompts=harmful,
        benign_prompts=harmless,
        max_new_tokens=fs.max_new_tokens,
        rebound_threshold_pp=fs.rebound_threshold_pp,
    )

    def validate(scaled: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> Mapping[str, Any]:
        _register_lora_tensors(
            post.model,
            {k: a for k, (a, _) in scaled.items()},
            {k: b for k, (_, b) in scaled.items()},
        )
        return pv.validate_metrics()

    return validate


def _default_baked_validator(
    fs: FusionSettings,
    backend: Any,
    harmful: Sequence[Any],
    harmless: Sequence[Any],
) -> Callable[[Any], Mapping[str, Any]]:
    """Real (GPU) baked validator: pre = exported fp16 model, post = EXL3."""
    from .exl3_backend import EXL3Backend, PostQuantValidator

    def validate(_: Any) -> Mapping[str, Any]:
        post = EXL3Backend(fs.out_dir)
        pv = PostQuantValidator(
            pre_model=backend,
            post_model=post,
            harmful_prompts=harmful,
            benign_prompts=harmless,
            max_new_tokens=fs.max_new_tokens,
            rebound_threshold_pp=fs.rebound_threshold_pp,
        )
        return pv.validate_metrics()

    return validate


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def _run_residual(
    fs: FusionSettings,
    plan: Any,
    work_dir: Path,
    out_dir: Path,
    quantize_fn: Callable[..., Any],
    validator: Callable[[Any], Mapping[str, Any]] | None,
    backend: Any,
    subspace: Any,
    harmful: Sequence[Any],
    harmless: Sequence[Any],
) -> dict[str, Any]:
    """Residual fusion: quantize base; ablation lives as fp16 LoRA (§A)."""
    if backend is None or subspace is None:
        raise ValueError("residual fusion requires backend + subspace")
    quantize_fn(
        fs.model_dir, work_dir, out_dir, fs.bits, fs.head_bits, fs.devices,
        fs.extra_args,
    )
    factors = plan_to_lora_factors(backend, subspace, plan)
    if validator is None:
        validator = _default_residual_validator(
            fs, backend, subspace, plan, harmful, harmless
        )
    sweep = lambda_sweep(validator, factors, fs.lambda_sweep)
    best = sweep.best_lambda
    scaled = {key: (A, best * B) for key, (A, B) in factors.items()}
    adapter_dir = attach_ablation_lora(
        out_dir,
        {k: a for k, (a, _) in scaled.items()},
        {k: b for k, (_, b) in scaled.items()},
        out_dir / "adapter",
        lambda_=best,
    )
    metrics = sweep.results[best]
    report = validation_report(
        metrics["refusal_pre"],
        metrics["refusal_post"],
        metrics["kl_post_vs_pre"],
        fs.rebound_threshold_pp,
        mode="residual",
        lambda_=best,
        lambda_sweep=sweep.results,
        adapter_dir=str(adapter_dir),
        exl3_dir=str(out_dir),
    )
    _write_report(out_dir / "fusion_report.json", report)
    return report


def _run_baked(
    fs: FusionSettings,
    plan: Any,
    work_dir: Path,
    out_dir: Path,
    quantize_fn: Callable[..., Any],
    export_fn: Callable[[Any, Path], Any] | None,
    validator: Callable[[Any], Mapping[str, Any]] | None,
    backend: Any,
    subspace: Any,
    harmful: Sequence[Any],
    harmless: Sequence[Any],
) -> dict[str, Any]:
    """Baked fusion: merge ablation pre-quant; rebound-triggered re-quant."""
    if backend is None or subspace is None:
        raise ValueError("baked fusion requires backend + subspace")
    if export_fn is None:

        def export_fn(current_plan: Any, merged_dir: Path) -> None:
            from .surgery import export_exact

            export_exact(backend, subspace, current_plan, norm_preserve=True)
            backend.save_merged(str(merged_dir))

    if validator is None:
        validator = _default_baked_validator(fs, backend, harmful, harmless)

    current = plan
    attempt = 0
    report: dict[str, Any] = {}
    while True:
        merged_dir = work_dir / f"merged_iter{attempt}"
        merged_dir.mkdir(parents=True, exist_ok=True)
        export_fn(current, merged_dir)
        quantize_fn(
            merged_dir, work_dir / f"job_iter{attempt}", out_dir, fs.bits,
            fs.head_bits, fs.devices, fs.extra_args,
        )
        metrics = dict(validator(None))
        report = validation_report(
            metrics["refusal_pre"],
            metrics["refusal_post"],
            metrics["kl_post_vs_pre"],
            fs.rebound_threshold_pp,
            mode="baked",
            iterations=attempt + 1,
            max_weight={
                name: p.max_weight for name, p in current.params.items()
            },
            exl3_dir=str(out_dir),
        )
        if report["passed"] or attempt + 1 >= BAKED_MAX_ITERATIONS:
            break
        attempt += 1
        current = bump_plan_weights(current)
    report["baked_max_iterations"] = BAKED_MAX_ITERATIONS
    _write_report(out_dir / "fusion_report.json", report)
    return report


def run_fusion_pipeline(
    settings: Any,
    plan: Any,
    fusion: str = "baked",
    *,
    backend: Any = None,
    subspace: Any = None,
    harmful_prompts: Sequence[Any] | None = None,
    harmless_prompts: Sequence[Any] | None = None,
    quantize_fn: Callable[..., Any] | None = None,
    export_fn: Callable[[Any, Path], Any] | None = None,
    validator: Callable[[Any], Mapping[str, Any]] | None = None,
    install_hook_fn: Callable[..., Callable[[], None]] | None = None,
) -> dict[str, Any]:
    """Orchestrate the fused ablate+quantize pipeline (SPEC §A).

    ``settings`` is a :class:`FusionSettings` (or duck-typed equivalent);
    ``plan`` is the winning ``TrialPlan`` from the abliteration search.

    residual: quantize base (hooked calibration) → compute LoRA factors →
    λ-sweep on the fp16 residual → write adapter + ``fusion_report.json``.
    baked: export_exact → quantize merged → validate → rebound > threshold:
    bump max_weight ×1.15 and re-quant, max ``BAKED_MAX_ITERATIONS``.

    The injectable callables (``quantize_fn`` / ``export_fn`` /
    ``validator`` / ``install_hook_fn``) exist so tests can drive the whole
    loop CPU-only with fakes; ``None`` selects the real ExLlamaV3 path
    (which lazy-imports exllamav3 and raises :class:`QuantUnavailable`
    without CUDA).
    """
    fs = _fusion_settings(settings)
    fusion = fusion or fs.fusion
    out_dir = Path(fs.out_dir)
    work_dir = Path(fs.work_dir) if fs.work_dir else Path(f"{fs.out_dir}-work")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if harmful_prompts is None or harmless_prompts is None:
        from .prompts import load_prompts, packaged

        if harmful_prompts is None:
            harmful_prompts = load_prompts(packaged("harmful_extraction"))
        if harmless_prompts is None:
            harmless_prompts = load_prompts(packaged("harmless_extraction"))
    cal_prompts = [*harmful_prompts, *harmless_prompts]

    restore: Callable[[], None] | None = None
    if quantize_fn is None:
        hook = install_hook_fn or install_calibration_hook
        restore = hook(cal_prompts, frac=fs.cal_frac, seed=fs.cal_seed)
        quantize_fn = quantize_base_to_exl3
    try:
        if fusion == "residual":
            return _run_residual(
                fs, plan, work_dir, out_dir, quantize_fn, validator,
                backend, subspace, harmful_prompts, harmless_prompts,
            )
        if fusion == "baked":
            return _run_baked(
                fs, plan, work_dir, out_dir, quantize_fn, export_fn, validator,
                backend, subspace, harmful_prompts, harmless_prompts,
            )
        raise ValueError(f"unknown fusion mode: {fusion!r} (residual|baked)")
    finally:
        if restore is not None:
            restore()
