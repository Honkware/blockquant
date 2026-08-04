"""Rank-k abliteration math: trial plans → LoRA adapters → exact export.

The per-layer ablation strength follows heretic's tent kernel
(github.com/p-e-w/heretic): peak ``max_weight`` at ``max_weight_position``,
linear decay to ``min_weight`` at distance ``min_weight_distance``.
``max_weight > 1.0`` is intentional boundary overshoot (design review P0-5).

Exact rank-k application (SPEC §3): with unit-norm basis rows ``v_j``,

    ΔW = -w * Σ_j v_j (v_jᵀ W)        (v over the OUTPUT dimension of W)

which factorizes exactly as a rank-k LoRA: ``lora_A = stack(v_jᵀ W)``,
``lora_B = -w * stack(v_j)`` (columns). :func:`export_exact` applies the same
ΔW directly to the base weights, optionally renormalizing rows to their
original norms (norm-preserving bilateral abliteration, grimjim NPBA).

Layer→basis mapping: decoder layer ``l`` uses basis row ``l + 1`` — the
residual stream *after* that layer (basis row 0 is the embedding output).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import ModelBackend
from .subspace import RefusalSubspace


@dataclass(frozen=True)
class AbliterationParameters:
    """Tent-kernel parameters for one component (attn_o_proj/mlp_down_proj)."""

    max_weight: float  # [0.0, 1.5] — >1.0 IS the boundary-overshoot (P0-5)
    max_weight_position: float  # layer float in [0, num_layers-1]
    min_weight: float  # [0.0, 1.0]
    min_weight_distance: float  # >= 1.0


@dataclass(frozen=True)
class TrialPlan:
    """One point in search space (SPEC §3/§5)."""

    direction_index: float | None  # None = per-layer bases; float = interp. global
    k: int  # subspace rank 1..4 (shared across regions)
    whiten: bool
    params: dict[str, AbliterationParameters]


def tent_weight(layer_index: int, params: AbliterationParameters) -> float:
    """Heretic-style tent kernel: linear from ``max_weight`` at the peak
    position down to ``min_weight`` at ``min_weight_distance`` and beyond."""
    distance = abs(layer_index - params.max_weight_position)
    if distance >= params.min_weight_distance:
        return params.min_weight
    span = params.max_weight - params.min_weight
    return params.min_weight + span * (1.0 - distance / params.min_weight_distance)


def _base_weight(module: nn.Module) -> torch.Tensor:
    """Weight of the underlying Linear, unwrapping PEFT's LoRA wrapper."""
    base = getattr(module, "base_layer", module)
    return base.weight


def _basis_rows(
    subspace: RefusalSubspace, plan: TrialPlan, layer_index: int
) -> torch.Tensor:
    """Unit-norm basis rows (k, d_model) for decoder layer ``layer_index``.

    Per-layer mode uses row ``layer_index + 1``. Interpolated-global mode
    linearly blends adjacent basis rows at the float ``direction_index`` and
    renormalizes each blended row to unit norm.
    """
    k = min(plan.k, subspace.k)
    n_rows = subspace.bases.shape[0]  # num_layers + 1
    if plan.direction_index is None:
        return subspace.bases[layer_index + 1, :k]
    position = min(max(float(plan.direction_index), 0.0), n_rows - 1.0)
    lo = int(position)
    hi = min(lo + 1, n_rows - 1)
    frac = position - lo
    rows = (1.0 - frac) * subspace.bases[lo, :k] + frac * subspace.bases[hi, :k]
    norms = rows.norm(dim=1, keepdim=True)
    return rows / norms.clamp_min(1e-12)


def plan_to_adapters(
    model: ModelBackend, subspace: RefusalSubspace, plan: TrialPlan
) -> None:
    """Realize a trial plan as exact rank-k LoRA factors on the model.

    Runs AFTER :meth:`ModelBackend.zero_adapters` (the caller resets between
    trials); components/layers with ``w == 0`` are left zeroed.

    Plans with ``k`` below the adapter rank r (e.g. a k=1 heretic-equivalent
    plan on adapters set up with r=4) are exact: the LoRA factors are zero
    padded to rank r, contributing nothing to ``lora_B @ lora_A``. If the
    adapter rank is somehow smaller than k, the basis rows are sliced.
    """
    # Adapter rank when known (real ModelBackend); None for protocol-only
    # backends (tests), where set_adapter receives the unpadded factors.
    rank = getattr(model, "adapter_rank", None)
    for layer_index in range(model.num_layers):
        rows_cache: torch.Tensor | None = None
        for component, params in plan.params.items():
            weight = tent_weight(layer_index, params)
            if weight == 0.0:
                continue
            if rows_cache is None:
                rows_cache = _basis_rows(subspace, plan, layer_index)
                if rank is not None:
                    if rows_cache.shape[0] < rank:
                        pad = torch.zeros(
                            rank - rows_cache.shape[0], rows_cache.shape[1]
                        )
                        rows_cache = torch.cat([rows_cache, pad], dim=0)
                    elif rows_cache.shape[0] > rank:
                        rows_cache = rows_cache[:rank]
            for module_idx, module in enumerate(
                model.get_layer_modules(layer_index)[component]
            ):
                w = _base_weight(module).detach().to(torch.float32).cpu()
                # lora_A row j = v_jᵀ W ; lora_B col j = -w * v_j.
                lora_a = rows_cache @ w  # (r, in_features)
                lora_b = -weight * rows_cache.T  # (out_features, r)
                model.set_adapter(
                    layer_index, component, module_idx, lora_a, lora_b
                )


def export_exact(
    model: ModelBackend,
    subspace: RefusalSubspace,
    plan: TrialPlan,
    norm_preserve: bool = True,
) -> None:
    """Apply the plan exactly to the base weights, in place.

    Restores the pinned base weights first, zeroes adapters (the exact path
    bypasses LoRA), then patches ``W' = W - w Σ_j v_j (v_jᵀ W)``. With
    ``norm_preserve`` each row is rescaled to its original norm (grimjim
    NPBA), which empirically preserves capabilities better than raw
    directional projection.
    """
    model.restore_base_weights()
    model.zero_adapters()
    with torch.no_grad():
        for layer_index in range(model.num_layers):
            rows_cache: torch.Tensor | None = None
            for component, params in plan.params.items():
                weight = tent_weight(layer_index, params)
                if weight == 0.0:
                    continue
                if rows_cache is None:
                    rows_cache = _basis_rows(subspace, plan, layer_index)
                for module in model.get_layer_modules(layer_index)[component]:
                    w_param = _base_weight(module)
                    w = w_param.detach().to(torch.float32).cpu()
                    original_row_norms = w.norm(dim=1, keepdim=True)
                    delta = -weight * (rows_cache.T @ (rows_cache @ w))
                    new_w = w + delta
                    if norm_preserve:
                        new_norms = new_w.norm(dim=1, keepdim=True)
                        scale = torch.where(
                            new_norms > 0, original_row_norms / new_norms.clamp_min(1e-12), torch.ones_like(new_norms)
                        )
                        new_w = new_w * scale
                    w_param.copy_(new_w.to(device=w_param.device, dtype=w_param.dtype))
