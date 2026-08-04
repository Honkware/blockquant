"""Whitened rank-k refusal subspace extraction (SPEC §3).

Per layer (including the embedding output, ``num_layers + 1`` entries) we form
four contrast vectors from winsorized residuals:

1. **pooled diff** — ``mean(harmful) - mean(harmless)`` (the heretic direction,
   Arditi et al. 2024, arXiv:2406.11717);
2. **split-A diff** / 3. **split-B diff** — the same mean diff computed on two
   deterministic random half-splits of both prompt sets (noisy replicas of the
   pooled direction, giving SVD something to rank beyond k=1);
4. **prompt-mean-centered diff** — pooled diff after centering each prompt's
   activation vector by its own mean over hidden units (removes the per-prompt
   DC offset, exposing a complementary contrast direction).

``k == 1`` returns the normalized pooled diff exactly (heretic parity — unit
tested). For ``k > 1`` the stacked contrast matrix is optionally whitened by
the inverse square root of the **harmless** covariance (Ledoit-Wolf shrinkage;
Mahalanobis = benign protection, design review P0-1, no separate deflation),
then reduced via SVD to the top-k orthonormal rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import ModelBackend
from .prompts import Prompt

# Fixed seed for the deterministic half-splits: extraction is a pure function
# of (model, prompts, k, whiten), which SPEC §5 determinism relies on.
_SPLIT_SEED = 42
# Eigenguard for the covariance inverse square root.
_EIG_EPS = 1e-6


@dataclass
class RefusalSubspace:
    """Per-layer orthonormal refusal directions (SPEC §3)."""

    bases: torch.Tensor  # (num_layers+1, k, d_model) orthonormal rows, FP32 CPU
    k: int
    direction_norms: torch.Tensor  # (num_layers+1,) ||mean diff|| per layer


def _contrast_vectors(
    resid_harmful: torch.Tensor, resid_harmless: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the 4 contrast vectors for one layer.

    ``resid_*``: (n_prompts, d_model). Returns (stacked (4, d), pooled (d,)).
    """
    pooled = resid_harmful.mean(dim=0) - resid_harmless.mean(dim=0)

    n_h, n_b = resid_harmful.shape[0], resid_harmless.shape[0]
    if n_h >= 2 and n_b >= 2:
        rng = np.random.default_rng(_SPLIT_SEED)
        perm_h = rng.permutation(n_h)
        perm_b = rng.permutation(n_b)
        h_a, h_b = perm_h[: n_h // 2], perm_h[n_h // 2 :]
        b_a, b_b = perm_b[: n_b // 2], perm_b[n_b // 2 :]
        split_a = resid_harmful[h_a].mean(dim=0) - resid_harmless[b_a].mean(dim=0)
        split_b = resid_harmful[h_b].mean(dim=0) - resid_harmless[b_b].mean(dim=0)
    else:  # degenerate tiny sets: replicate the pooled diff
        split_a = pooled.clone()
        split_b = pooled.clone()

    # Prompt-mean-centered diff: remove each prompt's DC offset over hidden
    # units, then take the class mean difference.
    centered_h = resid_harmful - resid_harmful.mean(dim=1, keepdim=True)
    centered_b = resid_harmless - resid_harmless.mean(dim=1, keepdim=True)
    mean_centered = centered_h.mean(dim=0) - centered_b.mean(dim=0)

    stacked = torch.stack([pooled, split_a, split_b, mean_centered], dim=0)
    return stacked, pooled


def _whitening_transform(
    resid_harmless: torch.Tensor, shrinkage: str
) -> torch.Tensor:
    """C^{-1/2} from the HARMLESS residual covariance (Ledoit-Wolf shrinkage).

    Eigendecomposed with eps-regularized inverse square root. Whitening the
    contrast directions by the benign covariance is Mahalanobis scaling:
    directions where benign activations already vary a lot are de-emphasized,
    protecting benign behavior (design review P0-1).
    """
    if shrinkage != "ledoit-wolf":
        raise ValueError(f"Unknown shrinkage: {shrinkage!r}")

    # Ledoit-Wolf + eigendecomposition in torch, on whatever device the
    # residuals already live on. This is the same estimator and the same
    # eps-regularized inverse square root as the sklearn/numpy version, but
    # eigh is O(d^3) (d = hidden size), so running it on the CPU in float64 --
    # after copying the residuals off the GPU -- dominated the whole search.
    x = resid_harmless.to(torch.float64)
    n, d = x.shape
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    cov = (xc.T @ xc) / n

    # Ledoit-Wolf shrinkage toward mu_hat * I (closed form, LW 2004 eq. 14).
    mu_hat = torch.diagonal(cov).mean()
    delta2 = ((cov - mu_hat * torch.eye(d, dtype=cov.dtype, device=cov.device)) ** 2).sum() / d
    beta2 = ((xc ** 2).T @ (xc ** 2) / (n * n)).sum() / d - (cov ** 2).sum() / (n * d)
    beta2 = torch.clamp(beta2, min=torch.zeros((), dtype=cov.dtype, device=cov.device), max=delta2)
    shrink = beta2 / delta2 if float(delta2) > 0 else torch.zeros((), dtype=cov.dtype, device=cov.device)
    cov = (1.0 - shrink) * cov + shrink * mu_hat * torch.eye(d, dtype=cov.dtype, device=cov.device)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    eps = _EIG_EPS * max(float(eigvals.max()), 1.0)
    inv_sqrt = (eigvecs * torch.rsqrt(torch.clamp(eigvals, min=eps))) @ eigvecs.T
    return inv_sqrt.to(torch.float32)


def extract_subspace(
    model: ModelBackend,
    harmful: list[Prompt],
    harmless: list[Prompt],
    k: int,
    whiten: bool,
    shrinkage: str = "ledoit-wolf",
    whiten_cache: dict | None = None,
) -> RefusalSubspace:
    """Extract a per-layer rank-k orthonormal refusal subspace.

    For ``k == 1`` the basis row is exactly the normalized pooled mean
    difference (heretic's direction) up to sign. For ``k > 1`` the basis is
    the top-k right-singular rows of the (optionally whitened) stacked
    contrast matrix.
    """
    if k < 1 or k > 4:
        raise ValueError(f"k must be in 1..4, got {k}")

    resid_harmful = model.get_residuals(harmful)  # (n_h, L+1, d) FP32 CPU
    resid_harmless = model.get_residuals(harmless)  # (n_b, L+1, d) FP32 CPU
    # The per-layer work is a d x d eigendecomposition plus a small SVD; on CPU
    # in float64 that dominated the whole search. Run it on the model's device
    # when there is one and copy the small results back.
    dev = getattr(model, "device", None) or torch.device("cpu")
    n_stops = resid_harmful.shape[1]  # num_layers + 1
    d_model = resid_harmful.shape[2]

    bases = torch.empty(n_stops, k, d_model, dtype=torch.float32)
    direction_norms = torch.empty(n_stops, dtype=torch.float32)

    for layer in range(n_stops):
        stacked, pooled = _contrast_vectors(
            resid_harmful[:, layer, :], resid_harmless[:, layer, :]
        )
        direction_norms[layer] = pooled.norm()

        if whiten:
            # C^{-1/2} depends only on (layer, shrinkage) -- NOT on k -- so the
            # four whitened k-combos would otherwise redo identical work.
            key = (layer, shrinkage)
            inv_sqrt = whiten_cache.get(key) if whiten_cache is not None else None
            if inv_sqrt is None:
                inv_sqrt = _whitening_transform(
                    resid_harmless[:, layer, :].to(dev), shrinkage
                )
                if whiten_cache is not None:
                    whiten_cache[key] = inv_sqrt
            stacked = stacked.to(inv_sqrt.device) @ inv_sqrt

        if k == 1:
            # Heretic parity: k=1 IS the normalized pooled diff (sign-free).
            # Degenerate layers (zero mean separation, e.g. the embedding row
            # when every prompt shares the same final template tokens) yield a
            # zero row — a safe no-op downstream.
            direction = stacked[0]
            norm = direction.norm()
            row = direction / norm if norm > 0 else direction
            bases[layer, 0] = row.to(bases.device, torch.float32)
        else:
            # Top-k orthonormal rows via SVD of the stacked contrasts.
            _, _, vh = torch.linalg.svd(stacked.to(dev), full_matrices=False)
            rows = vh[:k]
            if rows.shape[0] < k:  # d_model < k, pathological tiny models
                pad = torch.zeros(k - rows.shape[0], d_model, device=rows.device)
                rows = torch.cat([rows, pad], dim=0)
            bases[layer] = rows.to(bases.device, torch.float32)

    return RefusalSubspace(bases=bases, k=k, direction_norms=direction_norms)
