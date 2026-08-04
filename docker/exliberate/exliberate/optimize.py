"""Multi-objective abliteration search (SPEC §5).

Optuna MOTPE (TPESampler(multivariate=True)) over three minimized objectives
[keyword_refusal_rate, multi_token_KL, capability_delta], ASHA
(SuccessiveHalvingPruner) pruning on a 32-prompt refusal subset at rung 1,
Pareto-front selection by documented scalarization, and a re-probe loop that
re-extracts the refusal subspace after the best plan is applied.

Core modules (exliberate.subspace / exliberate.surgery) are imported lazily
inside functions so this module stays importable before the core branch is
merged, and so tests can stub them via ``sys.modules``.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import optuna

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Settings
    from .model import ModelBackend
    from .prompts import Prompt
    from .scorers.base import Scorer
    from .subspace import RefusalSubspace
    from .surgery import TrialPlan

#: Component names matching ModelBackend.get_layer_modules() / TrialPlan.params.
COMPONENTS: tuple[str, str] = ("attn_o_proj", "mlp_down_proj")

#: Objective keys expected in the ``scorers`` mapping (all minimized).
DEFAULT_OBJECTIVES: tuple[str, str, str] = ("keyword", "kl", "capability")

#: Rung-1 pruner subset size (SPEC §5: keyword refusal rate on 32 prompts).
RUNG1_SUBSET_SIZE = 32

#: Re-probe trigger: residual max direction norm > this fraction of original.
REPROBE_THRESHOLD = 0.25

_EPS = 1e-12


@dataclass
class StudyResult:
    """Result of one optimization study."""

    plan: "TrialPlan"                    # best Pareto plan by scalarization
    scalarized: float                    # scalarized objective of the best plan
    objective_values: tuple[float, ...]  # per-objective values of the best plan
    study: optuna.study.Study


@dataclass
class ReprobeResult:
    """Result of the re-probe loop (SPEC §5).

    Semantics (v1, documented): each round re-extracts the refusal subspace
    from the CURRENT model state (previous round's plan active via adapters).
    If the residual max direction norm exceeds 0.25x the original, a
    half-budget round-2 study runs against the residual subspaces. The best
    plan of the LAST round wins and is returned together with the subspace
    cache it was computed against, so callers can apply/export it coherently.
    """

    final_plan: "TrialPlan"
    subspaces: dict[tuple[int, bool], "RefusalSubspace"]
    rounds: int
    residual_ratios: list[float] = field(default_factory=list)


def seed_everything(seed: int) -> None:
    """Full determinism: python, numpy, torch (SPEC §5)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in sandbox
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def precompute_subspaces(
    model: "ModelBackend",
    harmful: Sequence["Prompt"],
    harmless: Sequence["Prompt"],
    k_values: Sequence[int] = (1, 2, 3, 4),
) -> dict[tuple[int, bool], "RefusalSubspace"]:
    """Extract refusal subspaces for all (k, whiten) combos ONCE (SPEC §5).

    v1 simplification: 4 k-values x 2 whiten = 8 extractions up front on the
    extraction prompt sets.
    """
    from .subspace import extract_subspace

    cache: dict[tuple[int, bool], "RefusalSubspace"] = {}
    # The whitening transform is per (layer, shrinkage) and identical across
    # k, so share one cache across every extraction instead of recomputing a
    # d x d eigendecomposition per layer per k.
    whiten_cache: dict = {}
    for k in k_values:
        for whiten in (False, True):
            cache[(int(k), whiten)] = extract_subspace(
                model, harmful, harmless, int(k), whiten, whiten_cache=whiten_cache
            )
    return cache


def _suggest_plan(
    trial: optuna.Trial,
    num_layers: int,
    k_values: Sequence[int],
    fixed_params: Mapping[str, Any] | None = None,
) -> "TrialPlan":
    """Sample a TrialPlan from the SPEC §5 search space.

    - k in k_values (default {1,2,3,4}), whiten in {False, True}
    - direction_index: "per_layer" (None) or global float in [0, num_layers-1]
    - per component: max_weight [0, 1.5] (>1 = boundary overshoot),
      max_weight_position [0, L-1], min_weight [0, 1.0],
      min_weight_distance [1, L/2]
    """
    from .surgery import AbliterationParameters, TrialPlan

    fixed_params = dict(fixed_params or {})
    L = num_layers

    def pick(name: str, suggest, default=None):
        if name in fixed_params:
            return fixed_params[name]
        return suggest() if default is None else suggest(default)

    if "k" in fixed_params:
        k = int(fixed_params["k"])
    else:
        k = trial.suggest_categorical("k", [int(x) for x in k_values])
    if "whiten" in fixed_params:
        whiten = bool(fixed_params["whiten"])
    else:
        whiten = bool(trial.suggest_categorical("whiten", [0, 1]))
    direction_mode = trial.suggest_categorical("direction_mode", ["per_layer", "global"])
    if direction_mode == "global":
        direction_index: float | None = trial.suggest_float(
            "direction_index", 0.0, float(L - 1)
        )
    else:
        direction_index = None

    params: dict[str, "AbliterationParameters"] = {}
    for comp in COMPONENTS:
        params[comp] = AbliterationParameters(
            max_weight=pick(
                f"{comp}.max_weight",
                lambda: trial.suggest_float(f"{comp}.max_weight", 0.0, 1.5),
            ),
            max_weight_position=pick(
                f"{comp}.max_weight_position",
                lambda: trial.suggest_float(f"{comp}.max_weight_position", 0.0, float(L - 1)),
            ),
            min_weight=pick(
                f"{comp}.min_weight",
                lambda: trial.suggest_float(f"{comp}.min_weight", 0.0, 1.0),
            ),
            min_weight_distance=pick(
                f"{comp}.min_weight_distance",
                lambda: trial.suggest_float(f"{comp}.min_weight_distance", 1.0, max(1.0, L / 2)),
            ),
        )
    return TrialPlan(
        direction_index=direction_index, k=int(k), whiten=whiten, params=params
    )


def scalarize(values: Sequence[float]) -> float:
    """Documented Pareto selection scalarization (SPEC §5).

    ``keyword_refusal_rate + multi_token_KL + capability_delta`` — plain sum;
    all objectives are minimized and all scorers share the lower-is-better
    convention (SPEC §3).
    """
    return float(sum(values))


def plan_from_dict(d: Mapping[str, Any]) -> "TrialPlan":
    """Inverse of plan_to_dict."""
    from .surgery import AbliterationParameters, TrialPlan

    return TrialPlan(
        direction_index=d.get("direction_index"),
        k=int(d["k"]),
        whiten=bool(d["whiten"]),
        params={
            comp: AbliterationParameters(**vals) for comp, vals in d["params"].items()
        },
    )


def select_best_plan(
    study: optuna.study.Study,
    num_layers: int,
) -> tuple["TrialPlan", tuple[float, ...]]:
    """Pick the min-scalarized trial from the study's Pareto front.

    Trials carry their sampled plan in ``user_attrs['plan']`` (robust to
    fixed params absent from ``trial.params``); falls back to reconstructing
    from raw params for externally constructed studies.
    """
    from .surgery import AbliterationParameters, TrialPlan

    front = study.best_trials  # Pareto-optimal trials for multi-objective
    if not front:
        raise RuntimeError("study produced no completed Pareto-optimal trials")

    def reconstruct(t: optuna.trial.FrozenTrial) -> "TrialPlan":
        if "plan" in t.user_attrs:
            return plan_from_dict(t.user_attrs["plan"])
        p = t.params
        direction_index = (
            p.get("direction_index") if p.get("direction_mode") == "global" else None
        )
        params = {
            comp: AbliterationParameters(
                max_weight=float(p[f"{comp}.max_weight"]),
                max_weight_position=float(p[f"{comp}.max_weight_position"]),
                min_weight=float(p[f"{comp}.min_weight"]),
                min_weight_distance=float(p[f"{comp}.min_weight_distance"]),
            )
            for comp in COMPONENTS
        }
        return TrialPlan(
            direction_index=direction_index,
            k=int(p["k"]),
            whiten=bool(p["whiten"]),
            params=params,
        )

    best_trial = min(front, key=lambda t: scalarize(t.values))
    return reconstruct(best_trial), tuple(float(v) for v in best_trial.values)


def _make_rung1_scorer(scorers: Mapping[str, "Scorer"]) -> "Scorer":
    """Keyword scorer restricted to the 32-prompt rung-1 subset (SPEC §5).

    Uses an explicitly provided ``scorers['keyword_subset']`` if present, else
    rebuilds the keyword scorer's class on its first 32 prompts, else falls
    back to the full keyword scorer.
    """
    if "keyword_subset" in scorers:
        return scorers["keyword_subset"]
    kw = scorers["keyword"]
    prompts = getattr(kw, "prompts", None)
    if prompts is not None and len(prompts) > RUNG1_SUBSET_SIZE:
        try:
            return type(kw)(list(prompts)[:RUNG1_SUBSET_SIZE])
        except TypeError:
            pass
    return kw


def run_study(
    model: "ModelBackend",
    subspaces_cache: Mapping[tuple[int, bool], "RefusalSubspace"],
    scorers: Mapping[str, "Scorer"],
    settings: "Settings",
    *,
    objective_keys: Sequence[str] = DEFAULT_OBJECTIVES,
    fixed_params: Mapping[str, Any] | None = None,
    k_values: Sequence[int] = (1, 2, 3, 4),
    n_trials: int | None = None,
) -> StudyResult:
    """Run one MOTPE + ASHA study and return the best Pareto plan (SPEC §5).

    ``objective_keys`` selects which scorers form the objective vector (default
    [keyword, kl, capability], all minimized); benchmark's heretic-equivalent
    arm passes ("keyword", "kl"). ``fixed_params`` pins search-space entries
    (e.g. {"k": 1, "whiten": False} for the heretic-equivalent arm).
    """
    from .surgery import plan_to_adapters

    seed = int(getattr(settings, "seed", 0))
    n = int(n_trials if n_trials is not None else getattr(settings, "n_trials", 40))
    seed_everything(seed)

    objective_keys = tuple(objective_keys)
    rung1_scorer = _make_rung1_scorer(scorers)
    num_layers = int(model.num_layers)

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(
        directions=["minimize"] * len(objective_keys),
        sampler=sampler,
        pruner=pruner,
    )

    # NOTE (optuna constraint): Trial.report/should_prune are not supported for
    # multi-objective studies, so the SuccessiveHalvingPruner attached above
    # never sees intermediate values. The rung-1 ASHA gate (SPEC §5: refusal
    # subset, NOT KL) is therefore enforced by an equivalent median-cut rule
    # over prior rung-1 values inside the objective. Deterministic: the gate
    # depends only on trial completion order.
    rung1_values: list[float] = []
    ASHA_WARMUP = 4  # trials before the gate engages

    def _asha_gate(value: float) -> bool:
        if len(rung1_values) < ASHA_WARMUP:
            return False
        return value > float(np.median(rung1_values))

    def objective(trial: optuna.Trial) -> tuple[float, ...]:
        plan = _suggest_plan(trial, num_layers, k_values, fixed_params)
        trial.set_user_attr("plan", plan_to_dict(plan))
        model.zero_adapters()
        plan_to_adapters(model, subspaces_cache[(plan.k, plan.whiten)], plan)

        # Rung 1: refusal rate on the 32-prompt subset -> report + prune.
        rung1 = float(rung1_scorer.score(model).value)
        try:
            trial.report(rung1, step=1)  # single-objective studies only
            prune = trial.should_prune()
        except NotImplementedError:  # multi-objective: manual ASHA gate
            prune = _asha_gate(rung1)
        rung1_values.append(rung1)
        if prune:
            raise optuna.TrialPruned()

        # Rung 2: full objective vector.
        values = tuple(float(scorers[k].score(model).value) for k in objective_keys)
        return values

    study.optimize(objective, n_trials=n)

    plan, values = select_best_plan(study, num_layers)
    return StudyResult(
        plan=plan, scalarized=scalarize(values), objective_values=values, study=study
    )


def reprobe_loop(
    model: "ModelBackend",
    plan: "TrialPlan",
    original_subspaces: Mapping[tuple[int, bool], "RefusalSubspace"],
    scorers: Mapping[str, "Scorer"],
    settings: "Settings",
    harmful: Sequence["Prompt"],
    harmless: Sequence["Prompt"],
    *,
    max_rounds: int = 2,
    k_values: Sequence[int] = (1, 2, 3, 4),
) -> ReprobeResult:
    """Re-probe loop (SPEC §5), max 2 rounds.

    Round r: apply the current best plan via adapters, re-extract the
    subspace FROM THE CURRENT MODEL STATE, compare max direction norm
    against the original extraction. If residual > 0.25x original and
    rounds remain, precompute residual subspaces (k x whiten combos) and run
    a half-budget study on top. Returns the last round's plan plus the
    subspace cache it belongs to.
    """
    from .subspace import extract_subspace
    from .surgery import plan_to_adapters

    n_trials = int(getattr(settings, "n_trials", 40))
    original = original_subspaces[(plan.k, plan.whiten)]
    original_max = float(original.direction_norms.max()) + _EPS

    current_plan = plan
    current_subspaces: Mapping[tuple[int, bool], "RefusalSubspace"] = original_subspaces
    ratios: list[float] = []

    for round_idx in range(max_rounds):
        # Apply the current plan via adapters (SPEC: "after best plan applied
        # via adapters, re-extract subspace FROM CURRENT MODEL state").
        model.zero_adapters()
        plan_to_adapters(model, current_subspaces[(current_plan.k, current_plan.whiten)], current_plan)

        residual = extract_subspace(
            model, harmful, harmless, k=current_plan.k, whiten=current_plan.whiten
        )
        ratio = float(residual.direction_norms.max()) / original_max
        ratios.append(ratio)
        if ratio <= REPROBE_THRESHOLD or round_idx == max_rounds - 1:
            break

        # Next round: half-budget study against residual subspaces extracted
        # from the current (already ablated) state.
        residual_cache = precompute_subspaces(model, harmful, harmless, k_values)
        result = run_study(
            model,
            residual_cache,
            scorers,
            settings,
            k_values=k_values,
            n_trials=max(1, n_trials // 2),
        )
        current_plan = result.plan
        current_subspaces = residual_cache

    return ReprobeResult(
        final_plan=current_plan,
        subspaces=dict(current_subspaces),
        rounds=len(ratios),
        residual_ratios=ratios,
    )


def plan_to_dict(plan: "TrialPlan") -> dict[str, Any]:
    """JSON-serializable dict of a TrialPlan (benchmark/reproduce.json)."""
    return dataclasses.asdict(plan)
