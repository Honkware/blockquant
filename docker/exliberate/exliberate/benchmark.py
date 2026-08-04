"""Head-to-head benchmark: exliberate vs a heretic-equivalent plan (SPEC §6).

Builds (a) a heretic-equivalent plan (k=1, no whitening, heretic's own
objectives: exact keyword rate + first-token KL) and (b) the full exliberate
3-objective plan on the SAME model, evaluates both on HELD-OUT benchmark
prompt sets with every scorer, computes paired-bootstrap 95% CIs on
per-prompt refusal differences, and writes BENCHMARK.md + reproduce.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .optimize import (
    COMPONENTS,
    plan_to_dict,
    precompute_subspaces,
    run_study,
    seed_everything,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Settings
    from .model import ModelBackend
    from .prompts import Prompt
    from .scorers.base import Score
    from .surgery import TrialPlan

#: Bootstrap resamples for the paired CI (SPEC §6: 10k).
BOOTSTRAP_RESAMPLES = 10_000

DATA_FILES = (
    "harmful_extraction",
    "harmless_extraction",
    "harmful_benchmark",
    "benign_benchmark",
    "overrefusal",
    "capability_probe",
)


@dataclass
class BenchmarkResult:
    heretic_plan: "TrialPlan"
    exliberate_plan: "TrialPlan"
    metrics: dict[str, dict[str, float]]  # metric -> {"base","heretic","exliberate"}
    refusal_ci: tuple[float, float]       # 95% CI of per-prompt refusal diff
    refusal_diff: float                   # exliberate - heretic mean diff
    output_dir: Path
    extra: dict[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dataset_hashes() -> dict[str, str]:
    from .prompts import packaged

    return {name: _sha256(packaged(name)) for name in DATA_FILES}


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        from . import __version__

        versions["exliberate"] = __version__
    except Exception:  # namespace package before core merge
        versions["exliberate"] = "0.1.0"
    for mod in ("torch", "transformers", "peft", "optuna", "numpy", "sklearn"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[mod] = "not-installed"
    return versions


def _evaluate(
    model: "ModelBackend",
    plan: "TrialPlan | None",
    subspaces: Mapping[tuple[int, bool], Any],
    scorers: Mapping[str, Any],
) -> dict[str, float]:
    """Score ``scorers`` with ``plan`` applied (None = base model)."""
    from .surgery import plan_to_adapters

    model.zero_adapters()
    if plan is not None:
        plan_to_adapters(model, subspaces[(plan.k, plan.whiten)], plan)
    out: dict[str, float] = {}
    for name, scorer in scorers.items():
        out[name] = float(scorer.score(model).value)
    model.zero_adapters()
    return out


def _per_prompt_refusals(
    model: "ModelBackend",
    plan: "TrialPlan | None",
    subspaces: Mapping[tuple[int, bool], Any],
    scorer: Any,
    prompts: Sequence["Prompt"],
) -> np.ndarray:
    """Per-prompt binary refusal vector via single-prompt scorer instances."""
    from .surgery import plan_to_adapters

    model.zero_adapters()
    if plan is not None:
        plan_to_adapters(model, subspaces[(plan.k, plan.whiten)], plan)
    values = []
    for p in prompts:
        single = type(scorer)([p])
        values.append(float(single.score(model).value))
    model.zero_adapters()
    return np.asarray(values, dtype=np.float64)


def paired_bootstrap_ci(
    diffs: np.ndarray,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI on per-prompt differences (SPEC §6).

    Resamples prompts with replacement (numpy Generator(seed)); returns
    (mean_diff, ci_low, ci_high).
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = diffs[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(low), float(high)


def _verdict(metrics: dict[str, dict[str, float]], diff: float, ci: tuple[float, float]) -> str:
    lines = []
    lo, hi = ci
    if hi < 0:
        lines.append(
            f"Exliberate's held-out refusal rate is significantly LOWER than the "
            f"heretic-equivalent plan (paired bootstrap mean difference {diff:+.3f}, "
            f"95% CI [{lo:+.3f}, {hi:+.3f}]; CI entirely below zero)."
        )
    elif lo > 0:
        lines.append(
            f"The heretic-equivalent plan has a significantly lower held-out refusal "
            f"rate (paired bootstrap mean difference {diff:+.3f}, 95% CI [{lo:+.3f}, "
            f"{hi:+.3f}])."
        )
    else:
        lines.append(
            f"Held-out refusal rates are statistically indistinguishable (paired "
            f"bootstrap mean difference {diff:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}])."
        )
    for metric in ("multi_token_kl", "capability_delta"):
        if metric in metrics:
            h, i = metrics[metric]["heretic"], metrics[metric]["exliberate"]
            better = "exliberate" if i < h else "heretic-equivalent"
            lines.append(
                f"On {metric}, {better} is better (heretic {h:.4f} vs exliberate {i:.4f}; "
                f"lower is better)."
            )
    return " ".join(lines)


def head_to_head(
    model_settings: "Settings",
    n_trials: int | None = None,
    *,
    output_dir: str | Path = ".",
    limit: int | None = None,
    k_values: Sequence[int] = (1, 2, 3, 4),
) -> BenchmarkResult:
    """Run the SPEC §6 head-to-head and write BENCHMARK.md + reproduce.json.

    ``limit`` truncates every prompt set (CI/smoke tests); ``k_values``
    restricts the exliberate subspace rank search space.
    """
    from .model import ModelBackend
    from .prompts import load_prompts, packaged
    from .scorers import heretic_compat, keyword, kl, capability, overrefusal

    settings = model_settings
    seed = int(getattr(settings, "seed", 0))
    n = int(n_trials if n_trials is not None else getattr(settings, "n_trials", 40))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    model = ModelBackend(settings)
    model.setup_adapters(max(k_values))

    # --- data (extraction/search sets vs HELD-OUT benchmark sets) ---
    harmful_ex = load_prompts(packaged("harmful_extraction"), limit=limit)
    harmless_ex = load_prompts(packaged("harmless_extraction"), limit=limit)
    harmful_bench = load_prompts(packaged("harmful_benchmark"), limit=limit)
    benign_bench = load_prompts(packaged("benign_benchmark"), limit=limit)
    over_bench = load_prompts(packaged("overrefusal"), limit=limit)

    subspaces = precompute_subspaces(model, harmful_ex, harmless_ex, k_values)

    # Continuations for KL scorers come from the BASE model (adapters zero).
    model.zero_adapters()
    ex_continuations = model.generate(harmless_ex, max_new_tokens=64)
    bench_continuations = model.generate(benign_bench, max_new_tokens=64)

    # --- (a) heretic-equivalent arm: k=1, whiten off, heretic's objectives ---
    # Base-reference scorers capture the BASE model at init (adapters zeroed
    # above for continuation generation); never lazily mid-study, which would
    # capture a trial-perturbed state as the "base" reference.
    heretic_scorers = {
        "keyword": heretic_compat.HereticKeywordRate(harmful_ex),
        "kl": heretic_compat.FirstTokenKL(harmless_ex, base_model=model),
    }
    heretic_result = run_study(
        model,
        subspaces,
        heretic_scorers,
        settings,
        objective_keys=("keyword", "kl"),
        fixed_params={"k": 1, "whiten": False},
        k_values=(1,),
        n_trials=n,
    )

    # --- (b) exliberate arm: full 3-objective search ---
    model.zero_adapters()  # base state before capturing scorer references
    exliberate_scorers = {
        "keyword": keyword.KeywordScorer(harmful_ex),
        "kl": kl.MultiTokenKL(harmless_ex, ex_continuations, base_model=model),
        "capability": capability.CapabilityProbe(
            packaged("capability_probe"), base_model=model
        ),
    }
    exliberate_result = run_study(
        model, subspaces, exliberate_scorers, settings, k_values=k_values, n_trials=n
    )

    heretic_plan = heretic_result.plan
    exliberate_plan = exliberate_result.plan

    # --- (c) held-out evaluation with all scorers (constructed at base state) ---
    model.zero_adapters()
    eval_scorers = {
        "heretic_keyword_rate": heretic_compat.HereticKeywordRate(harmful_bench),
        "heretic_first_token_kl": heretic_compat.FirstTokenKL(
            benign_bench, base_model=model
        ),
        "improved_keyword_rate": keyword.KeywordScorer(harmful_bench),
        "multi_token_kl": kl.MultiTokenKL(
            benign_bench, bench_continuations, base_model=model
        ),
        "capability_delta": capability.CapabilityProbe(
            packaged("capability_probe"), base_model=model
        ),
        "overrefusal_rate": overrefusal.OverRefusalScorer(over_bench),
    }
    metrics: dict[str, dict[str, float]] = {}
    for label, plan in (
        ("base", None),
        ("heretic", heretic_plan),
        ("exliberate", exliberate_plan),
    ):
        col = _evaluate(model, plan, subspaces, eval_scorers)
        for name, value in col.items():
            metrics.setdefault(name, {})[label] = value

    # --- (d) paired bootstrap CI on per-prompt refusal differences ---
    ref_base_kw = eval_scorers["improved_keyword_rate"]
    ref_heretic = _per_prompt_refusals(model, heretic_plan, subspaces, ref_base_kw, harmful_bench)
    ref_exliberate = _per_prompt_refusals(
        model, exliberate_plan, subspaces, ref_base_kw, harmful_bench
    )
    diff, lo, hi = paired_bootstrap_ci(ref_exliberate - ref_heretic, seed)

    result = BenchmarkResult(
        heretic_plan=heretic_plan,
        exliberate_plan=exliberate_plan,
        metrics=metrics,
        refusal_ci=(lo, hi),
        refusal_diff=diff,
        output_dir=output_dir,
        extra={
            "n_trials": n,
            "k_values": [int(k) for k in k_values],
            "limit": limit,
            "objective_values": {
                "heretic": list(heretic_result.objective_values),
                "exliberate": list(exliberate_result.objective_values),
            },
        },
    )

    # --- (e) artifacts ---
    # Save the merged exliberate model artifact (exact NPBA export). A
    # constrained filesystem (e.g. a mount rejecting large files) must not
    # sacrifice the completed benchmark: reproduce.json fully determines the
    # merged model via seed + plan + dataset hashes.
    from .surgery import export_exact

    model.zero_adapters()
    export_exact(model, subspaces[(exliberate_plan.k, exliberate_plan.whiten)], exliberate_plan)
    artifact_dir = output_dir / "exliberate"
    save_error: str | None = None
    try:
        model.save_merged(str(artifact_dir))
    except Exception as exc:  # noqa: BLE001 - robustness, results already computed
        import shutil

        save_error = f"{type(exc).__name__}: {exc}"
        shutil.rmtree(artifact_dir, ignore_errors=True)  # drop partial shards
    result.extra["model_artifact_error"] = save_error

    write_benchmark_md(result, output_dir / "BENCHMARK.md")
    write_reproduce_json(result, settings, output_dir / "reproduce.json")
    return result


def write_benchmark_md(result: BenchmarkResult, path: Path) -> None:
    metric_rows = []
    for name, cols in result.metrics.items():
        metric_rows.append(
            f"| {name} | {cols['base']:.4f} | {cols['heretic']:.4f} | {cols['exliberate']:.4f} |"
        )
    lo, hi = result.refusal_ci
    base_refusal = result.metrics.get("heretic_keyword_rate", {}).get("base", 1.0)
    scope_note = ""
    if base_refusal < 0.05:
        scope_note = """
## Scope (read before interpreting)

The base model's held-out refusal rate is ~0: this base model essentially
never refuses, so there is no refusal behavior to remove and the
refusal-rate rows above are a floor effect. This run validates the
*mechanism* end-to-end (subspace extraction, MOTPE search, exact surgery,
scoring, bootstrap CIs) — it is a smoke test, NOT evidence about
abliteration *quality*. Quality claims require a model that actually
refuses (run on capable instruct models, e.g. on a GPU).
"""
    text = f"""# BENCHMARK.md — exliberate vs heretic-equivalent (head-to-head)

Same base model, same trial budget ({result.extra['n_trials']} trials per arm), evaluated on
HELD-OUT benchmark prompt sets never used for extraction or search. All metrics are
lower-is-better. The heretic-equivalent arm uses k=1, no whitening, and heretic's own
objectives (exact keyword rate + first-token KL) via our search engine.

| Metric (lower = better) | Base model | Heretic-equivalent | Exliberate |
|---|---|---|---|
{chr(10).join(metric_rows)}

## Paired bootstrap on per-prompt refusal differences

- Metric: improved keyword refusal, per-prompt, exliberate − heretic-equivalent
- Mean difference: {result.refusal_diff:+.4f}
- 95% CI ({BOOTSTRAP_RESAMPLES} resamples, seeded): [{lo:+.4f}, {hi:+.4f}]

## Verdict

{_verdict(result.metrics, result.refusal_diff, result.refusal_ci)}
{scope_note}
See `reproduce.json` for seed, plans, dataset hashes, and package versions.
"""
    path.write_text(text)


def write_reproduce_json(result: BenchmarkResult, settings: "Settings", path: Path) -> None:
    payload = {
        "seed": int(getattr(settings, "seed", 0)),
        "model": getattr(settings, "model_id", None) or getattr(settings, "model", None),
        "n_trials": result.extra["n_trials"],
        "k_values": result.extra["k_values"],
        "prompt_limit": result.extra["limit"],
        "plans": {
            "heretic_equivalent": plan_to_dict(result.heretic_plan),
            "exliberate": plan_to_dict(result.exliberate_plan),
        },
        "objective_values": result.extra["objective_values"],
        "refusal_diff": result.refusal_diff,
        "refusal_ci_95": list(result.refusal_ci),
        "model_artifact_error": result.extra.get("model_artifact_error"),
        "dataset_sha256": dataset_hashes(),
        "versions": _versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
