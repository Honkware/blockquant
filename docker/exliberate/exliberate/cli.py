"""``exliberate`` command-line interface (SPEC §6).

    exliberate MODEL [--config x.toml] [--n-trials N] [--output-dir DIR]
    exliberate --evaluate-model M --model BASE
    exliberate --benchmark-vs-heretic --model M [--output-dir DIR]

Core modules (config/model/scorers/surgery) are imported lazily so this file
stays importable before the core/scorers branches are merged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Settings

DEFAULT_OUTPUT_DIR = "exliberate-out"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="exliberate",
        description="Fully automatic, surgical refusal removal for open-weight LLMs.",
    )
    p.add_argument("model", nargs="?", default=None, help="base model id or local path")
    p.add_argument("--model", dest="model_flag", default=None,
                   help="base model id/path (alternative to positional MODEL)")
    p.add_argument("--config", default=None, help="path to a TOML settings file")
    p.add_argument("--n-trials", type=int, default=None, help="optuna trials (default 40)")
    p.add_argument("--seed", type=int, default=None, help="determinism seed")
    p.add_argument("--output-dir", "-o", default=None,
                   help="where to save the abliterated model (or the EXL3 model "
                        "with --quantize)")
    p.add_argument("--quantize", action="store_true",
                   help="fused ablate+quantize via ExLlamaV3: after the search, "
                        "quantize to EXL3 with refusal-aware calibration")
    p.add_argument("--bits", type=float, default=4.25,
                   help="EXL3 average bits per weight (default 4.25)")
    p.add_argument("--head-bits", type=int, default=6,
                   help="EXL3 lm_head bits (default 6)")
    p.add_argument("--fusion", choices=("baked", "residual"), default="baked",
                   help="baked: ablation merged into the EXL3 weights (default, "
                        "single self-contained model, rebound-triggered re-quant); "
                        "residual: ablation as an fp16 runtime LoRA on the EXL3 "
                        "model (sweepable post-quant)")
    p.add_argument("--work-dir", default=None,
                   help="quantizer scratch dir (default <output-dir>-work)")
    p.add_argument("--cal-frac", type=float, default=0.15,
                   help="fraction of calibration rows replaced by "
                        "harmful/harmless prompt rows (default 0.15)")
    p.add_argument("--lambda-sweep", default="0.9,1.0,1.1,1.25",
                   help="comma-separated λ multipliers swept on the fp16 "
                        "ablation residual post-quant")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow models that ship custom modeling code.")
    p.add_argument("--limit", type=int, default=None,
                   help="truncate every prompt set to N items (smoke/CI runs; "
                        "extraction/benchmark disjointness is preserved)")
    p.add_argument("--evaluate-model", default=None, metavar="M",
                   help="run all scorers on M vs --model BASE and print a table")
    p.add_argument("--benchmark-vs-heretic", action="store_true",
                   help="head-to-head vs a heretic-equivalent plan; writes BENCHMARK.md "
                        "and reproduce.json into --output-dir")
    return p


def _load_settings(args: argparse.Namespace) -> "Settings":
    """Load Settings from TOML (config module, SPEC §3) + apply CLI overrides."""
    from .config import Settings

    if args.config:
        # pydantic-settings resolves the TOML path from model_config at class
        # definition time; a per-path subclass is the supported override.
        from pydantic_settings import SettingsConfigDict

        class _TomlSettings(Settings):
            model_config = SettingsConfigDict(
                env_prefix="EXLIBERATE_", toml_file=args.config, extra="ignore"
            )

        settings = _TomlSettings()
    else:
        settings = Settings()

    overrides: dict[str, Any] = {}
    model_id = args.model_flag or args.model
    if model_id:
        overrides["model_id"] = model_id
    if args.n_trials is not None:
        overrides["n_trials"] = args.n_trials
    if args.seed is not None:
        overrides["seed"] = args.seed
    if getattr(args, "trust_remote_code", False):
        overrides["trust_remote_code"] = True
    if overrides:
        if hasattr(settings, "model_copy"):  # pydantic v2
            settings = settings.model_copy(update=overrides)
        else:  # plain dataclass / namespace fallback
            for k, v in overrides.items():
                setattr(settings, k, v)
    return settings


def _print_metrics_table(console, title: str, metrics: dict[str, dict[str, float]]) -> None:
    from rich.table import Table

    columns: list[str] = []
    for cols in metrics.values():
        for c in cols:
            if c not in columns:
                columns.append(c)
    table = Table(title=title)
    table.add_column("Metric (lower = better)")
    for c in columns:
        table.add_column(c.capitalize(), justify="right")
    for name, cols in metrics.items():
        table.add_row(name, *[f"{cols.get(c, float('nan')):.4f}" for c in columns])
    console.print(table)


def _print_plan(console, title: str, plan: Any) -> None:
    from rich.table import Table

    table = Table(title=title)
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("k", str(plan.k))
    table.add_row("whiten", str(plan.whiten))
    table.add_row("direction_index",
                  "per-layer" if plan.direction_index is None else f"{plan.direction_index:.2f}")
    for comp, prm in plan.params.items():
        table.add_row(
            comp,
            f"max_w={prm.max_weight:.3f} @ layer {prm.max_weight_position:.2f} "
            f"min_w={prm.min_weight:.3f} dist={prm.min_weight_distance:.2f}",
        )
    console.print(table)


def _cmd_default(args: argparse.Namespace, settings: "Settings") -> int:
    """Default run: optimize -> re-probe -> exact export -> save + summary."""
    from rich.console import Console

    from .model import ModelBackend
    from .optimize import precompute_subspaces, reprobe_loop, run_study
    from .prompts import load_prompts, packaged
    from .scorers import capability, keyword, kl
    from .surgery import export_exact

    console = Console()
    if not getattr(settings, "model_id", None):
        console.print("[red]error:[/red] no model given (positional MODEL or --model).")
        return 2

    console.print(f"[bold]exliberate[/bold] loading {settings.model_id} ...")
    model = ModelBackend(settings)
    model.setup_adapters(4)

    limit = getattr(args, "limit", None)
    harmful = load_prompts(packaged("harmful_extraction"), limit=limit)
    harmless = load_prompts(packaged("harmless_extraction"), limit=limit)
    console.print(f"extracting refusal subspaces ({len(harmful)} harmful / "
                  f"{len(harmless)} harmless prompts, 8 k×whiten combos) ...")
    subspaces = precompute_subspaces(model, harmful, harmless)

    model.zero_adapters()
    continuations = model.generate(harmless, max_new_tokens=64)
    # Base-reference scorers capture the base model NOW (adapters zeroed).
    scorers = {
        "keyword": keyword.KeywordScorer(harmful),
        "kl": kl.MultiTokenKL(harmless, continuations, base_model=model),
        "capability": capability.CapabilityProbe(
            packaged("capability_probe"), base_model=model
        ),
    }

    n_trials = int(getattr(settings, "n_trials", 40))
    console.print(f"searching ({n_trials} trials, MOTPE + ASHA) ...")
    result = run_study(model, subspaces, scorers, settings)
    _print_plan(console, "Best Pareto plan (round 1)", result.plan)
    console.print(
        f"objectives [keyword, KL, capability] = "
        f"{[round(v, 4) for v in result.objective_values]} "
        f"(scalarized {result.scalarized:.4f})"
    )

    reprobe = reprobe_loop(
        model, result.plan, subspaces, scorers, settings, harmful, harmless
    )
    console.print(
        f"re-probe: {reprobe.rounds} round(s), residual ratios "
        f"{[round(r, 3) for r in reprobe.residual_ratios]}"
    )
    final_plan = reprobe.final_plan
    final_subspaces = reprobe.subspaces

    if args.quantize:
        if _try_fusion(args, settings, console, model, final_plan, final_subspaces):
            return 0
        # QuantUnavailable: message printed; fall through to the fp16 export
        # so the completed search is not wasted.

    model.zero_adapters()
    export_exact(model, final_subspaces[(final_plan.k, final_plan.whiten)], final_plan)

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_merged(output_dir)
    console.print(f"[green]saved[/green] abliterated model -> {output_dir}")
    _print_plan(console, "Final plan (after re-probe)", final_plan)
    return 0


def _try_fusion(
    args: argparse.Namespace,
    settings: "Settings",
    console: Any,
    model: Any,
    plan: Any,
    subspaces: dict[Any, Any],
) -> bool:
    """--quantize: run the fused ablate+quantize pipeline (SPEC ADDENDUM §D).

    Returns True when the EXL3 artifact + report were written; False when
    ExLlamaV3/CUDA are unavailable (caller falls back to the fp16 export).
    """
    from .quant import FusionSettings, QuantUnavailable, run_fusion_pipeline

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    lambda_sweep = tuple(
        float(x) for x in str(args.lambda_sweep).split(",") if x.strip()
    )
    fusion_settings = FusionSettings(
        model_dir=settings.model_id,
        out_dir=output_dir,
        work_dir=args.work_dir,
        bits=args.bits,
        head_bits=args.head_bits,
        cal_frac=args.cal_frac,
        cal_seed=int(getattr(settings, "seed", 0)),
        lambda_sweep=lambda_sweep,
        fusion=args.fusion,
        max_new_tokens=int(getattr(settings, "max_new_tokens", 64)),
    )
    console.print(
        f"[bold]fused quantization[/bold] ({args.fusion}, {args.bits} bpw, "
        f"head {args.head_bits} bits) -> {output_dir}"
    )
    try:
        report = run_fusion_pipeline(
            fusion_settings,
            plan,
            fusion=args.fusion,
            backend=model,
            subspace=subspaces[(plan.k, plan.whiten)],
        )
    except QuantUnavailable as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        console.print(
            "[yellow]falling back to the standard fp16 export; re-run with "
            "--quantize on a CUDA machine with exllamav3 installed.[/yellow]"
        )
        return False
    console.print(
        f"[green]EXL3 model[/green] -> {report['exl3_dir']}  "
        f"(rebound {report['rebound_pp']:+.1f}pp, "
        f"KL {report['kl_post_vs_pre']:.4f}, "
        f"passed={report['passed']})"
    )
    if args.fusion == "residual":
        console.print(
            f"[green]ablation adapter[/green] -> {report['adapter_dir']} "
            f"(lambda={report['lambda_']})"
        )
    else:
        console.print(
            f"[green]baked[/green] iterations={report['iterations']} "
            f"max_weight={report['max_weight']}"
        )
    console.print(f"[green]report[/green] -> {output_dir}/fusion_report.json")
    return True


def _cmd_evaluate(args: argparse.Namespace, settings: "Settings") -> int:
    """--evaluate-model M --model BASE: all scorers on M vs BASE."""
    from rich.console import Console

    from .model import ModelBackend
    from .prompts import load_prompts, packaged
    from .scorers import capability, keyword, kl, overrefusal

    console = Console()
    base_id = args.model_flag or args.model
    if not base_id or not args.evaluate_model:
        console.print("[red]error:[/red] --evaluate-model M requires a base model "
                      "(positional MODEL or --model BASE).")
        return 2

    base_settings = settings.model_copy(update={"model_id": base_id}) \
        if hasattr(settings, "model_copy") else settings
    eval_settings = settings.model_copy(update={"model_id": args.evaluate_model}) \
        if hasattr(settings, "model_copy") else settings
    if not hasattr(settings, "model_copy"):
        import copy

        eval_settings = copy.copy(settings)
        eval_settings.model_id = args.evaluate_model

    console.print(f"loading BASE {base_id} ...")
    base_model = ModelBackend(base_settings)
    console.print(f"loading EVAL {args.evaluate_model} ...")
    eval_model = ModelBackend(eval_settings)

    harmful = load_prompts(packaged("harmful_benchmark"))
    benign = load_prompts(packaged("benign_benchmark"))
    over = load_prompts(packaged("overrefusal"))
    continuations = base_model.generate(benign, max_new_tokens=64)

    # Scorers cache their BASE reference at init -> capture against BASE.
    scorers = {
        "keyword_refusal": keyword.KeywordScorer(harmful),
        "multi_token_kl": kl.MultiTokenKL(benign, continuations, base_model=base_model),
        "capability_delta": capability.CapabilityProbe(
            packaged("capability_probe"), base_model=base_model
        ),
        "overrefusal": overrefusal.OverRefusalScorer(over),
    }
    # Optional judge (SPEC §4): config-driven, never breaks the pipeline.
    judge_id = getattr(settings, "judge_model", None)
    if judge_id:
        try:
            from .scorers import judge

            scorers["judge_refusal"] = judge.JudgeScorer(harmful, model_id=judge_id)
        except Exception as exc:  # JudgeUnavailable or load failure
            console.print(f"[yellow]judge unavailable ({exc}); skipping[/yellow]")

    metrics: dict[str, dict[str, float]] = {}
    for label, m in (("base", base_model), ("eval", eval_model)):
        for name, scorer in scorers.items():
            metrics.setdefault(name, {})[label] = float(scorer.score(m).value)
    _print_metrics_table(console, f"Evaluation: {args.evaluate_model} vs {base_id}", metrics)
    return 0


def _cmd_benchmark(args: argparse.Namespace, settings: "Settings") -> int:
    """--benchmark-vs-heretic --model M: head-to-head + artifacts."""
    from rich.console import Console

    from .benchmark import head_to_head
    from .optimize import plan_to_dict

    console = Console()
    if not getattr(settings, "model_id", None):
        console.print("[red]error:[/red] --benchmark-vs-heretic requires --model M.")
        return 2

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    result = head_to_head(
        settings,
        n_trials=args.n_trials,
        output_dir=output_dir,
        limit=args.limit,
    )
    _print_metrics_table(console, "Head-to-head (held-out)", result.metrics)
    lo, hi = result.refusal_ci
    console.print(
        f"paired bootstrap refusal diff (exliberate − heretic): {result.refusal_diff:+.4f} "
        f"95% CI [{lo:+.4f}, {hi:+.4f}]"
    )
    _print_plan(console, "Heretic-equivalent plan", result.heretic_plan)
    _print_plan(console, "Exliberate plan", result.exliberate_plan)
    console.print(f"[green]artifacts[/green] -> {output_dir}/BENCHMARK.md, "
                  f"{output_dir}/reproduce.json")
    save_error = result.extra.get("model_artifact_error")
    if save_error:
        console.print(f"[yellow]merged model artifact not saved ({save_error}); "
                      f"the plan in reproduce.json fully determines it[/yellow]")
    else:
        console.print(f"[green]merged model[/green] -> {output_dir}/exliberate/")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _load_settings(args)
    if args.evaluate_model:
        return _cmd_evaluate(args, settings)
    if args.benchmark_vs_heretic:
        return _cmd_benchmark(args, settings)
    return _cmd_default(args, settings)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
