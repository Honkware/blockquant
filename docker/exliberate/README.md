# exliberate

**exliberate — quant-aware abliteration, from the ExLlama lineage. Heretic proved abliteration can be automatic; exliberate makes it surgical AND ships it quantized.**

`exliberate MODEL` — fully automatic refusal removal for open-weight LLMs, with the
same one-command UX as [heretic](https://github.com/p-e-w/heretic), but with a
rank-k whitened refusal subspace, three aligned objectives, a re-probe loop, and a
written proof of superiority (`BENCHMARK.md` + `reproduce.json`) instead of vibes.

## Why

Heretic (p-e-w) automated the "refusal direction" ablation of Arditi et al. (2024)
behind a single CLI command and an Optuna search over a layer-wise ablation kernel.
It works — but it removes *one* direction, scores refusals with a 34-marker keyword
list, guards capabilities with *first-token* KL, optimizes two objectives, applies
the plan in a single pass, and ships no reproducible head-to-head evidence.

Exliberate keeps the UX and fixes the math, the objectives, and the proof:

| Feature | heretic | exliberate |
|---|---|---|
| Refusal geometry | rank-1 direction | **rank-k (1–4) Mahalanobis-whitened subspace** (benign-covariance protection) |
| Refusal scoring | 34 keyword markers | **improved keyword scorer** (regexes, lecture detection, compliance override) + optional LLM judge |
| Capability guard | first-token KL | **multi-token teacher-forced KL** + **loglik capability probe** |
| Objectives | 2 (refusal, KL) | **3** (refusal, KL, capability delta) |
| Search | TPE | **multivariate MOTPE + ASHA (SuccessiveHalving) pruning** |
| Application | single pass | **re-probe loop** (re-extracts residual direction, round-2 mop-up) |
| Overshoot | max_weight ≤ 1.5 | same, **interpreted as boundary-overshoot** λ > 1 |
| Evidence | none shipped | **`BENCHMARK.md` + `reproduce.json`** (paired bootstrap CIs, dataset hashes) |
| Exactness at export | lossy `svd_lowrank` path | **exact in-place norm-preserving ablation (NPBA)** |
| Quantization | none (fp16 export only) | **fused ablate+quantize (EXL3)**: ablation baked into the quantized weights (default), refusal-aware calibration, post-quant validation + auto-correction, optional sweepable fp16 residual LoRA |

## Method

1. **Subspace extraction.** Residual-stream activations at the first generated token
   are captured for packaged harmful/harmless contrast sets (winsorized). Per layer,
   the refusal direction is the mean difference; whitening by the Ledoit–Wolf
   covariance of *harmless* residuals turns removal into a Mahalanobis projection that
   avoids high benign-variance directions — the protected-subspace idea of SRA
   generalized to rank k (cross-validated: KL 2.088 → 0.044 at 0% refusals). Ranks
   k > 1 come from an SVD over pooled / half-split / prompt-mean contrast vectors,
   following the multi-direction evidence from concept cones (Wollschläger et al.,
   ICML 2025) and SOM (Piras et al., AAAI 2026) that refusal lives in a *subspace*,
   not a line (Arditi et al., 2024, is the rank-1 special case).
2. **Search.** Optuna MOTPE (`TPESampler(multivariate=True)`) minimizes
   `[keyword_refusal_rate, multi_token_KL, capability_delta]` over k ∈ {1..4},
   whitening, per-layer vs interpolated global direction, and a per-component
   (attention out-proj / MLP down-proj) tent-kernel weight schedule with
   `max_weight ∈ [0, 1.5]` — values above 1.0 *are* boundary overshoot
   (Piras, arXiv:2605.21706). ASHA prunes hopeless trials on a 32-prompt refusal
   subset at rung 1. Subspaces are extracted once per (k, whiten) combo, and every
   trial is an exact rank-k LoRA delta (`lora_A = VᵀW`, `lora_B = −wV`), so trial
   reset is a memset.
3. **Re-probe loop.** After the best Pareto plan (min scalarized
   `keyword + KL + capability`) is applied, the subspace is re-extracted *from the
   current model state*. If the residual direction norm exceeds 0.25× the original,
   a half-budget round-2 search mops up what remains. Max 2 rounds.
4. **Exact export.** The final plan is applied in place as
   `W' = W − w Σⱼ vⱼ(vⱼᵀW)` with rows renormalized to their original norms —
   grimjim's norm-preserving biprojected abliteration (NPBA) generalized to rank k.
5. **Proof.** `--benchmark-vs-heretic` builds a heretic-equivalent plan (k=1, no
   whitening, heretic's *own* keyword + first-token-KL objectives, same trial budget)
   and the exliberate plan on the same weights, evaluates both on **held-out**
   benchmark prompts with all scorers (heretic-compat metrics included, for
   apples-to-apples), computes paired-bootstrap 95% CIs (10k resamples, seeded) on
   per-prompt refusal differences, and writes `BENCHMARK.md` + `reproduce.json`
   (seed, both plans, dataset SHA-256s, package versions, timestamp).

### References

- Arditi et al., 2024. *Refusal in LLMs Is Mediated by a Single Direction.* arXiv:2406.11717
- grimjim, 2024. *Norm-Preserving Biprojected Abliteration (NPBA).* (LLM abliteration community)
- SRA / protected-subspace safety editing; safety residual space (ICML 2025)
- Wollschläger et al., ICML 2025. *Concept cones / the geometry of refusal.* arXiv:2502.17420
- Piras et al., AAAI 2026. *SOM: multi-direction refusal geometry.*
- Piras, 2026. *Boundary overshoot in refusal ablation.* arXiv:2605.21706
- Mazeika et al., 2024. *HarmBench.* arXiv:2402.04249 (optional judge backend)
- Röttger et al., 2023. *XSTest.* arXiv:2308.01263 (over-refusal evaluation)
- Zou et al., 2023. *Universal and Transferable Adversarial Attacks on Aligned LLMs (AdvBench).*
- p-e-w/heretic: https://github.com/p-e-w/heretic

## Install

```bash
pip install exliberate            # or: pip install git+https://github.com/zephyrlabs/exliberate
pip install "exliberate[quant]"   # + ExLlamaV3, for fused quantization (below)
```

Requires Python ≥ 3.12; CUDA recommended, CPU works (slowly).

## Fused quantization (EXL3)

Abliterating and *then* quantizing as two separate steps is silently broken:
the quantizer's error minimization reallocates weight error with no knowledge
of your surgery, and measured refusal rates swing back by 12–68pp after a
naive ablate→quant. Exliberate v0.2 makes ablation and quantization **one
aware pipeline** via ExLlamaV3 — to our knowledge the only tool that does
this.

**Fusion mode `baked` (default).** One self-contained EXL3 model — the
ablation is merged into the weights before quantization, nothing else to
load. Exliberate:

1. runs the normal abliteration search → best plan;
2. merges the ablation into the weights (exact norm-preserving NPBA export);
3. quantizes the merged model to EXL3 with **refusal-aware calibration** —
   ~15% of the converter's calibration rows are replaced by chat-formatted
   harmful/harmless prompts (+ short responses), so the Hessian-weighted
   quantizer spends its error budget on the activations where refusal lives
   (monkeypatches `exllamav3.conversion.convert_model.get_default_calibration`);
4. validates the actual EXL3 artifact post-quant (keyword refusal rate +
   first-token KL vs the ablated fp16 reference) and, if refusal rebound
   exceeds 5pp, **automatically strengthens the ablation** (`max_weight`
   ×1.15) **and re-quants** — up to 2 passes — so the file you ship is the
   file that was measured.

**Fusion mode `residual` (optional).** If you want a sweepable ablation
instead: our ablation is already an exact rank-k LoRA (`lora_A = VᵀW`,
`lora_B = −λ·w·V`), and ExLlamaV3 applies fp16 runtime LoRAs *after*
dequantizing each weight. exliberate quantizes the **base** model and writes
the ablation as a standard ExLlamaV3 LoRA adapter
(`adapter/adapter.safetensors` + `adapter_config.json`, loadable by vanilla
ExLlamaV3 via `LoRA.from_directory`) — quantization never touches the
ablation, so there is zero silent rebalance, and λ can be swept post-quant in
seconds (each candidate is a LoRA reload, not a re-quant). Trade-off: you must
always load the adapter alongside the model.

```bash
# One command: search → bake → quantize → validate → auto-correct.
exliberate meta-llama/Llama-3.1-8B-Instruct --quantize -o ./llama-3.1-8b-exl3
#   -> ./llama-3.1-8b-exl3/           EXL3 weights, ablation baked in (ExLlamaV3-ready)
#   -> ./llama-3.1-8b-exl3/fusion_report.json

exliberate MODEL --quantize --bits 4.25 --head-bits 6 \
    --work-dir /fast/scratch --cal-frac 0.15 -o OUT

exliberate MODEL --quantize --fusion residual \
    --lambda-sweep "0.9,1.0,1.1,1.25" -o OUT   # sweepable-adapter variant
```

**Requirements:** `pip install "exliberate[quant]"`, a CUDA GPU (CUDA ≥ 12.4
toolchain for ExLlamaV3's extension), ~8 GB VRAM to quantize an 8B (4–5 GB for
EXL3 inference), host RAM for the fp16 model. Without ExLlamaV3 or CUDA,
`--quantize` prints install instructions and falls back to the standard fp16
export — nothing crashes.

## Usage

```bash
# Fully automatic abliteration: optimize, re-probe, exact export, save.
exliberate google/gemma-3-4b-it --output-dir ./gemma-3-4b-it-exliberate

# Reproducible configuration.
exliberate MODEL --config my.toml --n-trials 60 --seed 1234

# Evaluate an already-abliterated model against its base (all scorers).
exliberate --evaluate-model ./gemma-exliberate --model google/gemma-3-4b-it

# Prove it: head-to-head vs a heretic-equivalent plan on held-out data.
exliberate --benchmark-vs-heretic --model google/gemma-3-4b-it --output-dir ./bench
#   -> ./bench/BENCHMARK.md, ./bench/reproduce.json, ./bench/exliberate/
```

All metrics are lower-is-better. Prompt sets ship with the package
(`exliberate/data/*.jsonl`); extraction/search sets are strictly disjoint from the
held-out benchmark sets.

## Limitations

- **Dense models only** (v1): `attn_o_proj` / `mlp_down_proj` module layout assumed.
- **No vLLM / multimodal / reasoning-model CoT handling** in v1.
- The LLM judge (HarmBench-style classifier) is **optional and config-driven**;
  the pipeline never depends on it.
- Re-probe round 2 returns the best plan of the *last* round (residual-steered
  re-search); explicit plan composition is future work.
- Abliterated models can still be re-aligned by downstream fine-tuning; this is a
  research tool for studying refusal geometry. Use responsibly and follow the base
  model's license.

## Roadmap

- MoE support (per-expert module maps)
- vLLM inference backend for the search loop
- Reasoning-model (CoT) refusal handling
- DPO "healing" pass after ablation
- Affine bias terms, SAE-feature banks, adapter export format, warm-starting
- 13B judge in the search loop

## License

AGPL-3.0. Datasets are generic evaluation fixtures written for this project
(AdvBench-style harmful requests, alpaca-style benign instructions, XSTest-style
seemingly-harmful prompts, commonsense loglik probes).
