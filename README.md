# BlockQuant

Discord bot and CLI for quantizing HuggingFace models to EXL3, publishing one repo per
bitrate with cross-linked model cards and a collection.

Quantization is [ExLlamaV3](https://github.com/turboderp-org/exllamav3). This repo covers
job queueing and approval, GPU provisioning, KL measurement against the unquantized model,
and publishing. Jobs run on a local GPU or a rented pod; the provider is a flag.

## Pipeline

```
request ──► approval ──► download ──► convert.py ──► measure ──► upload
                                        (per bpw)      │
                                                       ├─ KL vs unquantized (qbench)
                                                       └─ optional test prompt
```

Each bitrate is a separate repo, `{model}-exl3-{bpw}bpw`, and can be re-run on its own.
A quantized vision tower appends `-V{n}`; its absence means the tower was copied at fp16.
After upload, every card for the model is regenerated with the full quants table and added
to the collection.

## Usage

```bash
# local GPU
python -m blockquant.cli quant --model Qwen/Qwen3-8B --bpw 4.0

# rented pod: provisions, runs, syncs results, terminates
python backend/scripts/run_runpod_job.py \
    --model Qwen/Qwen3-8B --bpw 3.0,4.0,6.0 --hf-org yourorg
```

Slash commands:

| Command | Purpose |
| --- | --- |
| `/quant` | Request a quantization (admin approves) |
| `/stop` | Stop a job and terminate its pods |
| `/config` | (Admin) Set the role that may quant without approval |
| `/catbench` | Compare bitrates by SVG/Python cat drawings |
| `/queue` | Queue state |
| `/health` | Bot and service status |
| `/info` | Model metadata lookup |

A member holding the quanter role skips approval and may run one job at a time. Admins
stop any job; a quanter stops their own. Stopping terminates the pod before the
controller, since the reaper will not touch a pod younger than 15 minutes.

`/quant` options:

| Option | Values | Default |
| --- | --- | --- |
| `url` | HF model URL or ID | required |
| `bpw` | comma-separated bitrates, 1–8 | required |
| `prompt` | test prompt run on each finished quant | none |
| `subfolder` | directory holding the model | detected |
| `vision_bits` | 1–8, or 16 to copy the tower unquantized | the arch decides |
| `head_bits` | 1–8, or 16 to leave the head unquantized | 6 |
| `codebook` | `auto`, `mul1`, `mcg`, `3inst` | `auto` |

Unset means exllamav3 decides, rather than a copy of its default pinned here. `head_bits`
is 6. `vision_bits` is the tower's own `default_vision_bits`, which is 6 on the
architectures that declare their tower validated — Qwen3-VL, Gemma4, GLM4V, Step3,
DeepSeek-V4-Vision, MuseGlimmer — and 16 everywhere else, so most towers are copied whole.
`codebook: auto` picks `mul1` for dense and `mcg` for MoE, whose fused expert kernel was
`mcg`-only on older images.

`subfolder` is for repos that ship several formats side by side (`BF16/`, `FP8/`,
`GGUF/`). Pre-flight picks the one unquantized directory whose architecture exllamav3
knows, and only that subtree is downloaded. An already-quantized directory is never
chosen: it has been rounded once, and quantizing it again compounds that.

## Providers

`local` runs against the GPUs on the host. `runpod` rents one per job.

Pod selection orders candidates by model size: cheapest-first under 25 GB, capable-first
above. Datacenter Blackwell (B200/B300), AMD and low-power inference
cards are excluded. Pods are terminated on
every exit path and swept by name prefix. Downloads run under a stall watchdog.

## Image

`docker/Dockerfile.runpod` — one image for every supported architecture, listed in
`backend/arch_support.json`.

- exllamav3 pinned to a release commit (`EXLLAMAV3_REF`); bump behind an end-to-end run,
  and bump `selfcal/VENDOR_MANIFEST.json` with it or the vendor test fails
- Python 3.12 — fla's Triton kernels fail to import on 3.11
- the CUDA extension is compiled at build time, not JITed on the pod's first boot
- flash-attn not installed; exllamav3's Triton attention handles sliding window
- wikitext-2 is warmed into the dataset cache so a paid pod cannot lose its KL number
  to a Hugging Face outage

## Measurement

KL is measured with [qbench](https://github.com/turboderp-org/exllamav3/tree/master/eval/qbench),
vendored under `backend/src/blockquant/selfcal/eval/qbench/`, so the number can be read
against turboderp's and ezexl3's rather than being ours alone. wiki2 at 10×2048, which is
what qbench's own example project uses and the only corpus `sc_measure` implements.

Cards quote the **median**, not the mean. Per qbench's own note, the mean is dominated by
tokens where the reference itself is undecided, where any perturbation is amplified; the
median and the high-confidence buckets are what isolate quantization damage. The full
spread and the buckets go in each repo's `bq_quality.json`.

Two passes, one model resident at a time: the unquantized model writes reference logits to
disk, then the quant is streamed against them. That is qbench's arrangement and it is what
lets a 35B reference fit on an 80 GB card.

wiki2's test split is not what EXL3 calibrates on — that is `standard_cal_data/wiki.utf8`,
a general Wikipedia dump. 393 sampled 12-word shingles from the test split appear in it
zero times.

## Layout

```
src/                        Discord bot: commands, queue, embeds
backend/src/blockquant/     download → quantize → verify → quality → report → upload
  providers/                local, runpod
  remote/quant.py           runs on the pod
  selfcal/                  vendored exllamav3 self-calibration + qbench (see its manifest)
backend/scripts/            run_runpod_job.py, publish_quant.py
docker/                     RunPod image
```

## Requirements

Python 3.10+ and Node 20+. A CUDA GPU for local runs, or `RUNPOD_API_KEY` for pods. An HF
token with write access goes in `.env`.
