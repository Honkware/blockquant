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
                                                       ├─ KL vs unquantized
                                                       └─ optional test prompt
```

Each bitrate is a separate repo, `{model}-exl3-{bpw}bpw`, and can be re-run on its own.
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
| `/catbench` | Compare bitrates by SVG/Python cat drawings |
| `/queue` | Queue state |
| `/health` | Bot and service status |
| `/info` | Model metadata lookup |

`/quant` options:

| Option | Values | Default |
| --- | --- | --- |
| `url` | HF model URL or ID | required |
| `bpw` | comma-separated bitrates, 1–8 | required |
| `prompt` | test prompt run on each finished quant | none |
| `codebook` | `mul1`, `mcg`, `3inst` | `mul1` |
| `vision` | `auto`, `fp16`, `6` | `auto` |

`vision` sets the tower bitrate on multimodal models. `auto` defers to exllamav3, which
uses 6 bpw where the architecture declares the tower validated and fp16 elsewhere. `fp16`
copies the tower unquantized.

## Providers

`local` runs against the GPUs on the host. `runpod` rents one per job.

Pod selection orders candidates by model size: cheapest-first under 25 GB, capable-first
above. Datacenter Blackwell (B200/B300), AMD and low-power inference
cards are excluded. Pods are terminated on
every exit path and swept by name prefix. Downloads run under a stall watchdog.

## Image

`docker/Dockerfile.runpod` — one image for every supported architecture, listed in
`backend/arch_support.json`.

- exllamav3 pinned to a release commit (`EXLLAMAV3_REF`); bump behind an end-to-end run
- Python 3.12 — fla's Triton kernels fail to import on 3.11
- built with `EXLLAMA_NOCOMPILE=1`; the CUDA extension builds on the pod's first boot
- flash-attn not installed

## Layout

```
src/                        Discord bot: commands, queue, embeds
backend/src/blockquant/     download → quantize → verify → quality → report → upload
  providers/                local, runpod
  remote/quant.py           runs on the pod
backend/scripts/            run_runpod_job.py, publish_quant.py
docker/                     RunPod image
```

## Requirements

Python 3.10+ and Node 20+. A CUDA GPU for local runs, or `RUNPOD_API_KEY` for pods. An HF
token with write access goes in `.env`.
