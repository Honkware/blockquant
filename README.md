# BlockQuant

A Discord bot that quantizes Hugging Face models to EXL3 on rented RunPod GPUs and publishes one repo per bitrate.

The conversion itself is [ExLlamaV3](https://github.com/turboderp-org/exllamav3). This repo is everything around it: approval, picking and renting the pod, self-calibration, KL measurement against the unquantized model, and the model cards.

## Running it

```bash
cp .env.example .env          # BOT_TOKEN, CLIENT_ID, GUILD_ID, HF_TOKEN, RUNPOD_API_KEY, ADMIN_IDS, RUNPOD_IMAGE
npm ci
python3 -m venv backend/venv  # the bot runs backend/venv/bin/python unless PYTHON_BIN says otherwise
backend/venv/bin/pip install -r backend/requirements.txt
npm start
```

Node 20+, Python 3.10+. Pods need `~/.ssh/id_rsa{,.pub}` on the bot's host; the public key is handed to each pod.

## /quant

```
/quant url:Qwen/Qwen3-8B bpw:4.0
/quant url:ukisai/Swift-Qwen3.8-27b bpw:low sc:True head_bits:5
```

| Option | What it does |
| --- | --- |
| `url` | HF model URL or `org/name` |
| `bpw` | Comma-separated, 1-8, or a preset: `low` (2.0-3.5), `mid` (4.0-5.0), `high` (6.0, 8.0), `ladder` (2.0-6.0) |
| `sc` | Self-calibrated: measure the model and convert from a per-tensor recipe. Needs `head_bits` |
| `sc_mode` | `full` (default) or `trace`: calibrate on the model's own output with `-hq 2` and skip the measurement. Implies `sc` |
| `head_bits` | 1-8, or 16 for an unquantized head. exllamav3's default (6) when unset |
| `vision_bits` | 1-8, or 16 to copy the tower. Unset lets the arch decide: 6 where exllamav3 marks the tower validated, 16 elsewhere |
| `mtp_bits`, `ngram_bits`, `hq` | Passed to convert as `-mb`, `-ngb`, `-hq` |
| `codebook` | `auto` (mul1 for dense, mcg for MoE), `mul1`, `mcg`, `3inst` |
| `gpu_count` | Cards per pod. Two convert about 1.7x faster for 15-20% more per job |
| `subfolder` | For repos that ship `BF16/`, `FP8/`, `GGUF/` side by side. Picked automatically when only one is usable |
| `draft_model` | A drafter (e.g. a DFlash2 repo) to quantize at 4bpw on the same pod |
| `prompt` | Run on each finished quant; the reply shows in the thread |

Requests wait for an admin to approve them, unless the requester has the role set with `/config quanter-role`, which runs one job at a time without approval. The request card shows the quote, the card it expects to land on, and any quants others have already published at those bitrates.

Repos are named `{model}-exl3-{bpw}bpw`. A self-calibrated one is `-exl3-SC-{bpw}bpw-H{n}`, or `-SCT-` for `sc_mode:trace`. `-V{n}`, `-M{n}`, `-NG{n}` and `-HQ{n}` are added only when that setting was pinned and took effect, so a plain name always means exllamav3's defaults.

## Other commands

| Command | |
| --- | --- |
| `/quants [model]` | Every EXL3 quant of a model on HF, ours and the community's. With no model, the bot's latest |
| `/stop [job]` | Stop a job and terminate its pods. Admins stop any; a quanter stops their own |
| `/queue`, `/history` | What is running or waiting, and what finished and what it cost |
| `/health`, `/diag` | Uptime, live controllers, RunPod balance; `/diag` is admin-only and more verbose |
| `/pause`, `/resume` | Admin. Paused, approvals are refused and quanter jobs wait for approval. Running jobs keep going |
| `/catbench [model]` | Ask a model for a cute kitten twice, as SVG and as Python, and post what comes out |
| `/config quanter-role`, `/cache` | Admin settings |

## What a job does

```
/quant ─► approval ─► rent pod ─► download (pinned commit) ─► [SC: trace ─► probe ─► measure ─► optimize]
                                                                        │
                       cards + chart + collection ◄─ upload ◄─ KL ◄─ convert (per bpw)
```

Plain jobs get a pod per bitrate. SC jobs run every bitrate on one pod after a single calibration, and build a 6.0bpw donor first if the model has no quant of its own to trace through.

Each published repo carries `blockquant.json` (the base model's commit, the exllamav3 version and commit, every setting) and, when KL was measured, `bq_quality.json`. The completion message links each repo at the commit it had once its card went up, which is the one to pin.

## Which card, and what it costs

`backend/gpu_table.json` holds each card's fallback price and its conversion speed against an RTX 3090, measured where the file says so and estimated from the architecture otherwise. The launcher sorts candidates by price over speed and tries each on community before secure, under a per-size price cap. A plain job also needs enough VRAM to load the unquantized model for its KL number, so a 27B goes to an 80-96 GB card, not the cheapest 24 GB one.

The bot's quote runs the same selection against RunPod's live prices and stock. Every pod reports what it actually cost when it exits, and the completion message puts that next to the quote.

From the logs: a Qwen3.8-27B converted in a 1905 s pod on an RTX PRO 6000 Blackwell ($1.16) and in 5748-6135 s on an A100 SXM ($2.54-2.71).

## Measurement

KL against the unquantized model, with [qbench](https://github.com/turboderp-org/exllamav3/tree/master/eval/qbench) vendored under `backend/src/blockquant/selfcal/eval/qbench/` so the numbers read against turboderp's and ezexl3's. Scored on the model's own sampled output when an SC trace exists, wiki2 at 10x2048 otherwise. The card says which, and the KL-vs-bpw chart is only drawn when every quant in the family was scored on the same one.

Cards quote the median, not the mean: per qbench's own note, the mean is dominated by tokens the reference is itself unsure of. The full spread and the confidence buckets are in `bq_quality.json`.

One model is resident at a time. The reference writes its logits to disk, then the quant runs against them.

## Without Discord

```bash
# what the bot runs, one controller per pod
backend/venv/bin/python backend/scripts/run_runpod_job.py \
    --model Qwen/Qwen3-8B --variants 4.0 --hf-org yourorg --image ghcr.io/honkware/blockquant:sha-<commit>

# local GPU pipeline (the bot does not use it)
cd backend/src && ../venv/bin/python -m blockquant --model Qwen/Qwen3-8B --variants 4.0 --provider local
```

`run_runpod_job.py --help` has the rest, including `--sc`, `--sc-lite`, `--gpu`, `--max-price` and `--keep-pod` (leave the pod up after a failure, to rescue the output with `rescue_upload.py`).

## Image

`docker/Dockerfile.runpod`, built by `.github/workflows/build-runpod-image.yml` on a tag or by hand. One image for every architecture in `backend/arch_support.json`.

- exllamav3 is pinned to a commit (`EXLLAMAV3_REF`, 1.6.5 now). The self-calibration scripts in `backend/src/blockquant/selfcal/` are vendored from the same commit, and the vendor test fails if the two drift. Bump both behind a real run.
- Python 3.12 and torch 2.8/cu128 on the CUDA runtime base image, with exllamav3 from turboderp's prebuilt wheel (sm 7.5-12.0, sha256 checked). fla's Triton kernels do not import on 3.11.
- No nvcc in the image, so nothing compiles on a pod. B200/B300 have kernels in the wheel but are untested and never rented.
- RunPod Community hosts do not pull from ghcr.io: a Community 3090 never started the ghcr image in 15 minutes, and the same image from Docker Hub was up in 118 s. Point `RUNPOD_IMAGE` at the Docker Hub copy (`docker.io/honkware69/blockquant:sha-<commit>`). Secure pulls either.
- wikitext-2 is cached in the image so a pod cannot lose its KL number to an HF outage.

## Layout

```
src/                          Discord bot: commands, quoting, job state (data/jobs.json)
scripts/                      Python the bot shells out to: preflight, donor lookup, HF helpers
backend/scripts/              run_runpod_job.py (one controller per pod), publish_quant.py, pod_watchdog.py (cron)
backend/src/blockquant/
  remote/quant.py             what runs on the pod
  providers/runpod/           renting, SSH, health checks, prices
  cards.py                    repo names and model cards
  selfcal/                    vendored exllamav3 scripts, see VENDOR_MANIFEST.json
  pipeline.py, stages/        the local GPU pipeline
backend/src/api/              FastAPI + Celery front end for the local pipeline
docker/                       the RunPod image
```

## Known limits

- Self-calibration runs turboderp's scripts as shipped, but ezexl3 notes that MoE sensitivity measurement is still unvalidated, and most of the SC figures in the quote come from one 27B run.
- A model whose unquantized weights do not fit one 96 GB card gets no KL number unless the job asks for more GPUs.
- Card speeds not marked measured in `gpu_table.json` are estimates. A wrong one costs money, not quality.
- The API dashboard and `python -m blockquant` only see the local pipeline, not RunPod jobs.
