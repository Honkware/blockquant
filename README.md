# BlockQuant

Quantize HuggingFace LLMs to **EXL3** (and GGUF) on your own GPU or in the
cloud, and publish them to HuggingFace — from a Discord slash command or a
plain CLI.

BlockQuant wraps [ExLlamaV3](https://github.com/turboderp-org/exllamav3) with
a FastAPI + Celery backend, a Discord front-end, a pluggable provider layer,
and a live browser dashboard that renders the model's mixture-of-experts as
it quantizes in real time.

---

## What's in the box

| Layer | What it does |
|---|---|
| **Discord bot** (`src/`) | `/quant` command, live progress thread, EXP charging, queue + history |
| **FastAPI + Celery** (`backend/src/api/`, `backend/src/blockquant/`) | Job orchestration, pipeline stages, progress reporting |
| **Providers** (`backend/src/blockquant/providers/`) | `local` (your RTX 4090) and `runpod` (H100 / A100 cloud). Lambda + Modal are shelved — see `experimental/README.md` |
| **CLI** (`backend/scripts/run_runpod_job.py`) | One-shot RunPod job without the Discord bot |
| **Live dashboard** (`backend/scripts/log_dashboard.py`) | Browser UI that tails the quant log and visualises the 48×256 expert matrix being compressed |

## Supported paths (actually validated)

- **Local** — runs on your own NVIDIA GPU via ExLlamaV3's `convert.py`.
- **RunPod** — provisions a pod, bootstraps the stack, quantizes, uploads to HF, tears down. Tested end-to-end on H100 NVL with a 35B MoE.

**Shelved** (not shipped as supported): Modal, Lambda. See
[`experimental/README.md`](experimental/README.md) for the un-shelving
checklist.

---

## Quickstart — local GPU

Prereqs: Node 20+, Python 3.10+, NVIDIA GPU with recent CUDA, an
[ExLlamaV3 clone](https://github.com/turboderp-org/exllamav3), Redis.

```bash
git clone https://github.com/Honkware/blockquant.git
cd blockquant

# Vendored upstream — kept out of this repo
git clone https://github.com/turboderp-org/exllamav3.git

# Python backend
cd backend && python -m venv venv && ./venv/bin/pip install -r requirements.txt

# Node bot
cd .. && npm install

# Config
cp .env.example .env && $EDITOR .env
# Minimum: BOT_TOKEN, CLIENT_ID, GUILD_ID, HF_TOKEN, EXLLAMAV3_DIR

# Run (in three terminals)
./backend/venv/bin/redis-server
./backend/venv/bin/python -m celery -A scheduler.tasks worker --loglevel=info -P solo
./backend/venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
node src/index.js
```

Then in Discord: `/quant url:meta-llama/Llama-3.1-8B-Instruct bpw:4.5 provider:local`.

## Quickstart — RunPod cloud

Prereqs: above, plus a [RunPod](https://runpod.io) account and an SSH key
pair at `~/.ssh/id_rsa{,.pub}` (the public key is injected into every pod
via `env[PUBLIC_KEY]` — no need to pre-register it).

```bash
# Add to .env
RUNPOD_API_KEY=...  # from runpod.io console

# CLI — bypass the Discord bot entirely
./backend/venv/bin/python backend/scripts/run_runpod_job.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variants 4.5 \
  --gpu "NVIDIA H100 80GB HBM3" \
  --gpu-fallback "NVIDIA H100 NVL,NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB" \
  --hf-org ""  # blank = upload to your personal HF account

# Optional live dashboard
./backend/venv/bin/python backend/scripts/log_dashboard.py \
  --port 8088 \
  --log backend/logs/runpod-qwen35b-4.5.log
# → open http://localhost:8088
```

On first `create_pod` failure (out of stock), the CLI walks through the
`--gpu-fallback` list automatically. Bootstrap and a 35B-class quant land at
~4h on H100 NVL for ~$10 community-cloud.

## Live dashboard

`log_dashboard.py` is a single-file FastAPI app. It tails the quant log, runs
a small **LogParser** that emits typed events from regex rules, derives
stats (tensor rate, seconds per layer, runtime, spend) anchored on log
timestamps + file mtime so they survive a refresh, and streams the state to
the browser via Server-Sent Events.

The centerpiece of the UI is a live visualisation of the 48-layer × 256-expert
grid for the model being quantized — cells settle from cream (pending) →
vermillion (measuring) → sage (quantized) in real time.

---

## Slash commands

| Command | Description | Access |
|---|---|---|
| `/quant` | Submit a quant job (local or RunPod) | All |
| `/queue` | Check queue status | All |
| `/health` | Bot health + Celery status | All |
| `/score` | Check EXP balance | All |
| `/leaderboard` | Top EXP earners / quantized models | All |
| `/history` | Recent jobs | All |
| `/give` | Grant EXP | Admin |
| `/pause` \| `/resume` | Control the job queue | Admin |
| `/cache` | Local model cache ops | Admin |
| `/diag` | Detailed diagnostics | Admin |

## Pipeline stages

1. **Download** — HF snapshot pulled to workspace
2. **Convert** — GGUF only (EXL3 uses HF directly)
3. **Quantize** — ExLlamaV3 `convert.py`, streaming progress
4. **Verify** — load-test each variant
5. **Quality** *(optional, EXL3 only)* — KL-div + PPL vs FP16 baseline via `exllamav3/eval/model_diff.py`
6. **Report** — model-card README with measured quality
7. **Upload** — one repo per bpw: `{org}/{model}-exl3-{bpw}bpw`

Cloud providers run stages 1–5 remotely; the dashboard reads their tailed
log. Local provider runs stages in-process.

## Project layout

```
BlockQuant/
├── src/                       Discord bot (Node.js)
├── backend/
│   ├── src/
│   │   ├── api/main.py        FastAPI: /api/v1/quant, /health
│   │   └── blockquant/
│   │       ├── pipeline.py    6-stage orchestrator
│   │       ├── models.py      QuantConfig + friends
│   │       ├── providers/     local, runpod, base ABC
│   │       └── stages/        download/convert/quantize/verify/quality/report/upload
│   ├── scripts/
│   │   ├── run_runpod_job.py  CLI
│   │   ├── log_dashboard.py   Browser dashboard
│   │   ├── rename_hf_repo_on_complete.py  Post-upload HF repo rename
│   │   ├── cleanup_pods.py / list_pods.py / list_gpus.py   Ops utilities
│   │   └── requirements.txt
│   └── tests/providers/       27 mock-based tests for RunPod
├── experimental/              Shelved providers (Modal, Lambda) — not supported
├── docs/internal/             Planning / cost / migration notes (not user-facing)
└── .env.example
```

## Contributing

New provider? Subclass `backend/src/blockquant/providers/base.Provider`.
The ABC defines three required methods (`launch`, `terminate`, `run`) plus
seven optional hooks with safe defaults — so you only implement what your
backend actually needs. Look at `runpod_provider.py` for the reference
shape (SSH + SFTP + retry-on-transient-network-error throughout).

All new providers must have a matching test file under
`backend/tests/providers/` modelled on `test_runpod_provider.py`.

## Security

The FastAPI server has **no authentication** by design — it binds to
`127.0.0.1` and is reached only by the local Discord bot or CLI. If you
expose port 8000 publicly, anyone reaching it can submit quant jobs
against your RunPod credits. Put a reverse proxy with auth in front if
you need a public surface.

The RunPod provider uses paramiko's `AutoAddPolicy` for SSH host keys.
This is intentional — each pod is ephemeral and gets a fresh key, so
strict checking would require throwing the integrity check away anyway.
The pod's identity is verified through the RunPod API (the `pod_id` we
asked for) and traffic is encrypted over SSH.

Report security issues privately to the maintainer rather than via a
public issue.

## License

MIT — see [LICENSE](LICENSE).
