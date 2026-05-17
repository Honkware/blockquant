# BlockQuant

Quantize HuggingFace models to EXL3 or GGUF from a local NVIDIA GPU or a RunPod pod, then publish the results back to HuggingFace.

BlockQuant is mostly glue around real quantization tooling:

- EXL3 uses [ExLlamaV3](https://github.com/turboderp-org/exllamav3).
- GGUF uses the llama.cpp conversion path.
- Local and RunPod jobs use the same backend pipeline.
- Every run writes a receipt and manifest so failed jobs are easier to debug or resume.

The CLI is the best path for one-off jobs. The Discord bot and FastAPI stack are for running the same pipeline as a small service.

## What is included

| Area | Path | Notes |
|---|---|---|
| Backend pipeline | `backend/src/blockquant/` | download, convert, quantize, verify, quality, report, upload |
| Providers | `backend/src/blockquant/providers/` | `local` and `runpod` |
| RunPod CLI | `backend/scripts/run_runpod_job.py` | launches a pod, runs the job, syncs results, terminates |
| HTTP API | `backend/src/api/main.py` | `/health`, `/api/v1/quant`, `/api/v1/jobs/{job_id}` |
| Discord bot | `src/` | slash commands, queue/history/EXP, API polling |
| Dashboard | `backend/scripts/log_dashboard.py` | tails quant logs in a browser |
| Docker image | `docker/` | pinned RunPod image and startup scripts |

## Requirements

| Tool | Needed for |
|---|---|
| Python 3.10+ | backend, CLI, API worker |
| Node.js 20+ | Discord bot only |
| Redis | FastAPI/Celery/Discord stack only |
| NVIDIA GPU + CUDA | local quantization |
| RunPod API key | RunPod quantization |
| HuggingFace write token | uploading output repos |

Clone ExLlamaV3 beside this repo if you are running EXL3 jobs locally or using the default RunPod bootstrap path:

```bash
git clone https://github.com/Honkware/blockquant.git
cd blockquant
git clone https://github.com/turboderp-org/exllamav3.git
```

## Setup

```bash
cd blockquant
cp .env.example .env
# Fill in HF_TOKEN, RUNPOD_API_KEY if needed, and Discord keys if using the bot.

cd backend
python -m venv venv
./venv/bin/pip install -r requirements.txt
cd ..

npm install   # only needed for the Discord bot or Node checks
```

For tests without the heavy ML stack:

```bash
cd backend
python -m venv ../.venv
../.venv/bin/pip install -r requirements-test.txt
cd ..
```

## Quickstart: RunPod CLI

This path does not need Node, Redis, Celery, Discord, or a local GPU.

```bash
./backend/venv/bin/python backend/scripts/run_runpod_job.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --variants 4.5 \
  --profile balanced \
  --gpu "NVIDIA H100 NVL" \
  --gpu-fallback "NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB" \
  --hf-org ""
```

Useful flags:

- `--profile fast|balanced|quality` changes calibration size and preferred cloud tier.
- `--tune` prints the resolved GPU/config/cost estimate without launching a pod.
- `--image ghcr.io/honkware/blockquant:latest` uses the prebuilt image instead of bootstrapping the pod.
- `--keep-pod` leaves the pod up after failure for debugging or rescue uploads.

RunPod jobs upload a small remote script, stream progress, fetch result metadata, sync outputs, and terminate the pod unless `--keep-pod` is set.

## Quickstart: local backend CLI

Use this when the machine already has CUDA, enough VRAM, and ExLlamaV3 on disk.

```bash
set -a; source ./.env; set +a
PYTHONPATH=backend/src ./backend/venv/bin/python -m blockquant \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --format exl3 \
  --variants 4.5 \
  --provider local \
  --workspace ./tmp/workdir
```

GGUF variants use lowercase llama.cpp-style names:

```bash
set -a; source ./.env; set +a
PYTHONPATH=backend/src ./backend/venv/bin/python -m blockquant \
  --model Qwen/Qwen2.5-7B-Instruct \
  --format gguf \
  --variants q4_k_m,q5_k_m \
  --provider local
```

## Discord and API stack

The service path is:

```text
Discord slash command -> Node bot -> FastAPI -> Celery worker -> BlockQuant pipeline -> local GPU or RunPod
```

Start the backend pieces in separate shells:

```bash
# Redis
redis-server

# Celery worker
cd backend
set -a; source ../.env; set +a
PYTHONPATH=src ./venv/bin/celery -A scheduler.tasks worker --loglevel=info -P solo

# FastAPI server
cd backend
set -a; source ../.env; set +a
PYTHONPATH=src ./venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then start the bot from the repo root:

```bash
BLOCKQUANT_API_URL=http://localhost:8000 npm run dev
```

The API is intentionally unauthenticated and should stay bound to `127.0.0.1` unless you put authentication in front of it.

More details:

- [Discord setup and command flow](docs/DISCORD.md)
- [HTTP API reference](docs/API.md)
- [Receipts and manifests](docs/RECEIPTS.md)
- [RunPod notes](docs/RUNPOD.md)

## Pipeline output

Each job writes two JSON files in its workspace:

- `blockquant-job.json` — live receipt with job status, stage status, provider metadata, outputs, and the quant recipe.
- `blockquant-manifest.json` — final manifest with stages, outputs, timings, and the same recipe.

Secrets are not written to either file. See [docs/RECEIPTS.md](docs/RECEIPTS.md) for the shape.

## Development

Run the fast checks:

```bash
PYTHONPATH=backend/src .venv/bin/pytest backend/tests -q
npm run check
```

The provider tests are mocked and do not launch GPUs or call RunPod/HuggingFace. See [CONTRIBUTING.md](CONTRIBUTING.md) for provider notes and test expectations.

## Security notes

- The FastAPI app has no auth. Keep it local or put it behind a real auth layer.
- RunPod SSH host keys are accepted automatically because pods are ephemeral; pod identity comes from the RunPod control plane.
- Quantized weights inherit the license of the upstream base model.

## Acknowledgements

- [turboderp](https://github.com/turboderp) for ExLlamaV3 and EXL3.
- [bartowski](https://huggingface.co/bartowski), [ArtusDev](https://huggingface.co/ArtusDev), and [TheBloke](https://huggingface.co/TheBloke) for the publishing conventions this project follows.

## License

MIT — see [LICENSE](LICENSE).
