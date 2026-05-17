# Discord bot

The Discord bot is a thin front-end for the backend pipeline. It does not quantize models itself when `BLOCKQUANT_API_URL` is set; it submits jobs to FastAPI and polls until the Celery task finishes.

```text
/quant -> Node bot -> POST /api/v1/quant -> Celery worker -> blockquant.pipeline -> local or RunPod
```

Use the CLI for one-off jobs. Use the bot when you want a queue, server-visible status, history, and admin controls.

## Environment

Copy `.env.example` to `.env` and fill in the Discord keys:

```dotenv
BOT_TOKEN=
CLIENT_ID=
GUILD_ID=
ADMIN_IDS=123,456

HF_TOKEN=
HF_ORG=

# Required for provider=runpod
RUNPOD_API_KEY=

# Backend API used by src/services/api-client.js
BLOCKQUANT_API_URL=http://localhost:8000
```

Path keys used by the bot:

```dotenv
WORKSPACE_DIR=./tmp/workdir
EXLLAMAV3_DIR=./exllamav3
```

`src/config.js` reads `.env` from the repo root even if the process is started from another directory. For `WORKSPACE_DIR` and `EXLLAMAV3_DIR`, the repo `.env` value wins over an inherited shell value so stale Windows/WSL paths do not leak into a run.

## Start the backend

Run each long-running process in its own shell.

```bash
# 1. Redis
redis-server
```

```bash
# 2. Celery worker
cd backend
set -a; source ../.env; set +a
PYTHONPATH=src ./venv/bin/celery -A scheduler.tasks worker --loglevel=info -P solo
```

```bash
# 3. FastAPI
cd backend
set -a; source ../.env; set +a
PYTHONPATH=src ./venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Check the API before starting the bot:

```bash
curl http://127.0.0.1:8000/health
```

Expected shape:

```json
{"status":"ok","celery":true}
```

## Start the bot

From the repo root:

```bash
npm install
BLOCKQUANT_API_URL=http://localhost:8000 npm run dev
```

For a production-style process:

```bash
BLOCKQUANT_API_URL=http://localhost:8000 npm start
```

## `/quant`

Main options from `src/commands/definitions.js`:

| Option | Required | Notes |
|---|---:|---|
| `url` | yes | HuggingFace model ID or URL, for example `meta-llama/Llama-3.1-8B-Instruct` |
| `bpw` | yes | Comma-separated EXL3 bpw values such as `4.0,4.5,5.0`; GGUF uses names such as `q4_k_m` |
| `profile` | no | `fast`, `balanced`, or `quality` |
| `head_bits` | no | Admin override for EXL3 head bits |
| `format` | no | `exl3` by default; `gguf` is also accepted by the API |
| `category` | no | Used for bot metadata/history |
| `provider` | no | `local` or `runpod`; defaults to local in the command UI |

Example:

```text
/quant url:Qwen/Qwen2.5-7B-Instruct bpw:4.5 provider:runpod profile:balanced
```

## Other commands

| Command | Use |
|---|---|
| `/queue` | queue status |
| `/health` | bot/API health summary |
| `/score` | EXP balance |
| `/leaderboard` | EXP or model leaderboard |
| `/history` | recent jobs |
| `/pause`, `/resume` | admin queue controls |
| `/give` | admin EXP adjustment |
| `/cache` | admin cache status/prune/clear |
| `/diag` | admin diagnostics |

## Polling and failures

`src/services/api-client.js` polls `/api/v1/jobs/{job_id}` every 5 seconds. It stops when the API returns `complete` or `failed`.

If a job fails, check in this order:

1. Discord reply/error text.
2. FastAPI logs.
3. Celery worker logs.
4. The job workspace receipt: `blockquant-job.json`.
5. For RunPod jobs, the pod log or fetched RunPod result payload.

The API is not authenticated. Keep it bound to `127.0.0.1` or put a reverse proxy with auth in front of it.
