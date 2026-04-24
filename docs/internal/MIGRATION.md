# BlockQuant Backend Integration — Migration Guide

## What Changed

This release adds a **Python FastAPI + Celery backend** to the existing BlockQuant Discord bot. The bot now supports both **EXL3** (ExLlamaV3) and **GGUF** (llama.cpp) quantization formats, with jobs offloadable to a remote API.

### New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Python backend | `backend/` | Multi-format quantization pipeline |
| FastAPI server | `backend/src/api/main.py` | HTTP API for job submission/polling |
| Celery worker | `backend/src/scheduler/` | Async job queue backed by Redis |
| API client | `src/services/api-client.js` | Node.js client for the Python API |

### Modified Files

| File | Change |
|------|--------|
| `src/commands/quant.js` | Added `--format` option (`exl3` / `gguf`); branches to API or local mode |
| `src/commands/definitions.js` | Added `format` slash-command option |
| `.env.example` | Added `BLOCKQUANT_API_URL` |

### Preserved Files (unchanged)

All existing local quantization code remains intact:
- `src/services/quantizer.js`
- `src/services/queue.js`
- `src/services/huggingface.js`
- `src/services/workspace.js`
- `src/services/db.js`

## Setup

### 1. Install Python Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev,gguf]"
```

### 2. Start Redis

```bash
# Linux/macOS
redis-server --daemonize yes

# Windows (WSL or Docker)
docker run -d -p 6379:6379 redis:7-alpine
```

### 3. Start Celery Worker

```bash
cd backend
source venv/bin/activate
celery -A scheduler.tasks worker --loglevel=info
```

> **Windows users:** Celery's default prefork pool crashes on Windows with `PermissionError: [WinError 5] Access is denied`. Use the `solo` pool instead:
> ```powershell
> celery -A scheduler.tasks worker --loglevel=info -P solo
> ```

### 4. Start API Server

```bash
cd backend
source venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

### 5. Start Discord Bot

```bash
# In another terminal
cd BlockQuant
npm run start
```

## Configuration

### Env Vars

Add to `.env`:

```bash
# Optional — leave blank to use local subprocess mode
BLOCKQUANT_API_URL=http://localhost:8000
```

| Mode | `BLOCKQUANT_API_URL` | Behavior |
|------|----------------------|----------|
| API + local fallback | Set and reachable | EXL3 jobs can use API; GGUF jobs require API |
| Local only | Unset or unreachable | Falls back to existing `quantizer.js` / `queue.js` |

### GGUF Support

To quantize to GGUF format, you need `llama.cpp` built:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc)
```

## Usage

### Discord Commands

```
/quant url=meta-llama/Llama-3.1-8B-Instruct bpw=4.0 format=exl3
/quant url=mistralai/Mistral-7B-Instruct bpw=q4_k_m format=gguf
```

### CLI (Python Backend)

```bash
# Dry run
bq-pipeline --model microsoft/Phi-3-mini-4k-instruct --format exl3 --variants 4.0 --dry-run

# Real run
bq-pipeline --model microsoft/Phi-3-mini-4k-instruct --format exl3 --variants 4.0

# GGUF
bq-pipeline --model Qwen/Qwen2.5-7B --format gguf --variants q4_k_m,q5_k_m
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/api/v1/quant` | POST | Submit a quantization job |
| `/api/v1/jobs/{job_id}` | GET | Poll job status |

### Example

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/v1/quant \
  -H "Content-Type: application/json" \
  -d '{"model_id": "microsoft/Phi-3-mini-4k-instruct", "format": "exl3", "variants": ["4.0"]}'

curl http://localhost:8000/api/v1/jobs/<job_id>
```

## Rollback

To fully disable the API and return to the previous local-only behavior:

1. Remove or comment out `BLOCKQUANT_API_URL` in `.env`
2. Restart the bot

The bot will automatically fall back to spawning `exllamav3/convert.py` locally via `quantizer.js`.

## Architecture

```
┌─────────────┐     HTTP      ┌─────────────┐     Celery/Redis    ┌─────────────┐
│ Discord Bot │ ◄────────────► │  FastAPI    │ ◄─────────────────► │   Worker    │
│  (Node.js)  │   poll/status  │   (Python)  │      enqueue        │  (Python)   │
└─────────────┘                └─────────────┘                     └─────────────┘
                                      │                                  │
                                      │ subprocess                       │ subprocess
                                      ▼                                  ▼
                                ┌─────────────┐                   ┌─────────────┐
                                │ ExLlamaV3   │                   │ llama.cpp   │
                                │  convert.py │                   │  quantize   │
                                └─────────────┘                   └─────────────┘
```
