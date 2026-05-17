# HTTP API

The FastAPI app is the service boundary used by the Discord bot. It accepts a quant request, queues a Celery job, and exposes a polling endpoint.

It has no built-in authentication. Bind it to localhost unless another service handles auth.

## Start

```bash
# Redis
redis-server
```

```bash
# Worker
cd backend
set -a; source ../.env; set +a
PYTHONPATH=src ./venv/bin/celery -A scheduler.tasks worker --loglevel=info -P solo
```

```bash
# API
cd backend
set -a; source ../.env; set +a
PYTHONPATH=src ./venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Health

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "celery": true
}
```

`celery: false` means the API process could not import the worker app. Check `PYTHONPATH`, dependencies, and the backend virtualenv.

## Submit a job

```http
POST /api/v1/quant
Content-Type: application/json
```

Minimal EXL3 request:

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "format": "exl3",
  "variants": ["4.5"],
  "provider": "local",
  "hf_org": ""
}
```

RunPod request:

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "format": "exl3",
  "variants": ["4.5"],
  "provider": "runpod",
  "hf_org": "",
  "runpod_gpu_type": "NVIDIA H100 80GB HBM3",
  "runpod_cloud_type": "COMMUNITY",
  "runpod_container_disk_gb": 150,
  "runpod_volume_gb": 100,
  "runpod_ssh_key_path": "~/.ssh/id_rsa"
}
```

Response:

```json
{
  "job_id": "f8e8a7b1-...",
  "status": "queued",
  "check_url": "/api/v1/jobs/f8e8a7b1-..."
}
```

The API reads `HF_TOKEN` from the API process environment and injects it into the backend config. Do not send HF tokens from Discord clients.

## Request fields

| Field | Default | Notes |
|---|---|---|
| `model_id` | required | HuggingFace model ID. URLs should be normalized by callers before submitting. |
| `format` | `exl3` | `exl3` or `gguf`. |
| `variants` | `["4.0"]` | EXL3 bpw strings such as `4.0`; GGUF names such as `q4_k_m`. |
| `provider` | `local` | `local` or `runpod`. |
| `hf_org` | `""` | Blank uploads to the token owner's namespace. |
| `workspace` | `null` | Optional override for `workspace_dir`. |
| `parallel_mode` | `false` | Passed through to the quant config. |
| `high_quality_bpws` | `[]` | EXL3 variants that should use the high-quality path. |
| `head_bits_8_bpws` | `[]` | EXL3 variants that should use 8-bit heads. |
| `verify_quality` | `true` | Enables the quality stage where supported. |
| `runpod_gpu_type` | `NVIDIA H100 80GB HBM3` | Preferred RunPod GPU. |
| `runpod_cloud_type` | `COMMUNITY` | `COMMUNITY` or `SECURE`. |
| `runpod_container_disk_gb` | `150` | Must be positive. |
| `runpod_volume_gb` | `100` | Must be positive. |
| `runpod_ssh_key_path` | `~/.ssh/id_rsa` | Private key path for SSH/SFTP. |

`runpod_api_key` exists in the request model for compatibility, but normal deployments should use `RUNPOD_API_KEY` from the backend environment or the RunPod CLI flags.

## Poll a job

```http
GET /api/v1/jobs/{job_id}
```

Queued:

```json
{
  "job_id": "f8e8a7b1-...",
  "status": "queued"
}
```

Running:

```json
{
  "job_id": "f8e8a7b1-...",
  "status": "running",
  "progress": {
    "stage": "quantize",
    "percent": 42,
    "message": "quantize (42%)"
  }
}
```

Complete:

```json
{
  "job_id": "f8e8a7b1-...",
  "status": "complete",
  "result": {
    "job_id": "9b6c1a2f",
    "status": "complete",
    "outputs": [],
    "manifest_path": ".../blockquant-manifest.json"
  }
}
```

Failed:

```json
{
  "job_id": "f8e8a7b1-...",
  "status": "failed",
  "error": "..."
}
```

Status mapping comes from Celery:

| Celery | API |
|---|---|
| `PENDING` | `queued` |
| `STARTED` | `running` |
| `SUCCESS` | `complete` |
| `FAILURE` | `failed` |
| `RETRY` | `retrying` |

## curl example

```bash
curl -sS http://127.0.0.1:8000/api/v1/quant \
  -H 'content-type: application/json' \
  -d '{
    "model_id":"Qwen/Qwen2.5-7B-Instruct",
    "format":"exl3",
    "variants":["4.5"],
    "provider":"local"
  }'
```

Then poll the returned `check_url`.
