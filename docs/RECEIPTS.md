# Receipts and manifests

BlockQuant writes small JSON files next to each job so an operator can answer three questions quickly:

1. What was this job trying to build?
2. Which stage failed or skipped?
3. What outputs were produced and how were they verified?

The receipt is updated while the job runs. The manifest is the final summary.

## Files

| File | Written by | Purpose |
|---|---|---|
| `blockquant-job.json` | `backend/src/blockquant/receipts.py` | live job receipt |
| `blockquant-manifest.json` | `backend/src/blockquant/pipeline.py` | final pipeline manifest |

For local jobs, the workspace is usually:

```text
{workspace_dir}/{model_id with / replaced by --}/
```

Example:

```text
tmp/workdir/Qwen--Qwen2.5-7B-Instruct/blockquant-job.json
```

## Receipt shape

A new job starts with this structure:

```json
{
  "job_id": "9b6c1a2f",
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "provider": "runpod",
  "format": "exl3",
  "variants": ["4.5"],
  "status": "running",
  "created_at": 1760000000.0,
  "updated_at": 1760000000.0,
  "stages": {
    "cloud_setup": "pending",
    "cloud_run": "pending",
    "cloud_sync": "pending",
    "download": "pending",
    "convert": "pending",
    "quantize": "pending",
    "verify": "pending",
    "quality": "pending",
    "report": "pending",
    "upload": "pending"
  },
  "remote": {},
  "outputs": [],
  "recipe": {}
}
```

Stage values are simple strings:

- `pending`
- `running`
- `success`
- `failed`
- `skipped`

RunPod jobs use the `cloud_*` stages for pod launch, remote execution, and output sync. Local jobs use the normal pipeline stages.

## Recipe

The `recipe` block is shared by receipts and manifests. It is meant to be enough context to reproduce the run without copying secrets.

```json
{
  "base_model": {
    "repo_id": "Qwen/Qwen2.5-7B-Instruct",
    "revision": null
  },
  "quantization": {
    "format": "exl3",
    "variants": ["4.5"],
    "head_bits": 8,
    "cal_rows": 250,
    "cal_cols": 2048,
    "parallel_mode": false,
    "high_quality_bpws": [],
    "head_bits_8_bpws": [],
    "verify_quality": true
  },
  "runtime": {
    "provider": "runpod",
    "gpu_type": "NVIDIA H100 80GB HBM3",
    "cloud_type": "COMMUNITY",
    "container_disk_gb": 150,
    "volume_gb": 100
  }
}
```

The receipt code intentionally does not persist `HF_TOKEN`, `RUNPOD_API_KEY`, or other token values. Tests cover this so token regressions are caught.

## Manifest shape

The manifest is written at the end of the pipeline:

```json
{
  "job_id": "9b6c1a2f",
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "format": "exl3",
  "variants": ["4.5"],
  "recipe": {},
  "stages": [
    {
      "stage": "download",
      "success": true,
      "time": 12.5,
      "error": null
    }
  ],
  "outputs": [],
  "total_time": 1234.5
}
```

`stages` records elapsed time and failure text. `outputs` is the serialized `QuantOutput` list.

## Output verification

Each output keeps the old `verified` boolean and a structured verification object:

```json
{
  "variant": "4.5",
  "format": "exl3",
  "output_path": "...",
  "verified": true,
  "verification": {
    "status": "passed",
    "method": "filesystem",
    "message": "EXL3 directory present",
    "details": {}
  },
  "hf_repo_id": "",
  "hf_revision": "",
  "hf_url": "",
  "quality": {}
}
```

Verification status values:

| Status | Meaning |
|---|---|
| `pending` | verify stage has not touched the output yet |
| `passed` | load/filesystem check passed |
| `failed` | output was missing or failed verification |
| `skipped` | verifier could not run, usually due to an optional dependency |

GGUF verification uses `llama_cpp` when installed. If it is missing, verification is marked `skipped` instead of pretending to pass. EXL3 verification currently checks that the output directory exists and is non-empty.

## Debugging with receipts

Common checks:

```bash
jq '.status, .stages' tmp/workdir/*/blockquant-job.json
jq '.recipe.runtime, .outputs[]?.verification' tmp/workdir/*/blockquant-manifest.json
```

If a RunPod job fails after the remote quant finishes, look for:

- `remote.instance_id` in `blockquant-job.json`
- `cloud_run` status
- `cloud_sync` status
- fetched output list under `outputs`

Those fields usually tell you whether the failure was pod setup, remote execution, sync, verification, or upload.
