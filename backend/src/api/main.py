"""FastAPI server — the Discord bot calls this instead of running locally.

SECURITY: this service intentionally has no authentication. It is designed
to bind to ``127.0.0.1`` (see SETUP.md and README.md) and be reached only
from the local Discord bot or the local CLI. Exposing it on a public
interface would let any caller submit quant jobs against the operator's
RunPod credits. If you need a public surface, put it behind a reverse
proxy that handles auth.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from blockquant.models import (
    JobStatusResponse,
    ProviderName,
    QuantFormat,
    RunPodCloudType,
    validate_hf_org,
    validate_model_id,
    validate_variants,
)

import api.dashboard as dashboard

try:
    from scheduler.tasks import run_quantization, app as celery_app

    CELERY_OK = True
except ImportError:
    CELERY_OK = False

app = FastAPI(title="BlockQuant API", version="0.1.0")
app.include_router(dashboard.router)


class QuantRequest(BaseModel):
    model_id: str
    format: QuantFormat = QuantFormat.EXL3
    variants: list[str] = Field(default_factory=lambda: ["4.0"])
    provider: ProviderName = ProviderName.LOCAL
    hf_org: str = ""
    workspace: str | None = None
    parallel_mode: bool = False
    high_quality_bpws: list[str] = Field(default_factory=list)
    head_bits_8_bpws: list[str] = Field(default_factory=list)
    verify_quality: bool = True
    # RunPod settings
    runpod_api_key: str = ""
    runpod_gpu_type: str = "NVIDIA H100 80GB HBM3"
    runpod_cloud_type: RunPodCloudType = RunPodCloudType.COMMUNITY
    runpod_container_disk_gb: int = 150
    runpod_volume_gb: int = 100
    runpod_ssh_key_path: str = "~/.ssh/id_rsa"

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        return validate_model_id(value)

    @field_validator("hf_org")
    @classmethod
    def _validate_hf_org(cls, value: str) -> str:
        return validate_hf_org(value)

    @field_validator("runpod_container_disk_gb", "runpod_volume_gb")
    @classmethod
    def _validate_positive_gb(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("RunPod disk sizes must be positive")
        return value

    @model_validator(mode="after")
    def _validate_variants(self):
        self.variants = validate_variants(self.format, self.variants)
        self.high_quality_bpws = validate_variants(QuantFormat.EXL3, self.high_quality_bpws) if self.high_quality_bpws else []
        self.head_bits_8_bpws = validate_variants(QuantFormat.EXL3, self.head_bits_8_bpws) if self.head_bits_8_bpws else []
        return self


@app.get("/health")
async def health():
    return {"status": "ok", "celery": CELERY_OK}


@app.post("/api/v1/quant")
async def submit_job(request: QuantRequest):
    """Submit a quantization job. Returns job_id for polling."""
    if not CELERY_OK:
        raise HTTPException(503, "Celery not configured")

    config_dict = request.model_dump(mode="json", exclude={"workspace"})
    config_dict["hf_token"] = os.environ.get("HF_TOKEN", "")
    if request.workspace:
        config_dict["workspace_dir"] = request.workspace

    job = run_quantization.delay(config_dict)
    return {
        "job_id": job.id,
        "status": "queued",
        "check_url": f"/api/v1/jobs/{job.id}",
    }


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll job status — Discord bot hits this every few seconds."""
    if not CELERY_OK:
        raise HTTPException(503, "Celery not configured")

    from celery.result import AsyncResult

    result = AsyncResult(job_id, app=celery_app)

    status_map = {
        "PENDING": "queued",
        "STARTED": "running",
        "SUCCESS": "complete",
        "FAILURE": "failed",
        "RETRY": "retrying",
    }

    response = {
        "job_id": job_id,
        "status": status_map.get(result.status, result.status.lower()),
    }

    # Include live progress metadata while running
    if result.status == "STARTED" and result.info:
        response["progress"] = {
            "stage": result.info.get("stage", "running"),
            "percent": result.info.get("percent", 0),
            "message": result.info.get("message", "Running"),
        }

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)

    return response
