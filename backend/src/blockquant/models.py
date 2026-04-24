"""Pydantic models matching the existing bot's data shapes.

This mirrors what the Node.js queue.js expects so the API
responses fit directly into the existing bot's job tracking.
"""
from __future__ import annotations
from enum import Enum
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field


class QuantFormat(str, Enum):
    GGUF = "gguf"
    EXL3 = "exl3"


class QuantConfig(BaseModel):
    """Job configuration — maps to what the Discord bot collects."""

    model_id: str  # e.g., "mistralai/Mistral-7B-Instruct"
    format: QuantFormat = QuantFormat.EXL3  # Default matches existing bot
    variants: list[str] = Field(default_factory=lambda: ["4.0"])
    use_imatrix: bool = True  # For GGUF
    provider: str = "local"  # "local" | "runpod" (others shelved under experimental/)
    spot: bool = False
    hf_org: str = ""  # Maps to existing HF_ORG
    hf_token: str = ""
    workspace_dir: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "blockquant-work")
    head_bits: int = 8  # EXL3 param, matches existing config
    cal_rows: int | None = None  # EXL3 param
    cal_cols: int | None = None  # EXL3 param
    # Per-BPW quantization flags (inspired by ezexl3)
    parallel_mode: bool = False  # -pm, MoE speedup
    high_quality_bpws: list[str] = Field(default_factory=list)  # -hq applied to these variants
    head_bits_8_bpws: list[str] = Field(default_factory=list)   # -hb 8 applied to these variants
    # Quality verification
    verify_quality: bool = True  # Run KL + PPL after quantization
    # Cloud provider settings — RunPod
    runpod_api_key: str = ""
    runpod_gpu_type: str = "NVIDIA H100 80GB HBM3"
    runpod_cloud_type: str = "COMMUNITY"
    runpod_container_disk_gb: int = 150
    runpod_volume_gb: int = 100
    runpod_ssh_key_path: str = "~/.ssh/id_rsa"


class QuantOutput(BaseModel):
    """Single quantized output — one per variant."""

    variant: str  # "4.0" for EXL3, "q4_k_m" for GGUF
    format: QuantFormat
    output_path: str  # Absolute path to output dir/file
    file_size_mb: float = 0.0
    verified: bool = False
    hf_repo_id: str = ""  # e.g., "Honkware/Mistral-7B-4bpw-exl3"
    hf_revision: str = ""
    hf_url: str = ""
    quality: dict = Field(default_factory=dict)  # {"kl_div": float, "ppl": float}


class StageResult(BaseModel):
    """Per-stage status for progress reporting."""

    stage: str  # download|convert|quantize|verify|upload
    success: bool
    wall_time_seconds: float = 0.0
    error: str | None = None


class PipelineResult(BaseModel):
    """Full job result — this is what the API returns to the bot."""

    job_id: str = ""
    config: QuantConfig
    stages: list[StageResult]
    outputs: list[QuantOutput]
    total_wall_time: float = 0.0
    manifest_path: str = ""  # blockquant-manifest.json
    status: str = "pending"  # pending|running|complete|failed


class JobStatusResponse(BaseModel):
    """API status endpoint response."""

    job_id: str
    status: str  # PENDING|STARTED|SUCCESS|FAILURE|RETRY
    progress: dict | None = None  # {stage: str, percent: int, message: str}
    result: PipelineResult | None = None
    error: str | None = None
