"""Pydantic models matching the existing bot's data shapes.

This mirrors what the Node.js queue.js expects so the API
responses fit directly into the existing bot's job tracking.
"""
from __future__ import annotations
from enum import Enum
import re
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator


class QuantFormat(str, Enum):
    GGUF = "gguf"
    EXL3 = "exl3"


class ProviderName(str, Enum):
    LOCAL = "local"
    RUNPOD = "runpod"


class RunPodCloudType(str, Enum):
    COMMUNITY = "COMMUNITY"
    SECURE = "SECURE"


_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_HF_ORG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXL3_VARIANT_RE = re.compile(r"^(?:[1-9]\d*)(?:\.\d+)?$")
_GGUF_VARIANT_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def validate_model_id(value: str) -> str:
    value = value.strip()
    if not _MODEL_ID_RE.match(value):
        raise ValueError("model_id must look like 'org/model'")
    if ".." in value or value.startswith(('/', '.')):
        raise ValueError("model_id contains unsafe path segments")
    return value


def validate_hf_org(value: str) -> str:
    value = value.strip()
    if value and ("/" in value or not _HF_ORG_RE.match(value)):
        raise ValueError("hf_org must be a single HuggingFace namespace")
    return value


def validate_variants(format: QuantFormat, variants: list[str]) -> list[str]:
    cleaned = [v.strip() for v in variants if v and v.strip()]
    if not cleaned:
        raise ValueError("variants must contain at least one value")
    pattern = _EXL3_VARIANT_RE if format == QuantFormat.EXL3 else _GGUF_VARIANT_RE
    label = "EXL3 bpw" if format == QuantFormat.EXL3 else "GGUF quant"
    for variant in cleaned:
        if not pattern.match(variant) or ".." in variant or "/" in variant:
            raise ValueError(f"invalid {label} variant: {variant!r}")
    return cleaned


class QuantConfig(BaseModel):
    """Job configuration — maps to what the Discord bot collects."""

    model_id: str  # e.g., "mistralai/Mistral-7B-Instruct"
    format: QuantFormat = QuantFormat.EXL3  # Default matches existing bot
    variants: list[str] = Field(default_factory=lambda: ["4.0"])
    use_imatrix: bool = True  # For GGUF
    provider: ProviderName = ProviderName.LOCAL
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
