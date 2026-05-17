"""Small JSON receipts for jobs and quant recipes."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from blockquant.models import QuantConfig

STAGE_NAMES = [
    "cloud_setup",
    "cloud_run",
    "cloud_sync",
    "download",
    "convert",
    "quantize",
    "verify",
    "quality",
    "report",
    "upload",
]


def build_quant_recipe(config: QuantConfig) -> dict[str, Any]:
    recipe: dict[str, Any] = {
        "base_model": {
            "repo_id": config.model_id,
            "revision": None,
        },
        "quantization": {
            "format": config.format.value,
            "variants": list(config.variants),
            "head_bits": config.head_bits,
            "cal_rows": config.cal_rows,
            "cal_cols": config.cal_cols,
            "parallel_mode": config.parallel_mode,
            "high_quality_bpws": list(config.high_quality_bpws),
            "head_bits_8_bpws": list(config.head_bits_8_bpws),
            "verify_quality": config.verify_quality,
        },
        "runtime": {
            "provider": config.provider.value,
        },
    }
    if config.provider.value == "runpod":
        recipe["runtime"].update(
            {
                "gpu_type": config.runpod_gpu_type,
                "cloud_type": config.runpod_cloud_type.value,
                "container_disk_gb": config.runpod_container_disk_gb,
                "volume_gb": config.runpod_volume_gb,
            }
        )
    return recipe


def create_job_receipt(config: QuantConfig, job_id: str, workspace: Path) -> Path:
    receipt = {
        "job_id": job_id,
        "model_id": config.model_id,
        "provider": config.provider.value,
        "format": config.format.value,
        "variants": list(config.variants),
        "status": "running",
        "created_at": time.time(),
        "updated_at": time.time(),
        "stages": {stage: "pending" for stage in STAGE_NAMES},
        "remote": {},
        "outputs": [],
        "recipe": build_quant_recipe(config),
    }
    path = workspace / "blockquant-job.json"
    _write_json(path, receipt)
    return path


def update_job_receipt(path: Path | str, **updates: Any) -> None:
    path = Path(path)
    receipt = _read_json(path)
    receipt.update(updates)
    receipt["updated_at"] = time.time()
    _write_json(path, receipt)


def update_receipt_stage(path: Path | str, stage: str, status: str) -> None:
    path = Path(path)
    receipt = _read_json(path)
    receipt.setdefault("stages", {})[stage] = status
    receipt["updated_at"] = time.time()
    _write_json(path, receipt)


def update_receipt_remote(path: Path | str, **remote: Any) -> None:
    path = Path(path)
    receipt = _read_json(path)
    receipt.setdefault("remote", {}).update(remote)
    receipt["updated_at"] = time.time()
    _write_json(path, receipt)


def update_receipt_outputs(path: Path | str, outputs: list[Any]) -> None:
    path = Path(path)
    receipt = _read_json(path)
    receipt["outputs"] = [
        output.model_dump(mode="json") if hasattr(output, "model_dump") else output
        for output in outputs
    ]
    receipt["updated_at"] = time.time()
    _write_json(path, receipt)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
