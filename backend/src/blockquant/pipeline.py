"""6-stage pipeline.

Stages mirror the existing Node.js flow:
  1. download   → snapshot_download from HF Hub
  2. convert    → FP16 for GGUF; no-op for EXL3 (uses HF directly)
  3. quantize   → ExLlamaV3 convert.py OR llama.cpp quantize
  4. verify     → load test + sample generation
  5. report     → perplexity + model card generation
  6. upload     → huggingface-cli upload

The output shape matches what the existing queue.js expects.
"""
import json
import time
import uuid
from pathlib import Path

from blockquant.models import (
    QuantConfig,
    QuantFormat,
    QuantOutput,
    PipelineResult,
    StageResult,
)
from blockquant.stages import download, convert, quantize, verify, report, upload, quality
from blockquant.providers import get_provider
from blockquant.monitoring import record_job_start, record_job_complete
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)


def _run_remote_pipeline(config: QuantConfig, progress_callback) -> PipelineResult:
    """Run pipeline on a cloud provider instance."""
    provider_name = config.provider
    logger.info(f"Using remote provider: {provider_name}")
    t0 = time.time()

    provider_kwargs: dict = {}
    if provider_name == "runpod":
        provider_kwargs["api_key"] = config.runpod_api_key
        provider_kwargs["gpu_type"] = config.runpod_gpu_type
        provider_kwargs["cloud_type"] = config.runpod_cloud_type
        provider_kwargs["container_disk_gb"] = config.runpod_container_disk_gb
        provider_kwargs["volume_gb"] = config.runpod_volume_gb
        provider_kwargs["ssh_key_path"] = config.runpod_ssh_key_path

    provider = get_provider(provider_name, **provider_kwargs)
    instance_id = provider.launch(config.model_dump(mode="json"))

    def _report(stage: str, pct: int, msg: str = ""):
        if progress_callback:
            progress_callback(stage, pct, msg)

    def _predicted_hf_url(variant: str) -> str:
        """URL for the repo the remote script will upload this variant to.

        Matches the convention baked into runpod_provider._QUANT_SCRIPT:
        ``{org_or_user}/{basename}-{format}-{variant}bpw``.
        """
        basename = config.model_id.split("/")[-1]
        slug = f"{basename}-{config.format.value}-{variant}bpw"
        if config.hf_org:
            return f"https://huggingface.co/{config.hf_org}/{slug}"
        return f"https://huggingface.co/{slug}"

    try:
        # Wait for readiness + bootstrap (both are no-ops for providers
        # that don't need them — see Provider base class defaults).
        _report("cloud_setup", 5, f"Launching {provider_name} instance...")
        provider.wait_for_active(instance_id)
        _report("cloud_setup", 15, f"Instance {instance_id} active")

        _report("cloud_setup", 20, "Bootstrapping instance...")
        provider.bootstrap(instance_id)
        _report("cloud_setup", 25, "Bootstrap complete")

        # Kick off the remote pipeline (fire-and-forget for RunPod).
        _report("cloud_run", 30, "Starting remote quantization...")
        provider.run_pipeline(
            instance_id=instance_id,
            model_id=config.model_id,
            format=config.format.value,
            variants=config.variants,
            hf_token=config.hf_token,
            hf_org=config.hf_org,
            head_bits=config.head_bits,
            use_imatrix=config.use_imatrix,
        )

        # Poll until the remote process exits.
        last_progress = ""
        while provider.is_pipeline_running(instance_id):
            time.sleep(5)
            progress = provider.get_progress(instance_id)
            if progress and progress != last_progress:
                last_progress = progress
                last_line = progress.strip().split("\n")[-1]
                _report("cloud_run", 50, last_line[:200])

        _report("cloud_run", 90, "Remote pipeline finished")

        # Sync outputs back (no-op when the remote script uploads directly
        # to HuggingFace, which is the RunPod default).
        outputs: list[QuantOutput] = []
        _report("cloud_sync", 92, "Downloading outputs...")
        workspace = config.workspace_dir / config.model_id.replace("/", "--")
        downloaded = provider.sync_outputs(
            instance_id,
            workspace,
            remote_rel_path=config.model_id.replace("/", "--"),
        )
        _report("cloud_sync", 95, f"Downloaded {len(downloaded)} files")

        # Translate the remote's result JSON into QuantOutput records.
        remote_result = provider.get_result()
        if remote_result and remote_result.get("outputs"):
            for out in remote_result["outputs"]:
                variant = out["variant"]
                outputs.append(
                    QuantOutput(
                        variant=variant,
                        format=config.format,
                        output_path=out.get("path", ""),
                        file_size_mb=0.0,
                        hf_repo_id=out.get("hf_repo_id", ""),
                        hf_url=out.get("hf_url") or _predicted_hf_url(variant),
                    )
                )

        total_time = time.time() - t0
        return PipelineResult(
            job_id=instance_id[:8],
            config=config,
            stages=[StageResult(stage="remote", success=True, wall_time_seconds=total_time)],
            outputs=outputs,
            total_wall_time=total_time,
            status="complete",
        )
    except Exception as exc:
        logger.error(f"Remote pipeline failed: {exc}")
        total_time = time.time() - t0
        return PipelineResult(
            job_id=instance_id[:8],
            config=config,
            stages=[StageResult(stage="remote", success=False, wall_time_seconds=total_time, error=str(exc))],
            outputs=[],
            total_wall_time=total_time,
            status="failed",
        )
    finally:
        _report("cloud_teardown", 98, f"Terminating {provider_name} instance...")
        provider.terminate(instance_id)
        _report("complete", 100, "Cloud pipeline complete")


def run_pipeline(config: QuantConfig, progress_callback=None) -> PipelineResult:
    job_id = str(uuid.uuid4())[:8]
    workspace = config.workspace_dir / config.model_id.replace("/", "--")
    workspace.mkdir(parents=True, exist_ok=True)

    def _report(stage: str, percent: int, message: str = ""):
        if progress_callback:
            progress_callback(stage, percent, message or f"{stage} ({percent}%)")

    # Route to cloud provider if not local
    if config.provider != "local":
        return _run_remote_pipeline(config, progress_callback)

    logger.info(f"[{job_id}] Starting: {config.model_id} ({config.format.value})")
    record_job_start(
        workspace_dir=config.workspace_dir,
        job_id=job_id,
        model_id=config.model_id,
        format=config.format.value,
        variants=config.variants,
        provider=config.provider,
    )
    t0 = time.time()

    stages: list[StageResult] = []
    outputs: list[QuantOutput] = []

    # Stage 1: Download
    _report("download", 5, "starting download")
    stages.append(_run_stage("download", download.run, config, workspace, _report))
    _report("download", 15, "download complete")
    if not stages[-1].success:
        return _finalize(job_id, config, stages, outputs, t0)

    # Stage 2: Convert (GGUF only)
    if config.format == QuantFormat.GGUF:
        _report("convert", 18)
        stages.append(_run_stage("convert", convert.run, config, workspace))
        _report("convert", 22)
        if not stages[-1].success:
            return _finalize(job_id, config, stages, outputs, t0)

    # Stage 3: Quantize
    _report("quantize", 25, "starting quantization")
    q_result = quantize.run(config, workspace, _report)
    _report("quantize", 85, "quantization complete")
    stages.append(
        StageResult(
            stage="quantize",
            success=q_result is not None and q_result.get("ok", False),
            wall_time_seconds=q_result["time"] if q_result else 0,
            error=q_result.get("error") if q_result and not q_result.get("ok") else None,
        )
    )
    if q_result and q_result.get("ok"):
        outputs = q_result["outputs"]
    else:
        return _finalize(job_id, config, stages, outputs, t0)

    # Stage 4: Verify
    _report("verify", 90, "verifying outputs")
    stages.append(_run_stage("verify", verify.run, config, workspace, outputs))

    # Stage 4b: Quality (KL + PPL) — optional
    if config.verify_quality and config.format == QuantFormat.EXL3 and outputs:
        _report("quality", 91, "running quality metrics (KL + PPL)")
        stages.append(_run_stage("quality", _run_quality_stage, config, workspace, outputs))
        _report("quality", 94, "quality metrics complete")

    # Stage 5: Report (model card)
    _report("report", 95, "generating model cards")
    stages.append(_run_stage("report", report.run, config, workspace, outputs))

    # Stage 6: Upload
    _report("upload", 97, "uploading to HuggingFace")
    stages.append(_run_stage("upload", upload.run, config, workspace, outputs))

    # Write manifest (matches existing blockquant-manifest.json)
    manifest = {
        "job_id": job_id,
        "model_id": config.model_id,
        "format": config.format.value,
        "variants": [o.variant for o in outputs],
        "stages": [
            {"stage": s.stage, "success": s.success, "time": s.wall_time_seconds}
            for s in stages
        ],
        "outputs": [o.model_dump(mode="json") for o in outputs],
        "total_time": time.time() - t0,
    }
    manifest_path = workspace / "blockquant-manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    _report("complete", 100)
    result = _finalize(job_id, config, stages, outputs, t0, str(manifest_path))
    record_job_complete(
        workspace_dir=config.workspace_dir,
        job_id=job_id,
        success=(result.status == "complete"),
        wall_time_seconds=result.total_wall_time,
        provider=config.provider,
    )
    return result


def _run_stage(name, func, config, workspace, *extra_args) -> StageResult:
    t = time.time()
    try:
        func(config, workspace, *extra_args)
        return StageResult(stage=name, success=True, wall_time_seconds=time.time() - t)
    except Exception as e:
        logger.error(f"Stage {name} failed: {e}")
        return StageResult(
            stage=name, success=False, wall_time_seconds=time.time() - t, error=str(e)
        )


def _run_quality_stage(config, workspace, outputs):
    """Wrapper so _run_stage can call quality.run directly."""
    quality.run(config, workspace, outputs)


def _finalize(job_id, config, stages, outputs, t0, manifest_path=""):
    total = time.time() - t0
    ok = all(s.success for s in stages)
    return PipelineResult(
        job_id=job_id,
        config=config,
        stages=stages,
        outputs=outputs,
        total_wall_time=total,
        manifest_path=manifest_path,
        status="complete" if ok else "failed",
    )
