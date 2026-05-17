import json
from pathlib import Path

from blockquant.models import QuantConfig, QuantFormat, QuantOutput
from blockquant import pipeline
from blockquant.pipeline import run_pipeline
from blockquant.receipts import build_quant_recipe


def test_run_pipeline_writes_receipt_and_manifest(monkeypatch, tmp_path):
    def fake_download(config, workspace, progress_callback=None):
        (workspace / "model").mkdir(parents=True, exist_ok=True)

    def fake_quantize(config, workspace, progress_callback=None):
        out_dir = workspace / "out" / "4.0bpw"
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "time": 0.1,
            "outputs": [
                QuantOutput(
                    variant="4.0",
                    format=QuantFormat.EXL3,
                    output_path=str(out_dir),
                    file_size_mb=12.5,
                )
            ],
        }

    def fake_verify(config, workspace, outputs):
        outputs[0].verified = True

    def fake_report(config, workspace, outputs):
        (workspace / "README.md").write_text("model card\n", encoding="utf-8")

    def fake_upload(config, workspace, outputs):
        outputs[0].hf_repo_id = "tester/model-exl3-4.0bpw"
        outputs[0].hf_url = "https://huggingface.co/tester/model-exl3-4.0bpw"

    monkeypatch.setattr(pipeline.download, "run", fake_download)
    monkeypatch.setattr(pipeline.quantize, "run", fake_quantize)
    monkeypatch.setattr(pipeline.verify, "run", fake_verify)
    monkeypatch.setattr(pipeline.report, "run", fake_report)
    monkeypatch.setattr(pipeline.upload, "run", fake_upload)

    config = QuantConfig(
        model_id="tester/model",
        variants=["4.0"],
        hf_token="hf-secret-token",
        runpod_api_key="runpod-secret-token",
        workspace_dir=tmp_path,
        verify_quality=False,
    )

    result = run_pipeline(config)

    workspace = tmp_path / "tester--model"
    receipt = json.loads((workspace / "blockquant-job.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

    assert result.status == "complete"
    assert receipt["status"] == "complete"
    assert receipt["stages"]["download"] == "success"
    assert receipt["stages"]["convert"] == "skipped"
    assert receipt["stages"]["quality"] == "skipped"
    assert receipt["stages"]["upload"] == "success"
    assert receipt["outputs"][0]["verified"] is True
    assert receipt["recipe"] == manifest["recipe"]
    assert manifest["recipe"]["base_model"]["repo_id"] == "tester/model"
    assert manifest["recipe"]["quantization"]["variants"] == ["4.0"]
    assert manifest["recipe"]["runtime"]["provider"] == "local"

    persisted = json.dumps({"receipt": receipt, "manifest": manifest})
    assert "hf-secret-token" not in persisted
    assert "runpod-secret-token" not in persisted


def test_runpod_recipe_keeps_runtime_knobs_but_not_secrets(tmp_path):
    config = QuantConfig(
        model_id="tester/model",
        provider="runpod",
        variants=["4.5"],
        hf_token="hf-secret-token",
        runpod_api_key="runpod-secret-token",
        runpod_gpu_type="NVIDIA H100 NVL",
        runpod_cloud_type="SECURE",
        workspace_dir=tmp_path,
    )

    recipe = build_quant_recipe(config)

    assert recipe["runtime"] == {
        "provider": "runpod",
        "gpu_type": "NVIDIA H100 NVL",
        "cloud_type": "SECURE",
        "container_disk_gb": 150,
        "volume_gb": 100,
    }
    persisted = json.dumps(recipe)
    assert "hf-secret-token" not in persisted
    assert "runpod-secret-token" not in persisted
