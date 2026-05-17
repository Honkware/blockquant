import pytest
from pydantic import ValidationError

from blockquant.models import ProviderName, QuantConfig, QuantFormat, RunPodCloudType


def test_quant_config_accepts_supported_values(tmp_path):
    config = QuantConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        format="exl3",
        variants=["4.0", "4.5"],
        provider="runpod",
        runpod_cloud_type="SECURE",
        hf_org="blockblockblock",
        workspace_dir=tmp_path,
    )

    assert config.format == QuantFormat.EXL3
    assert config.provider == ProviderName.RUNPOD
    assert config.runpod_cloud_type == RunPodCloudType.SECURE
    assert config.variants == ["4.0", "4.5"]


@pytest.mark.parametrize(
    "patch",
    [
        {"model_id": ""},
        {"model_id": "../model"},
        {"model_id": "model-only"},
        {"provider": "lambda"},
        {"provider": "modal"},
        {"runpod_cloud_type": "CHEAP"},
        {"hf_org": "org/suborg"},
        {"variants": []},
        {"variants": ["../../oops"]},
        {"runpod_container_disk_gb": 0},
        {"runpod_volume_gb": -1},
    ],
)
def test_quant_config_rejects_bad_inputs(patch):
    data = {"model_id": "test/model", **patch}
    with pytest.raises(ValidationError):
        QuantConfig(**data)


def test_gguf_variants_use_safe_names():
    config = QuantConfig(model_id="test/model", format="gguf", variants=["q4_k_m", "q8_0"])
    assert config.variants == ["q4_k_m", "q8_0"]


@pytest.mark.parametrize("variant", ["4.0", "Q4_K_M", "q4-k-m", "../q4"])
def test_gguf_rejects_non_gguf_variant_names(variant):
    with pytest.raises(ValidationError):
        QuantConfig(model_id="test/model", format="gguf", variants=[variant])
