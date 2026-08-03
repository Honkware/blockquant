"""Abliteration flag: config surface + payload isolation.

The load-bearing property is that a job WITHOUT --abliterate is unchanged:
no new keys reach the pod, so the remote script takes exactly the code path
it took before the feature existed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockquant.models import QuantConfig
from blockquant.providers.runpod.provider import RunPodProvider


@pytest.fixture(autouse=True)
def mock_ssh_key(tmp_path):
    key = tmp_path / "id_rsa"
    key.write_text("-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----\n")
    (tmp_path / "id_rsa.pub").write_text("ssh-rsa AAAA... test@host\n")
    return key


def _cfg_sent(mock_ssh_key, **kwargs) -> dict:
    """Run run_pipeline with everything stubbed; return the pod's config JSON."""
    provider = RunPodProvider(api_key="k", ssh_key_path=str(mock_ssh_key))
    captured: dict = {}

    def _capture(instance_id, data, remote_path, mode=None):
        if remote_path.endswith("bq-config.json"):
            captured.update(json.loads(data.decode("utf-8")))

    with patch.object(provider, "_upload_bytes", side_effect=_capture), \
         patch.object(provider, "_upload_file", MagicMock()), \
         patch.object(provider, "run", MagicMock(return_value={"exit_code": 0, "stdout": "MISSING", "stderr": ""})), \
         patch.object(provider, "run_detached", MagicMock(return_value={"stdout": "123"})):
        provider.run_pipeline(
            instance_id="pod-1", model_id="org/model", format="exl3",
            variants=["4.0"], **kwargs,
        )
    return captured


def test_default_job_sends_no_abliterate_keys(mock_ssh_key):
    cfg = _cfg_sent(mock_ssh_key)
    assert "abliterate" not in cfg
    assert "abliterate_trials" not in cfg
    assert "abliterate_fusion" not in cfg


def test_abliterate_job_sends_the_keys(mock_ssh_key):
    cfg = _cfg_sent(mock_ssh_key, abliterate=True, abliterate_trials=12,
                    abliterate_fusion="residual")
    assert cfg["abliterate"] is True
    assert cfg["abliterate_trials"] == 12
    assert cfg["abliterate_fusion"] == "residual"


def test_quantconfig_defaults_off():
    c = QuantConfig(model_id="org/model")
    assert c.abliterate is False
    assert c.abliterate_fusion == "baked"
    assert c.abliterate_trials == 40


def test_quantconfig_rejects_bad_fusion():
    with pytest.raises(ValueError):
        QuantConfig(model_id="org/model", abliterate_fusion="sideways")


def test_quantconfig_normalizes_fusion_case():
    c = QuantConfig(model_id="org/model", abliterate_fusion="BAKED")
    assert c.abliterate_fusion == "baked"


@pytest.mark.parametrize("trials", [0, 201])
def test_quantconfig_rejects_out_of_range_trials(trials):
    with pytest.raises(ValueError):
        QuantConfig(model_id="org/model", abliterate_trials=trials)


# --- card ---------------------------------------------------------------

from blockquant.cards import build_abliteration_section, render_exl3_card

_CFG = {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 36}
_REPORT = {
    "base_repo": "Qwen/Qwen3-8B", "tool": "exliberate", "tool_version": "0.2.0",
    "refusal_pre": 0.92, "refusal_post": 0.04, "rebound_pp": 1.3,
    "kl_post_vs_pre": 0.0412, "trials": 40, "seed": 0, "passed": True,
    "mode": "baked", "reproduce_file": "fusion_report.json",
}


def _card(**kw):
    return render_exl3_card(
        base_repo="Qwen/Qwen3-8B", repo_id="o/r", variant="4.0", head_bits=8,
        cal_rows=250, size_gb=5.1, model_config=_CFG, quant_rows=[],
        collection_url="http://c", quantized_by="o", **kw,
    )


def test_no_abliteration_section_on_a_normal_card():
    card = _card()
    assert "## Abliteration" not in card
    assert "abliterated" not in card


def test_abliteration_section_is_autofilled():
    card = _card(abliteration=_REPORT)
    assert "Abliterated with exliberate v0.2.0 (rank-k whitened subspace), baked into EXL3 4.0" in card
    assert "| Refusal rate, before | 92.0% |" in card
    assert "| Refusal rate, after | 4.0% |" in card
    assert "| Rebound after quantization | +1.3 pp |" in card
    assert "`0.0412`" in card
    assert "fusion_report.json" in card
    assert "- abliterated" in card.split("---")[1]  # front-matter tags


def test_section_omits_values_the_report_lacks():
    section = build_abliteration_section(
        {"refusal_pre": 0.9, "refusal_post": 0.1}, "4.0"
    )
    assert "Refusal rate, before" in section
    assert "Rebound after quantization" not in section
    assert "Capability delta" not in section


def test_section_accepts_already_scaled_percentages():
    section = build_abliteration_section({"refusal_pre": 92.0, "refusal_post": 4.0}, "4.0")
    assert "92.0%" in section and "4.0%" in section
