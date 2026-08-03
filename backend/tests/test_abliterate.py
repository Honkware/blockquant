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
