"""Unit tests for RunPodProvider."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from blockquant.providers.runpod_provider import RunPodProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_ssh_key(tmp_path):
    """Provide a fake SSH key pair so RunPodProvider can init."""
    key = tmp_path / "id_rsa"
    key.write_text("-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----\n")
    pub = tmp_path / "id_rsa.pub"
    pub.write_text("ssh-rsa AAAA... test@host\n")
    return key


@pytest.fixture
def fake_pod():
    return {"id": "pod-abc123"}


def _running_pod_with_ssh():
    return {
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [
                {"privatePort": 22, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": "2222"}
            ]
        },
    }


def _mock_ssh_exec(mock_ensure_pk, exit_code: int, stdout: bytes = b"", stderr: bytes = b""):
    """Wire paramiko mocks so exec_command returns the given shape."""
    mock_client = MagicMock()
    mock_stdout = MagicMock()
    mock_stdout.channel.recv_exit_status.return_value = exit_code
    mock_stdout.read.return_value = stdout
    mock_stderr = MagicMock()
    mock_stderr.read.return_value = stderr
    mock_client.exec_command.return_value = (None, mock_stdout, mock_stderr)
    mock_client.get_transport.return_value = MagicMock()
    mock_ensure_pk.return_value = MagicMock()
    mock_ensure_pk.return_value.SSHClient.return_value = mock_client
    mock_ensure_pk.return_value.AutoAddPolicy.return_value = "policy"
    return mock_client


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_provider_requires_api_key(mock_ssh_key):
    with pytest.raises(ValueError, match="RunPod API key required"):
        RunPodProvider(api_key="", ssh_key_path=str(mock_ssh_key))


def test_provider_requires_ssh_key():
    with pytest.raises(ValueError, match="SSH private key not found"):
        RunPodProvider(api_key="fake-key", ssh_key_path="/nonexistent/key")


def test_provider_requires_public_key(tmp_path):
    # Only the private key exists — no .pub
    key = tmp_path / "id_rsa_no_pub"
    key.write_text("private")
    with pytest.raises(ValueError, match="SSH public key not found"):
        RunPodProvider(api_key="fake-key", ssh_key_path=str(key))


def test_public_key_loaded(mock_ssh_key):
    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider._pubkey.startswith("ssh-rsa ")


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_launch_injects_public_key(mock_ensure, mock_ssh_key, fake_pod):
    mock_rp = MagicMock()
    mock_rp.create_pod.return_value = fake_pod
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(
        api_key="fake-key",
        gpu_type="NVIDIA RTX A4000",
        ssh_key_path=str(mock_ssh_key),
    )
    instance_id = provider.launch({})

    assert instance_id == "pod-abc123"
    _, kwargs = mock_rp.create_pod.call_args
    assert kwargs["gpu_type_id"] == "NVIDIA RTX A4000"
    assert kwargs["cloud_type"] == "COMMUNITY"
    assert kwargs["container_disk_in_gb"] == 150
    assert kwargs["volume_in_gb"] == 100
    assert kwargs["start_ssh"] is True
    assert kwargs["support_public_ip"] is True
    # The critical fix — PUBLIC_KEY must be in env.
    assert kwargs["env"]["PUBLIC_KEY"].startswith("ssh-rsa ")


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_launch_merges_user_env(mock_ensure, mock_ssh_key, fake_pod):
    mock_rp = MagicMock()
    mock_rp.create_pod.return_value = fake_pod
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    provider.launch({"env": {"HF_TOKEN": "secret"}})

    _, kwargs = mock_rp.create_pod.call_args
    assert kwargs["env"]["HF_TOKEN"] == "secret"
    assert kwargs["env"]["PUBLIC_KEY"].startswith("ssh-rsa ")


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_launch_with_network_volume(mock_ensure, mock_ssh_key, fake_pod):
    mock_rp = MagicMock()
    mock_rp.create_pod.return_value = fake_pod
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(
        api_key="fake-key",
        ssh_key_path=str(mock_ssh_key),
        network_volume_id="vol-123",
        data_center_id="EU-RO-1",
    )
    provider.launch({})

    _, kwargs = mock_rp.create_pod.call_args
    assert kwargs["network_volume_id"] == "vol-123"
    assert kwargs["data_center_id"] == "EU-RO-1"


# ---------------------------------------------------------------------------
# Terminate
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_terminate(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    provider.terminate("pod-abc123")

    mock_rp.terminate_pod.assert_called_once_with("pod-abc123")


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_terminate_graceful_on_error(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.terminate_pod.side_effect = RuntimeError("network")
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    provider.terminate("pod-abc123")  # should swallow


# ---------------------------------------------------------------------------
# SSH endpoint polling
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_ssh_endpoint_ready(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    endpoint = provider._get_ssh_endpoint("pod-abc123", timeout=2)
    assert endpoint == {"host": "1.2.3.4", "port": 2222}


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_ssh_endpoint_terminal_state(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = {"desiredStatus": "EXITED"}
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    with pytest.raises(RuntimeError, match="terminal state: EXITED"):
        provider._get_ssh_endpoint("pod-abc123", timeout=1, interval=0.1)


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_ssh_endpoint_timeout(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = {
        "desiredStatus": "RUNNING",
        "runtime": {"ports": []},
    }
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    with pytest.raises(TimeoutError):
        provider._get_ssh_endpoint("pod-abc123", timeout=0.2, interval=0.1)


# ---------------------------------------------------------------------------
# SSH keepalive is set
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_ssh_keepalive_set(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    mock_client = _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"ok")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    provider.run("pod-abc123", "echo ok")

    mock_client.get_transport.return_value.set_keepalive.assert_called_with(30)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_bootstrap_success(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"ok")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.bootstrap("pod-abc123") is True


@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_bootstrap_short_circuits_when_marker_present(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    # First call (marker check) returns "yes" — bootstrap should early-exit.
    mock_client = _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"yes\n")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.bootstrap("pod-abc123") is True
    # Fast-path: interpreter probe + torch-version check + marker check = 3 calls.
    # No apt-get / pip-install should run.
    assert mock_client.exec_command.call_count == 3


@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_bootstrap_system_packages_fail(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=1, stdout=b"", stderr=b"apt failed")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.bootstrap("pod-abc123") is False


# ---------------------------------------------------------------------------
# run / get_progress / is_pipeline_running
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_run_command(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"nvidia-smi output")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    result = provider.run("pod-abc123", "nvidia-smi")

    assert result["code"] == 0
    assert "nvidia-smi output" in result["stdout"]


@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_progress_returns_tailed_log(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"[quantize] 4.5 bpw complete\n")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert "[quantize]" in provider.get_progress("pod-abc123")


@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_is_pipeline_running_true(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"running\n")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.is_pipeline_running("pod-abc123") is True


@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_is_pipeline_running_false(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"done\n")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.is_pipeline_running("pod-abc123") is False


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_result_parses_json(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    payload = b'{"status": "complete", "outputs": [{"variant": "4.5", "path": "/w/out"}], "total_time": 123.4}'
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=payload)

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    provider._pod_id = "pod-abc123"

    result = provider.get_result()
    assert result["status"] == "complete"
    assert result["outputs"][0]["variant"] == "4.5"


@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_result_none_when_empty(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    provider._pod_id = "pod-abc123"
    assert provider.get_result() is None


# ---------------------------------------------------------------------------
# run_pipeline starts background process
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_paramiko")
@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_run_pipeline_is_non_blocking(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    mock_client = _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"12345\n")
    # SFTP mock
    mock_sftp = MagicMock()
    mock_file = MagicMock()
    mock_sftp.file.return_value.__enter__ = MagicMock(return_value=mock_file)
    mock_sftp.file.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.open_sftp.return_value = mock_sftp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    result = provider.run_pipeline(
        instance_id="pod-abc123",
        model_id="foo/bar",
        format="exl3",
        variants=["4.5"],
        hf_token="tok",
    )
    assert result["status"] == "started"
    # The launch command must use nohup + background.
    launched = [c for c in mock_client.exec_command.call_args_list if "nohup" in str(c)]
    assert launched, "expected a nohup-based launch command"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_cost_per_hour_live(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_gpu.return_value = {
        "id": "NVIDIA H100 80GB HBM3",
        "communityPrice": 1.85,
        "securePrice": 2.99,
    }
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(
        api_key="fake-key",
        gpu_type="NVIDIA H100 80GB HBM3",
        ssh_key_path=str(mock_ssh_key),
    )
    assert provider.get_cost_per_hour() == 1.85


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_cost_per_hour_falls_back_to_secure(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_gpu.return_value = {
        "id": "NVIDIA H100 80GB HBM3",
        "communityPrice": None,
        "securePrice": 2.99,
    }
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(
        api_key="fake-key",
        gpu_type="NVIDIA H100 80GB HBM3",
        ssh_key_path=str(mock_ssh_key),
    )
    assert provider.get_cost_per_hour() == 2.99


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_cost_per_hour_static_fallback_on_error(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_gpu.side_effect = RuntimeError("network")
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(
        api_key="fake-key",
        gpu_type="NVIDIA H100 80GB HBM3",
        ssh_key_path=str(mock_ssh_key),
    )
    assert provider.get_cost_per_hour() == 1.99


# ---------------------------------------------------------------------------
# resolve_profile — preset + override merging
# ---------------------------------------------------------------------------

def test_resolve_profile_known_presets_have_required_keys():
    for name in ("fast", "balanced", "quality"):
        cfg = RunPodProvider.resolve_profile(name)
        assert "cloud_type" in cfg
        assert "gpu_preference" in cfg and isinstance(cfg["gpu_preference"], list)
        assert "cal_rows" in cfg and isinstance(cfg["cal_rows"], int)
        assert "cal_cols" in cfg and isinstance(cfg["cal_cols"], int)


def test_resolve_profile_unknown_raises():
    with pytest.raises(KeyError, match="Unknown profile"):
        RunPodProvider.resolve_profile("ludicrous")


def test_resolve_profile_explicit_override_wins():
    cfg = RunPodProvider.resolve_profile("fast", cal_rows=512)
    assert cfg["cal_rows"] == 512  # override
    assert cfg["cal_cols"] == 2048  # preset


def test_resolve_profile_none_override_keeps_preset():
    cfg = RunPodProvider.resolve_profile("balanced", cal_rows=None, cal_cols=None)
    assert cfg["cal_rows"] == 250
    assert cfg["cal_cols"] == 2048


def test_resolve_profile_quality_uses_secure_cloud():
    cfg = RunPodProvider.resolve_profile("quality")
    assert cfg["cloud_type"] == "SECURE"
    assert cfg["cal_rows"] >= 250  # always meets-or-exceeds balanced


def test_resolve_profile_internal_keys_excluded():
    """Profile metadata keys (_walltime_factor etc) shouldn't leak."""
    cfg = RunPodProvider.resolve_profile("balanced")
    assert all(not k.startswith("_") for k in cfg)


@patch("blockquant.providers.runpod_provider._ensure_runpod")
def test_get_cost_per_hour_unknown_gpu(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_gpu.return_value = None
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(
        api_key="fake-key",
        gpu_type="NVIDIA Imaginary GPU",
        ssh_key_path=str(mock_ssh_key),
    )
    assert provider.get_cost_per_hour() == 2.00
