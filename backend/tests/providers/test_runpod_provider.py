"""Unit tests for RunPodProvider."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from blockquant.providers.runpod.provider import RunPodProvider


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

@patch("blockquant.providers.runpod.provider._ensure_runpod")
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


@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_launch_merges_user_env(mock_ensure, mock_ssh_key, fake_pod):
    mock_rp = MagicMock()
    mock_rp.create_pod.return_value = fake_pod
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    provider.launch({"env": {"HF_TOKEN": "secret"}})

    _, kwargs = mock_rp.create_pod.call_args
    assert kwargs["env"]["HF_TOKEN"] == "secret"
    assert kwargs["env"]["PUBLIC_KEY"].startswith("ssh-rsa ")


@patch("blockquant.providers.runpod.provider._ensure_runpod")
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

@patch("blockquant.providers.runpod.provider.time.sleep", lambda *_: None)
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_terminate_confirmed_when_pod_gone(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = None  # pod no longer exists -> GONE
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.terminate("pod-abc123") is True
    mock_rp.terminate_pod.assert_called_with("pod-abc123")


@patch("blockquant.providers.runpod.provider.time.sleep", lambda *_: None)
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_terminate_confirmed_via_exited_status(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = {"desiredStatus": "EXITED"}
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.terminate("pod-abc123") is True


@patch("blockquant.providers.runpod.provider.time.sleep", lambda *_: None)
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_terminate_recovers_when_pod_gone_despite_api_error(mock_ensure, mock_ssh_key):
    """terminate_pod keeps erroring, but the pod is in fact gone -> confirmed."""
    mock_rp = MagicMock()
    mock_rp.terminate_pod.side_effect = RuntimeError("network")
    mock_rp.get_pod.return_value = None
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.terminate("pod-abc123") is True


@patch("blockquant.providers.runpod.provider.time.sleep", lambda *_: None)
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_terminate_unconfirmed_returns_false(mock_ensure, mock_ssh_key):
    """Pod stays RUNNING and terminate keeps failing -> returns False, no raise."""
    mock_rp = MagicMock()
    mock_rp.terminate_pod.side_effect = RuntimeError("network")
    mock_rp.get_pod.return_value = {"desiredStatus": "RUNNING"}
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.terminate("pod-abc123", verify_timeout=1, poll_interval=0) is False


@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_terminate_no_verify_returns_request_result(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.terminate("pod-abc123", verify=False) is True
    mock_rp.terminate_pod.assert_called_once_with("pod-abc123")
    mock_rp.get_pod.assert_not_called()


@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_get_pod_status(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_ensure.return_value = mock_rp
    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))

    mock_rp.get_pod.return_value = {"desiredStatus": "RUNNING"}
    assert provider.get_pod_status("pod-abc123") == "RUNNING"

    mock_rp.get_pod.return_value = None
    assert provider.get_pod_status("pod-abc123") == "GONE"


# ---------------------------------------------------------------------------
# SSH endpoint polling
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_get_ssh_endpoint_ready(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    endpoint = provider._get_ssh_endpoint("pod-abc123", timeout=2)
    assert endpoint == {"host": "1.2.3.4", "port": 2222}


@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_get_ssh_endpoint_terminal_state(mock_ensure, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = {"desiredStatus": "EXITED"}
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    with pytest.raises(RuntimeError, match="terminal state: EXITED"):
        provider._get_ssh_endpoint("pod-abc123", timeout=1, interval=0.1)


@patch("blockquant.providers.runpod.provider._ensure_runpod")
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

@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
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

@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_bootstrap_success(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"ok")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.bootstrap("pod-abc123") is True


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
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


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
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

@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_run_command(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"nvidia-smi output")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    result = provider.run("pod-abc123", "nvidia-smi")

    assert result["code"] == 0
    assert "nvidia-smi output" in result["stdout"]


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_get_progress_returns_tailed_log(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"[quantize] 4.5 bpw complete\n")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert "[quantize]" in provider.get_progress("pod-abc123")


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_is_pipeline_running_true(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_ensure_rp.return_value = mock_rp
    _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"running\n")

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    assert provider.is_pipeline_running("pod-abc123") is True


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
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

@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
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


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
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

@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
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


def _make_pipeline_provider_and_client(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    """Helper: wire SSH + SFTP mocks and return (provider, mock_client)."""
    mock_rp = MagicMock()
    mock_rp.get_pod.return_value = _running_pod_with_ssh()
    mock_rp.api_key = "test-api-key"
    mock_ensure_rp.return_value = mock_rp
    mock_client = _mock_ssh_exec(mock_ensure_pk, exit_code=0, stdout=b"12345\n")
    mock_sftp = MagicMock()
    mock_file = MagicMock()
    mock_sftp.file.return_value.__enter__ = MagicMock(return_value=mock_file)
    mock_sftp.file.return_value.__exit__ = MagicMock(return_value=False)
    mock_client.open_sftp.return_value = mock_sftp
    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    return provider, mock_client


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_run_pipeline_config_contains_pod_and_key(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    """Config JSON must include pod_id, runpod_api_key, and keep_pod."""
    provider, _ = _make_pipeline_provider_and_client(
        mock_ensure_rp, mock_ensure_pk, mock_ssh_key
    )
    uploaded: dict[str, bytes] = {}

    def _capture(instance_id, data, remote_path):
        uploaded[remote_path] = data

    with patch.object(provider, "_upload_bytes", side_effect=_capture):
        result = provider.run_pipeline(
            instance_id="pod-abc123",
            model_id="foo/bar",
            format="exl3",
            variants=["4.5"],
            hf_token="tok",
        )

    assert result["status"] == "started"
    assert "/root/bq-config.json" in uploaded, "bq-config.json not uploaded"
    cfg = json.loads(uploaded["/root/bq-config.json"])
    assert cfg["pod_id"] == "pod-abc123"
    assert "runpod_api_key" in cfg
    assert cfg["keep_pod"] is False


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_run_pipeline_keep_pod_true_honored(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    """keep_pod=True must propagate into the uploaded config JSON."""
    provider, _ = _make_pipeline_provider_and_client(
        mock_ensure_rp, mock_ensure_pk, mock_ssh_key
    )
    uploaded: dict[str, bytes] = {}

    def _capture(instance_id, data, remote_path):
        uploaded[remote_path] = data

    with patch.object(provider, "_upload_bytes", side_effect=_capture):
        provider.run_pipeline(
            instance_id="pod-abc123",
            model_id="foo/bar",
            format="exl3",
            variants=["4.5"],
            keep_pod=True,
        )

    cfg = json.loads(uploaded["/root/bq-config.json"])
    assert cfg["keep_pod"] is True


@patch("blockquant.providers.runpod.provider.time.sleep", return_value=None)
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_get_pod_resilient_retries_transient_errors(mock_ensure, mock_sleep, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.side_effect = [RuntimeError("connection reset by peer"), _running_pod_with_ssh()]
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    pod = provider._get_pod_resilient("pod-abc123")

    assert pod["desiredStatus"] == "RUNNING"
    assert mock_rp.get_pod.call_count == 2
    mock_sleep.assert_called_once()


@patch("blockquant.providers.runpod.provider.time.sleep", return_value=None)
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_get_pod_resilient_does_not_retry_non_transient(mock_ensure, mock_sleep, mock_ssh_key):
    mock_rp = MagicMock()
    mock_rp.get_pod.side_effect = ValueError("bad request")
    mock_ensure.return_value = mock_rp

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    with pytest.raises(ValueError, match="bad request"):
        provider._get_pod_resilient("pod-abc123")

    assert mock_rp.get_pod.call_count == 1
    mock_sleep.assert_not_called()


@patch("blockquant.providers.runpod.provider.time.sleep", return_value=None)
@patch("blockquant.providers.runpod.provider._ensure_paramiko")
def test_run_retries_transient_exec_error(mock_ensure_pk, mock_sleep, mock_ssh_key):
    fake_paramiko = MagicMock()
    fake_paramiko.SSHException = RuntimeError
    mock_ensure_pk.return_value = fake_paramiko
    first_client = MagicMock()
    second_client = MagicMock()

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    with patch.object(provider, "_connect_ssh", side_effect=[first_client, second_client]) as connect:
        with patch.object(
            provider,
            "_exec",
            side_effect=[RuntimeError("connection reset"), {"stdout": "ok", "stderr": "", "code": 0}],
        ):
            result = provider.run("pod-abc123", "echo ok", retries=2)

    assert result["stdout"] == "ok"
    assert connect.call_count == 2
    assert first_client.close.called
    assert second_client.close.called
    mock_sleep.assert_called_once()


@patch("blockquant.providers.runpod.provider.time.sleep", return_value=None)
@patch("blockquant.providers.runpod.provider._ensure_paramiko")
def test_sftp_put_retries_transient_write_error(mock_ensure_pk, mock_sleep, mock_ssh_key):
    fake_paramiko = MagicMock()
    fake_paramiko.SSHException = RuntimeError
    mock_ensure_pk.return_value = fake_paramiko
    first_client = MagicMock()
    second_client = MagicMock()
    first_sftp = MagicMock()
    second_sftp = MagicMock()
    first_client.open_sftp.return_value = first_sftp
    second_client.open_sftp.return_value = second_sftp
    attempts = []

    def flaky_put(sftp):
        attempts.append(sftp)
        if len(attempts) == 1:
            raise EOFError("dropped")
        return "ok"

    provider = RunPodProvider(api_key="fake-key", ssh_key_path=str(mock_ssh_key))
    with patch.object(provider, "_connect_ssh", side_effect=[first_client, second_client]):
        result = provider._sftp_put_with_retry("pod-abc123", flaky_put, retries=2)

    assert result == "ok"
    assert attempts == [first_sftp, second_sftp]
    assert first_sftp.close.called
    assert second_sftp.close.called
    mock_sleep.assert_called_once()


@patch("blockquant.providers.runpod.provider._ensure_paramiko")
@patch("blockquant.providers.runpod.provider._ensure_runpod")
def test_run_pipeline_does_not_put_tokens_on_command_line(mock_ensure_rp, mock_ensure_pk, mock_ssh_key):
    provider, mock_client = _make_pipeline_provider_and_client(
        mock_ensure_rp, mock_ensure_pk, mock_ssh_key
    )

    provider.run_pipeline(
        instance_id="pod-abc123",
        model_id="foo/bar",
        format="exl3",
        variants=["4.5"],
        hf_token="hf-secret-token",
    )

    commands = "\n".join(str(call.args[0]) for call in mock_client.exec_command.call_args_list)
    assert "hf-secret-token" not in commands
    assert "fake-key" not in commands


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

@patch("blockquant.providers.runpod.provider._ensure_runpod")
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


@patch("blockquant.providers.runpod.provider._ensure_runpod")
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


@patch("blockquant.providers.runpod.provider._ensure_runpod")
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


@patch("blockquant.providers.runpod.provider._ensure_runpod")
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
