"""RunPod GPU provider — on-demand pods with SSH + tail-log progress.

Design (matches Lambda provider so pipeline.py's poll loop works):

    launch()           → create_pod(start_ssh=True, env={PUBLIC_KEY})
    wait_for_active()  → poll get_pod() until public SSH port exposed
    bootstrap()        → idempotent deps install (marker file)
    run_pipeline()     → SFTP /root/quant.py, start via `nohup ... &`, return
    get_progress()     → `tail -n 30 /root/bq.log`
    is_pipeline_running() → `pgrep -f /root/quant.py`
    get_result()       → read /root/bq-result.json
    terminate()        → terminate_pod() (always in finally)

Requirements:
    1. RUNPOD_API_KEY env var.
    2. SSH key pair at ~/.ssh/id_rsa{,.pub}. The public key is injected into
       the pod via env["PUBLIC_KEY"] on create — no need to pre-register it
       with `runpod ssh add-key`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from blockquant.providers.base import Provider
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)

# Paramiko's transport layer emits WARN-level "Socket exception" /
# "Error reading SSH protocol banner" lines on every transient TCP reset.
# Our SSH retry wrappers (run / _get_pod_resilient / _sftp_put_with_retry)
# handle those transparently — surfacing them only adds noise to the
# dashboard ledger. Real ERROR-level paramiko output still passes through.
logging.getLogger("paramiko.transport").setLevel(logging.ERROR)

# Lazy imports so the module loads without SDKs present.
_runpod = None
_paramiko = None


def _ensure_runpod():
    global _runpod
    if _runpod is None:
        import runpod
        _runpod = runpod
    return _runpod


def _ensure_paramiko():
    global _paramiko
    if _paramiko is None:
        import paramiko
        _paramiko = paramiko
    return _paramiko


# Remote paths — kept under /root so they survive bootstrap but are pod-local.
REMOTE_SCRIPT = "/root/quant.py"
REMOTE_LOG = "/root/bq.log"
REMOTE_RESULT = "/root/bq-result.json"
BOOTSTRAP_MARKER = "/root/.bq-bootstrapped"


class RunPodProvider(Provider):
    name = "runpod"

    DEFAULT_SSH_KEY = Path.home() / ".ssh" / "id_rsa"

    # ----- Speedup profiles ----------------------------------------------
    # Curated sets of (cloud, GPU preference, calibration depth) that the
    # CLI exposes via --profile. Each preset is a default; any explicit
    # CLI flag from the user overrides the matching preset value.
    #
    # Wall-time / cost columns are rough multipliers vs `balanced` for a
    # 35B-class MoE — adjust expectations linearly for other model sizes.
    PROFILES: dict[str, dict] = {
        "fast": {
            "cloud_type": "COMMUNITY",
            "gpu_preference": [
                "NVIDIA H100 80GB HBM3",
                "NVIDIA H100 NVL",
                "NVIDIA H100 PCIe",
                "NVIDIA A100-SXM4-80GB",
            ],
            "cal_rows": 128,
            "cal_cols": 2048,
            "_walltime_factor": 0.6,
            "_cost_factor": 0.7,
            "_summary": "fewer cal rows, community cloud — quickest cheap run",
        },
        "balanced": {
            "cloud_type": "COMMUNITY",
            "gpu_preference": [
                "NVIDIA H100 80GB HBM3",
                "NVIDIA H100 NVL",
                "NVIDIA H100 PCIe",
                "NVIDIA A100-SXM4-80GB",
            ],
            "cal_rows": 250,
            "cal_cols": 2048,
            "_walltime_factor": 1.0,
            "_cost_factor": 1.0,
            "_summary": "default — ExLlamaV3's standard calibration",
        },
        "quality": {
            "cloud_type": "SECURE",
            "gpu_preference": ["NVIDIA H100 80GB HBM3"],
            "cal_rows": 512,
            "cal_cols": 2048,
            "_walltime_factor": 1.4,
            "_cost_factor": 1.4,
            "_summary": "secure cloud + deeper calibration — for the public-facing run you cite as canonical",
        },
    }

    @classmethod
    def resolve_profile(cls, profile: str, **overrides) -> dict:
        """Merge a named profile with explicit per-knob overrides.

        Returns a dict containing `cloud_type`, `gpu_preference` (list),
        `cal_rows`, `cal_cols`. Any kwarg in ``overrides`` whose value is
        not None replaces the profile's value. Unknown profile names
        raise ``KeyError``.
        """
        if profile not in cls.PROFILES:
            raise KeyError(
                f"Unknown profile {profile!r}. Available: {', '.join(cls.PROFILES)}"
            )
        base = {k: v for k, v in cls.PROFILES[profile].items() if not k.startswith("_")}
        for k, v in overrides.items():
            if v is not None and v != "":
                base[k] = v
        return base


    def __init__(
        self,
        api_key: str = "",
        gpu_type: str = "NVIDIA H100 80GB HBM3",
        cloud_type: str = "COMMUNITY",
        container_disk_gb: int = 150,
        volume_gb: int = 100,
        ssh_key_path: str = "",
        ssh_wait_timeout: int = 600,
        network_volume_id: str = "",
        data_center_id: str = "",
        install_flash_attn: bool = False,
        image: str = "",
    ):
        api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not api_key:
            raise ValueError(
                "RunPod API key required. Set RUNPOD_API_KEY env var or pass api_key."
            )
        _ensure_runpod().api_key = api_key
        self.gpu_type = gpu_type
        self.cloud_type = cloud_type
        self.container_disk_gb = container_disk_gb
        self.volume_gb = volume_gb
        self.ssh_key_path = Path(ssh_key_path).expanduser() if ssh_key_path else self.DEFAULT_SSH_KEY
        self.ssh_wait_timeout = ssh_wait_timeout
        self.network_volume_id = network_volume_id
        self.data_center_id = data_center_id
        self.install_flash_attn = install_flash_attn
        if not self.ssh_key_path.exists():
            raise ValueError(
                f"SSH private key not found at {self.ssh_key_path}. "
                f"Generate one with: ssh-keygen -t rsa -b 4096 -f {self.ssh_key_path} -N ''"
            )
        pubkey_path = self.ssh_key_path.with_suffix(self.ssh_key_path.suffix + ".pub") \
            if self.ssh_key_path.suffix else Path(str(self.ssh_key_path) + ".pub")
        if not pubkey_path.exists():
            raise ValueError(
                f"SSH public key not found at {pubkey_path}. "
                f"It is read and injected into the pod via env[PUBLIC_KEY]."
            )
        self._pubkey = pubkey_path.read_text(encoding="utf-8").strip()
        self._pod_id: str | None = None
        self._ssh_endpoint: dict | None = None
        self._last_result: dict | None = None
        self._gpu_price_cache: dict[str, float] | None = None
        # Python interpreter inside the pod, discovered during bootstrap.
        self._remote_py: str = "python3"
        # Override the pod image. When this points at our pre-baked
        # ghcr.io/honkware/blockquant:* image, bootstrap() short-circuits
        # to a no-op since every dep is already installed.
        self.image = image or os.environ.get("BLOCKQUANT_RUNPOD_IMAGE", "")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, config: dict) -> str:
        rp = _ensure_runpod()
        logger.info(f"Creating RunPod pod with GPU: {self.gpu_type}")

        # Image priority:
        #   1. Explicit override (constructor arg or BLOCKQUANT_RUNPOD_IMAGE env)
        #   2. Default to RunPod's pytorch base — bootstrap installs everything
        # When the image is our pre-baked ghcr.io/honkware/blockquant:*, bootstrap
        # short-circuits because every dep is already installed (see bootstrap()).
        image = self.image or "runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204"

        # Always inject the public key so `start_ssh=True` actually works.
        # (Without PUBLIC_KEY, RunPod's SSH daemon comes up with no authorized_keys.)
        env = {"PUBLIC_KEY": self._pubkey}
        env.update(config.get("env", {}) or {})

        kwargs = dict(
            name=f"blockquant-{int(time.time())}",
            image_name=image,
            gpu_type_id=self.gpu_type,
            cloud_type=self.cloud_type,
            container_disk_in_gb=self.container_disk_gb,
            volume_in_gb=self.volume_gb,
            support_public_ip=True,
            start_ssh=True,
            ports="22/tcp",
            env=env,
        )
        if self.network_volume_id:
            kwargs["network_volume_id"] = self.network_volume_id
            # Network volumes are region-pinned.
            if self.data_center_id:
                kwargs["data_center_id"] = self.data_center_id

        pod = rp.create_pod(**kwargs)
        self._pod_id = pod["id"]
        self._ssh_endpoint = None
        self._last_result = None
        logger.info(f"RunPod created: {self._pod_id}")
        return self._pod_id

    def terminate(self, instance_id: str):
        rp = _ensure_runpod()
        try:
            rp.terminate_pod(instance_id)
            logger.info(f"RunPod terminated: {instance_id}")
        except Exception as e:
            logger.warning(f"RunPod terminate failed: {e}")

    def _get_pod_resilient(self, instance_id: str):
        """rp.get_pod() with retry on transient network errors (TLS reset, etc.)."""
        rp = _ensure_runpod()
        for attempt in range(5):
            try:
                return rp.get_pod(instance_id)
            except Exception as e:
                # requests.ConnectionError, urllib3 ProtocolError, ssl errors, etc.
                msg = str(e).lower()
                transient = any(s in msg for s in (
                    "connection reset", "connection aborted", "connectionreset",
                    "remote host", "10054", "timed out", "temporarily unavailable",
                    "eof occurred", "max retries",
                ))
                if not transient or attempt == 4:
                    raise
                wait = min(2 ** attempt, 15)
                logger.warning(f"get_pod transient error, retry {attempt + 1}/5 in {wait}s: {e}")
                time.sleep(wait)

    def _get_ssh_endpoint(self, instance_id: str, timeout: int | None = None, interval: int = 5) -> dict:
        if self._ssh_endpoint is not None:
            return self._ssh_endpoint

        timeout = timeout or self.ssh_wait_timeout
        t0 = time.time()
        while time.time() - t0 < timeout:
            pod = self._get_pod_resilient(instance_id)
            if pod is None:
                raise RuntimeError(f"Pod {instance_id} not found")
            status = pod.get("desiredStatus", "UNKNOWN")
            if status in ("EXITED", "TERMINATED", "FAILED"):
                raise RuntimeError(f"Pod {instance_id} reached terminal state: {status}")
            runtime = pod.get("runtime")
            if runtime:
                ports = runtime.get("ports") or []
                for p in ports:
                    if p.get("privatePort") == 22 and p.get("isIpPublic"):
                        self._ssh_endpoint = {"host": p["ip"], "port": int(p["publicPort"])}
                        logger.info(
                            f"SSH endpoint ready: {self._ssh_endpoint['host']}:{self._ssh_endpoint['port']}"
                        )
                        return self._ssh_endpoint
            time.sleep(interval)
        raise TimeoutError(
            f"Pod {instance_id} did not expose public SSH endpoint within {timeout}s"
        )

    def wait_for_active(self, instance_id: str, timeout: int | None = None, interval: int = 5) -> dict:
        try:
            ssh = self._get_ssh_endpoint(instance_id, timeout=timeout, interval=interval)
            return {"status": "active", "id": instance_id, "ssh": ssh}
        except TimeoutError as e:
            return {"status": "timeout", "id": instance_id, "error": str(e)}
        except RuntimeError as e:
            return {"status": "failed", "id": instance_id, "error": str(e)}

    # ------------------------------------------------------------------
    # SSH
    # ------------------------------------------------------------------

    def _connect_ssh(self, instance_id: str, retries: int = 20):
        """Open a paramiko SSH client with keepalive and exp-backoff retries."""
        paramiko = _ensure_paramiko()
        endpoint = self._get_ssh_endpoint(instance_id)
        last_err = None
        for attempt in range(retries):
            client = paramiko.SSHClient()
            # AutoAddPolicy is intentional: each RunPod pod is ephemeral and
            # gets a fresh host key, so strict checking would require either
            # known_hosts gymnastics or skipping the integrity check anyway.
            # The pod's identity is verified via the API (we requested it,
            # we know the pod_id), and traffic to it is over SSH (encrypted).
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=endpoint["host"],
                    port=endpoint["port"],
                    username="root",
                    key_filename=str(self.ssh_key_path),
                    timeout=60,
                    banner_timeout=60,
                )
                # Keepalive is critical: RunPod's ingress silently drops idle
                # exec channels on multi-hour runs otherwise.
                transport = client.get_transport()
                if transport is not None:
                    transport.set_keepalive(30)
                return client
            except Exception as e:
                last_err = e
                wait = min(2 ** attempt, 10)
                logger.debug(f"SSH connect {attempt + 1}/{retries} failed, retry in {wait}s: {e}")
                client.close()
                time.sleep(wait)
        raise RuntimeError(
            f"Unable to connect to SSH at {endpoint['host']}:{endpoint['port']} "
            f"after {retries} attempts: {last_err}"
        )

    def _exec(self, client, command: str, timeout: int = 3600) -> dict:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return {"stdout": out, "stderr": err, "code": exit_code}

    def run(self, instance_id: str, command: str, retries: int = 4) -> dict:
        """Run a command via SSH with reconnect-on-transient-error.

        The poll-loop callers (get_progress, is_pipeline_running) hit this
        every ~20s for hours; over a long run, paramiko's idle SSH channel
        will eventually be reset by Windows TCP / RunPod ingress / antivirus.
        Without retry, that single reset terminates the whole pod via the
        outer finally block — losing hours of remote quant progress for a
        recoverable network blip.
        """
        paramiko = _ensure_paramiko()
        transient_excs = (
            paramiko.SSHException,
            EOFError,
            OSError,
            ConnectionResetError,
        )
        last_err = None
        for attempt in range(retries):
            try:
                client = self._connect_ssh(instance_id)
            except Exception as e:
                # _connect_ssh has its own retry; if even that gave up,
                # treat it as transient and bubble through the outer loop.
                last_err = e
                wait = min(2 ** attempt, 15)
                logger.warning(
                    f"SSH connect failed (run attempt {attempt + 1}/{retries}, "
                    f"retry in {wait}s): {e}"
                )
                time.sleep(wait)
                continue
            try:
                return self._exec(client, command, timeout=43200)
            except transient_excs as e:
                last_err = e
                if attempt == retries - 1:
                    raise
                wait = min(2 ** attempt, 15)
                logger.warning(
                    f"SSH exec transient error (attempt {attempt + 1}/{retries}, "
                    f"retry in {wait}s): {e}"
                )
                time.sleep(wait)
                # Drop any cached endpoint so the next _connect_ssh re-resolves
                # if the pod's port changed (rare, but defensible).
            finally:
                try:
                    client.close()
                except Exception:
                    pass
        # Loop exhausted without returning — surface the last error.
        raise RuntimeError(
            f"run() exhausted {retries} retries on {instance_id}: {last_err}"
        )

    def _sftp_put_with_retry(self, instance_id: str, fn, retries: int = 5):
        """Run a single SFTP operation with reconnect-on-transient-error.

        `fn(sftp)` does the actual put / write. We open a fresh SSH+SFTP
        per attempt so a stale connection from a previous error doesn't
        carry over.
        """
        paramiko = _ensure_paramiko()
        transient = (paramiko.SSHException, EOFError, OSError, ConnectionResetError)
        last_err = None
        for attempt in range(retries):
            try:
                client = self._connect_ssh(instance_id)
            except Exception as e:
                last_err = e
                if attempt == retries - 1:
                    raise
                wait = min(2 ** attempt, 15)
                logger.warning(f"SFTP connect retry {attempt + 1}/{retries} in {wait}s: {e}")
                time.sleep(wait)
                continue
            try:
                sftp = client.open_sftp()
                try:
                    return fn(sftp)
                finally:
                    sftp.close()
            except transient as e:
                last_err = e
                if attempt == retries - 1:
                    raise
                wait = min(2 ** attempt, 15)
                logger.warning(f"SFTP transient error retry {attempt + 1}/{retries} in {wait}s: {e}")
                time.sleep(wait)
            finally:
                try: client.close()
                except Exception: pass

    def _upload_file(self, instance_id: str, local_path: Path, remote_path: str):
        self._sftp_put_with_retry(
            instance_id,
            lambda sftp: sftp.put(str(local_path), remote_path),
        )

    def _upload_bytes(self, instance_id: str, data: bytes, remote_path: str):
        def _put(sftp):
            with sftp.file(remote_path, "wb") as f:
                f.write(data)
        self._sftp_put_with_retry(instance_id, _put)

    def _upload_directory(self, instance_id: str, local_dir: Path, remote_dir: str):
        """Upload a directory tree via SFTP with per-file retry + reconnect.

        Long SFTP sessions can be killed mid-stream by transient network drops
        (RunPod ingress, local Windows TCP, antivirus). On any SFTP/EOF error
        we reopen the connection and retry just that file, up to 5 times each.
        """
        paramiko = _ensure_paramiko()
        transient_errs = (
            paramiko.SSHException,
            EOFError,
            OSError,
            ConnectionResetError,
        )

        def _open():
            c = self._connect_ssh(instance_id)
            return c, c.open_sftp()

        client, sftp = _open()
        try:
            try:
                sftp.mkdir(remote_dir)
            except IOError:
                pass

            for root, dirs, files in os.walk(local_dir):
                rel_root = Path(root).relative_to(local_dir)
                for d in dirs:
                    remote_path = f"{remote_dir}/{rel_root}/{d}".replace("\\", "/")
                    try:
                        sftp.mkdir(remote_path)
                    except IOError:
                        pass

                for f in files:
                    local_path = Path(root) / f
                    remote_path = f"{remote_dir}/{rel_root}/{f}".replace("\\", "/")
                    for attempt in range(5):
                        try:
                            sftp.put(str(local_path), remote_path)
                            break
                        except transient_errs as e:
                            if attempt == 4:
                                raise
                            wait = min(2 ** attempt, 15)
                            logger.warning(
                                f"SFTP put({remote_path}) failed (attempt {attempt + 1}/5, "
                                f"retry in {wait}s): {e}"
                            )
                            try:
                                sftp.close()
                                client.close()
                            except Exception:
                                pass
                            time.sleep(wait)
                            client, sftp = _open()
        finally:
            try:
                sftp.close()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Bootstrap — idempotent
    # ------------------------------------------------------------------

    def _probe_remote_python(self, instance_id: str) -> str:
        """Discover the python binary inside the pod that has torch importable.

        The base image's torch is in a specific interpreter (often `python` on
        the venv path, not `/usr/bin/python3.11`). We prefer an interpreter with
        torch already installed; if none qualify, fall back to the first python
        found so we can at least install our own stack.
        """
        probe = self.run(
            instance_id,
            # Try candidates in preference order. The one with torch wins.
            # If none have torch, return the first one that exists at all.
            "set -u; FIRST=''; WINNER=''; "
            "for p in python python3 python3.12 python3.11 python3.10 python3.13; do "
            "  command -v $p >/dev/null 2>&1 || continue; "
            "  [ -z \"$FIRST\" ] && FIRST=$p; "
            "  if $p -c 'import torch' >/dev/null 2>&1; then WINNER=$p; break; fi; "
            "done; "
            "echo ${WINNER:-$FIRST}",
        )
        py = probe["stdout"].strip() or "python3"
        self._remote_py = py
        return py

    def bootstrap(self, instance_id: str, exllamav3_local_dir: Path | None = None) -> bool:
        # Pre-baked image short-circuit — every dep is in the image already
        # and /opt/blockquant/quant.py is in place. Just run the health check
        # and we're done.
        if self.image and "blockquant" in self.image.lower():
            py = self._probe_remote_python(instance_id)
            logger.info(f"Pre-baked image detected ({self.image}); skipping bootstrap")
            print(f"      pre-baked image: {self.image}", flush=True)
            health = self.run(
                instance_id,
                f"{py} -c \"import torch, exllamav3; "
                "print('[gpu]', torch.cuda.get_device_name(0)); "
                "print('[ok] exllamav3', exllamav3.__version__)\"",
            )
            if health["code"] != 0:
                logger.error(f"Pre-baked health check failed: {health['stderr'][:2000]}")
                return False
            logger.info(f"Health check output:\n{health['stdout'].strip()}")
            return True

        # Always probe so run_pipeline has the right interpreter, even on fast-path.
        py = self._probe_remote_python(instance_id)
        # Note whether the selected interpreter already has torch so it's visible in logs.
        torch_check = self.run(
            instance_id,
            f"{py} -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING",
        )
        torch_ver = torch_check["stdout"].strip()
        logger.info(f"Using remote interpreter: {py}  (torch: {torch_ver})")
        print(f"      remote python: {py}  torch: {torch_ver}", flush=True)

        # Fast-path: if a prior run on this pod already bootstrapped, skip.
        check = self.run(instance_id, f"test -f {BOOTSTRAP_MARKER} && echo yes || echo no")
        if check["stdout"].strip() == "yes":
            logger.info("Pod already bootstrapped")
            return True

        # The base image ships torch + CUDA — only add what's missing.
        # ninja-build + build-essential are required for exllamav3's JIT C++ extension.
        logger.info("Installing system packages...")
        result = self.run(
            instance_id,
            "apt-get update && apt-get install -y --no-install-recommends "
            "git wget ca-certificates ninja-build build-essential",
        )
        if result["code"] != 0:
            logger.error(f"System bootstrap failed: {result['stderr'][:2000]}")
            return False

        result = self.run(instance_id, f"{py} -m pip install --upgrade pip")
        if result["code"] != 0:
            logger.warning(f"pip upgrade warning: {result['stderr'][:200]}")

        # If the selected interpreter has no torch, install it against the image's CUDA.
        if torch_ver in ("", "MISSING"):
            logger.info("No torch on selected interpreter — installing torch==2.6 cu124...")
            print("      installing torch==2.6 (base image has no torch on this python)", flush=True)
            result = self.run(
                instance_id,
                f"{py} -m pip install --no-cache-dir torch==2.6.0 "
                f"--index-url https://download.pytorch.org/whl/cu124",
            )
            if result["code"] != 0:
                logger.error(f"torch install failed: {result['stderr'][:2000]}")
                return False

        # huggingface_hub is needed early by the remote quant.py to download the model,
        # and transformers/sentencepiece are used across the stack. These overlap with
        # exllamav3's own requirements.txt but we install them first so a pre-upload
        # failure is cheap to diagnose.
        logger.info("Installing HF deps...")
        result = self.run(
            instance_id,
            f"{py} -m pip install --no-cache-dir --upgrade "
            "huggingface-hub transformers sentencepiece tqdm psutil",
        )
        if result["code"] != 0:
            logger.error(f"HF deps install failed: {result['stderr'][:2000]}")
            return False

        if self.install_flash_attn:
            logger.info("Installing flash-attn (falls back to SDPA if build fails)...")
            # Let flash-attn pip-resolve the wheel that matches the image's torch.
            result = self.run(
                instance_id,
                f"{py} -m pip install flash-attn --no-build-isolation",
            )
            if result["code"] != 0:
                stderr_preview = result["stderr"][:300].encode("ascii", "replace").decode("ascii")
                logger.warning(f"flash-attn install failed (will use SDPA fallback): {stderr_preview}")
        else:
            logger.info("Skipping flash-attn (SDPA is adequate for quantization calibration)")

        if exllamav3_local_dir and exllamav3_local_dir.exists():
            logger.info(f"Uploading local exllamav3 from {exllamav3_local_dir}...")
            remote_dir = "/workspace/exllamav3"
            self._upload_directory(instance_id, exllamav3_local_dir, remote_dir)
            # Install upstream requirements minus flash_attn (we own that path
            # separately via install_flash_attn). This picks up marisa_trie,
            # rich, typing_extensions, pillow, pyyaml, and the exact
            # formatron/pydantic/kbnf pins the code was tested against.
            # Exclude torch AND flash_attn from the upstream upgrade.
            # - torch: the base image's torch (e.g. 2.8.0+cu129) is matched to the
            #   pod's NVIDIA driver. Letting pip pull torch>=2.6.0 from PyPI ends
            #   up installing a cu13x build that the older driver rejects.
            # - flash_attn: opt-in via install_flash_attn; SDPA is fine for quant.
            logger.info("Installing exllamav3 requirements.txt (excluding torch + flash_attn)...")
            result = self.run(
                instance_id,
                f"cd {remote_dir} && "
                f"grep -v -E '^(flash_attn|flash-attn|torch)' requirements.txt > /tmp/exl3-reqs.txt && "
                f"{py} -m pip install --no-cache-dir --upgrade -r /tmp/exl3-reqs.txt",
            )
            if result["code"] != 0:
                logger.error(f"exllamav3 requirements install failed: {result['stderr'][:2000]}")
                return False
            result = self.run(
                instance_id,
                f"cd {remote_dir} && {py} -m pip install -e . --no-deps",
            )
        else:
            logger.info("Installing exllamav3 from PyPI (with deps)...")
            result = self.run(instance_id, f"{py} -m pip install --upgrade exllamav3")

        # Formatron is chronically broken against current pydantic in many base images,
        # and the exllamav3 top-level __init__ hard-imports FormatronFilter even for
        # quant-only workflows that never touch generation. Test whether the import
        # works; if not, neuter it with a stub so conversion still imports.
        probe_formatron = self.run(
            instance_id,
            f"{py} -c 'from formatron.formatter import FormatterBuilder' 2>&1 || echo __FORMATRON_BROKEN__",
        )
        if "__FORMATRON_BROKEN__" in probe_formatron["stdout"]:
            logger.warning("formatron import broken; stubbing FormatronFilter for quant-only use")
            formatron_shim = (
                "class FormatronFilter:\n"
                "    def __init__(self, *a, **kw):\n"
                "        raise RuntimeError('FormatronFilter stubbed out (formatron broken in env)')\n"
            )
            # Find the exllamav3 filter module regardless of local vs PyPI install.
            locate = self.run(
                instance_id,
                f"{py} -c 'import exllamav3, os; "
                "p = os.path.join(os.path.dirname(exllamav3.__file__), \"generator\", \"filter\", \"formatron.py\"); "
                "print(p)' 2>/dev/null || true",
            )
            remote_formatron = locate["stdout"].strip().splitlines()
            remote_formatron = remote_formatron[-1] if remote_formatron else ""
            if not remote_formatron:
                # Fall back to the known local-upload path.
                remote_formatron = "/workspace/exllamav3/exllamav3/generator/filter/formatron.py"
            self._upload_bytes(
                instance_id, formatron_shim.encode("utf-8"), remote_formatron
            )
            logger.info(f"Stubbed {remote_formatron}")

        if result["code"] != 0:
            logger.error(f"exllamav3 install failed: {result['stderr'][:2000]}")
            return False

        logger.info("Health check...")
        result = self.run(
            instance_id,
            f"{py} -c \"import torch; print('[gpu]', torch.cuda.get_device_name(0)); "
            "import exllamav3; print('[ok] exllamav3 imported')\"",
        )
        if result["code"] != 0:
            logger.error(f"Health check failed: {result['stderr'][:2000]}")
            return False
        logger.info(f"Health check output:\n{result['stdout'].strip()}")

        # Mark bootstrapped so a re-run on the same pod is cheap.
        self.run(instance_id, f"touch {BOOTSTRAP_MARKER}")
        logger.info("Bootstrap complete")
        return True

    # ------------------------------------------------------------------
    # Pipeline — fire-and-forget; progress via tail-log
    # ------------------------------------------------------------------

    # Single source of truth: the remote quant script lives at
    # backend/src/blockquant/remote/quant.py and is read at import time.
    # Pre-baked Docker images bake the same file at /opt/blockquant/quant.py
    # so we don't have to SFTP it on every run when using those.
    _REMOTE_QUANT_PATH = Path(__file__).resolve().parents[1] / "remote" / "quant.py"
    _BAKED_REMOTE_QUANT = "/opt/blockquant/quant.py"

    @classmethod
    def _load_quant_script(cls) -> bytes:
        """Read the canonical remote quant script as bytes."""
        return cls._REMOTE_QUANT_PATH.read_bytes()

    def run_pipeline(
        self,
        instance_id: str,
        model_id: str,
        format: str,
        variants: list[str],
        hf_token: str = "",
        hf_org: str = "",
        head_bits: int = 8,
        use_imatrix: bool = True,
        cal_rows: int | None = None,
        cal_cols: int | None = None,
        keep_pod: bool = False,
    ) -> dict:
        """Start the remote quant script in the background. Returns immediately.

        ``cal_rows``/``cal_cols`` override ExLlamaV3's calibration defaults
        (250 × 2048). Lower values trade quality for speed; see PROFILES.
        """
        # Reset any cached result from a prior call.
        self._last_result = None

        # 1. Config JSON (keeps secrets out of command lines / logs).
        # pod_id + runpod_api_key + keep_pod let the in-pod script
        # self-terminate after [done] without depending on the local
        # poll loop being alive — see remote/quant.py:_self_terminate.
        cfg: dict = {
            "model_id": model_id,
            "variants": list(variants),
            "hf_token": hf_token,
            "hf_org": hf_org,
            "head_bits": head_bits,
            "pod_id": instance_id,
            "runpod_api_key": _ensure_runpod().api_key or "",
            "keep_pod": bool(keep_pod),
        }
        if cal_rows is not None:
            cfg["cal_rows"] = int(cal_rows)
        if cal_cols is not None:
            cfg["cal_cols"] = int(cal_cols)
        self._upload_bytes(instance_id, json.dumps(cfg).encode("utf-8"), "/root/bq-config.json")

        # 2. Locate the remote quant script. On pre-baked images it lives
        # at /opt/blockquant/quant.py; otherwise we SFTP the canonical
        # source from `remote/quant.py`.
        check = self.run(
            instance_id,
            f"test -f {self._BAKED_REMOTE_QUANT} && echo BAKED || echo MISSING",
        )
        if "BAKED" in check["stdout"]:
            script_path = self._BAKED_REMOTE_QUANT
            logger.info(f"Using pre-baked quant script at {script_path}")
        else:
            script_path = REMOTE_SCRIPT
            self._upload_bytes(instance_id, self._load_quant_script(), script_path)
            self.run(instance_id, f"chmod +x {script_path}")

        # 3. Fire-and-forget. setsid so the process survives our SSH channel.
        launch_cmd = (
            f"rm -f {REMOTE_RESULT} {REMOTE_LOG} && "
            f"nohup setsid {self._remote_py} {script_path} "
            f"> {REMOTE_LOG} 2>&1 < /dev/null & echo $!"
        )
        result = self.run(instance_id, launch_cmd)
        if result["code"] != 0:
            return {"status": "failed", "error": f"launch failed: {result['stderr'][:2000]}"}
        logger.info(f"Remote pipeline started on {instance_id} (pid={result['stdout'].strip()})")
        return {"status": "started"}

    def get_progress(self, instance_id: str, lines: int = 30) -> str:
        """Return the tail of the remote log for progress reporting.

        ``lines`` defaults to 30 for cheap routine polling; pass a larger
        value (e.g. 500) for a final drain after the run completes so the
        last batch of post-quantize + upload output is captured locally.
        """
        result = self.run(
            instance_id,
            f"tail -n {int(lines)} {REMOTE_LOG} 2>/dev/null || echo NO_LOG",
        )
        return result["stdout"]

    def is_pipeline_running(self, instance_id: str) -> bool:
        """True while the remote quant.py process is alive."""
        result = self.run(
            instance_id,
            f"pgrep -f '{REMOTE_SCRIPT}' >/dev/null && echo running || echo done",
        )
        return result["stdout"].strip() == "running"

    def get_result(self) -> dict | None:
        """Return the parsed /root/bq-result.json from the most recent run."""
        if self._last_result is not None:
            return self._last_result
        if self._pod_id is None:
            return None
        try:
            fetched = self.run(
                self._pod_id,
                f"cat {REMOTE_RESULT} 2>/dev/null || echo ''",
            )
            raw = fetched["stdout"].strip()
            if not raw:
                return None
            self._last_result = json.loads(raw)
            return self._last_result
        except Exception as e:
            logger.warning(f"get_result() failed: {e}")
            return None

    def sync_outputs(self, instance_id: str, local_dir: Path, remote_rel_path: str = "") -> list[Path]:
        # The remote script uploads directly to HuggingFace — nothing to sync.
        logger.info("RunPod: outputs uploaded to HF by remote script, skipping local sync")
        return []

    # ------------------------------------------------------------------
    # Cost — dynamic via SDK with hardcoded fallback
    # ------------------------------------------------------------------

    _STATIC_PRICES = {
        "NVIDIA RTX A4000": 0.32,
        "NVIDIA RTX A4500": 0.44,
        "NVIDIA RTX A5000": 0.46,
        "NVIDIA RTX A6000": 0.79,
        "NVIDIA A40": 0.47,
        "NVIDIA A100 80GB PCIe": 1.64,
        "NVIDIA A100-SXM4-80GB": 1.89,
        "NVIDIA H100 PCIe": 2.39,
        "NVIDIA H100 80GB HBM3": 1.99,
        "NVIDIA H100 NVL": 2.79,
        "NVIDIA H200": 3.99,
    }

    def _lookup_live_price(self) -> float | None:
        """Live-fetch the price for self.gpu_type. Cached per instance.

        Picks the price tier matching ``self.cloud_type`` so the CLI's
        ``--tune`` mode reports honest numbers when SECURE is selected.
        """
        if self._gpu_price_cache is not None:
            return self._gpu_price_cache.get(self.gpu_type)
        rp = _ensure_runpod()
        self._gpu_price_cache = {}
        try:
            # Only get_gpu(id) returns pricing; get_gpus() is list-view only.
            g = rp.get_gpu(self.gpu_type)
            if g:
                if self.cloud_type == "SECURE":
                    price = g.get("securePrice") or g.get("communityPrice")
                else:
                    price = g.get("communityPrice") or g.get("securePrice")
                if not price:
                    lp = g.get("lowestPrice")
                    if isinstance(lp, dict):
                        price = lp.get("uninterruptablePrice")
                if isinstance(price, (int, float)) and price > 0:
                    self._gpu_price_cache[self.gpu_type] = float(price)
                    return float(price)
        except Exception as e:
            logger.debug(f"Live GPU price lookup failed ({e}); using static table")
        return None

    def get_cost_per_hour(self) -> float:
        live = self._lookup_live_price()
        if live is not None:
            return live
        return self._STATIC_PRICES.get(self.gpu_type, 2.00)
