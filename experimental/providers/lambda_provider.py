"""Lambda Cloud provider.

Usage:
    1. Sign up at https://lambdalabs.com and get an API key.
    2. Add your SSH public key to the Lambda dashboard.
    3. Set LAMBDA_API_KEY env var or pass api_key to LambdaProvider.

The provider uses SSH (paramiko) to execute commands on Lambda instances.
"""
import os
import time
from pathlib import Path

import paramiko
import requests

from blockquant.providers.base import Provider
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)

LAMBDA_API = "https://cloud.lambdalabs.com/api/v1"
DEFAULT_INSTANCE_TYPE = "gpu_1x_a100_sxm4"
DEFAULT_REGION = "us-east-1"
BOOTSTRAP_MARKER = "/opt/.blockquant-bootstrapped"


class LambdaProvider(Provider):
    name = "lambda"

    def __init__(self, api_key: str | None = None, ssh_key_path: str | None = None):
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Lambda API key required. Set LAMBDA_API_KEY env var or pass api_key."
            )
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.ssh_key_path = ssh_key_path or os.path.expanduser("~/.ssh/id_rsa")

    # ------------------------------------------------------------------
    # HTTP API helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        resp = requests.get(f"{LAMBDA_API}{path}", headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_body: dict) -> dict:
        resp = requests.post(f"{LAMBDA_API}{path}", headers=self.headers, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    def check_capacity(self, instance_type: str, region: str) -> bool:
        """Check if the given instance type is available in the region."""
        data = self._get("/instance-types")
        types = data.get("data", {})
        info = types.get(instance_type)
        if not info:
            logger.warning(f"Unknown instance type: {instance_type}")
            return False
        for r in info.get("regions", []):
            if r.get("region_name") == region and r.get("available"):
                return True
        return False

    def launch(self, config: dict) -> str:
        """Launch a Lambda instance and return its ID."""
        region = config.get("region", DEFAULT_REGION)
        instance_type = config.get("instance_type", DEFAULT_INSTANCE_TYPE)
        if not self.check_capacity(instance_type, region):
            raise RuntimeError(
                f"No Lambda capacity for {instance_type} in {region}. "
                "Check https://lambdalabs.com/service/gpu-cloud or try a different region/type."
            )
        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": config.get("ssh_keys", []),
            "quantity": 1,
        }
        logger.info(f"Launching Lambda instance: {payload['instance_type_name']} in {payload['region_name']}")
        data = self._post("/instance-operations/launch", payload)
        instance_ids = data.get("data", {}).get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError("Lambda launch returned no instance IDs")
        instance_id = instance_ids[0]
        logger.info(f"Lambda instance launched: {instance_id}")
        return instance_id

    def terminate(self, instance_id: str):
        """Terminate a Lambda instance."""
        logger.info(f"Terminating Lambda instance: {instance_id}")
        try:
            self._post("/instance-operations/terminate", {"instance_ids": [instance_id]})
            logger.info(f"Lambda instance terminated: {instance_id}")
        except Exception as e:
            logger.warning(f"Lambda terminate failed: {e}")

    def get_status(self, instance_id: str) -> dict:
        """Get instance status and details."""
        data = self._get(f"/instances/{instance_id}")
        return data.get("data", {})

    def wait_for_active(self, instance_id: str, timeout: int = 600, interval: int = 10) -> dict:
        """Poll until instance is active. Returns instance details."""
        logger.info(f"Waiting for Lambda instance {instance_id} to become active...")
        start = time.time()
        while time.time() - start < timeout:
            info = self.get_status(instance_id)
            status = info.get("status", "unknown")
            logger.info(f"  Instance {instance_id} status: {status}")
            if status == "active":
                return info
            if status in ("terminating", "terminated", "error"):
                raise RuntimeError(f"Lambda instance {instance_id} entered status: {status}")
            time.sleep(interval)
        raise TimeoutError(f"Lambda instance {instance_id} did not become active within {timeout}s")

    # ------------------------------------------------------------------
    # SSH helpers
    # ------------------------------------------------------------------

    def _ssh_client(self, instance_id: str) -> paramiko.SSHClient:
        """Create and connect an SSH client to the instance."""
        info = self.get_status(instance_id)
        ip = info.get("ip")
        if not ip:
            raise RuntimeError(f"Instance {instance_id} has no IP address")

        key_path = Path(self.ssh_key_path)
        if not key_path.exists():
            raise FileNotFoundError(
                f"SSH key not found: {key_path}\n"
                "Add your SSH public key to the Lambda dashboard."
            )

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ip,
            username="ubuntu",
            key_filename=str(key_path),
            timeout=30,
            banner_timeout=30,
        )
        return client

    def _exec(self, client: paramiko.SSHClient, command: str, timeout: int = 3600) -> dict:
        """Execute a command over an existing SSH connection."""
        logger.debug(f"SSH exec: {command[:100]}...")
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        return {"stdout": out, "stderr": err, "code": code}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap(self, instance_id: str) -> bool:
        """Run one-time setup on the instance. Idempotent — skips if marker exists."""
        client = self._ssh_client(instance_id)
        try:
            # Check if already bootstrapped
            check = self._exec(client, f"test -f {BOOTSTRAP_MARKER} && echo yes || echo no", timeout=10)
            if check["stdout"].strip() == "yes":
                logger.info("Instance already bootstrapped")
                return True

            logger.info("Bootstrapping Lambda instance...")
            script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "bootstrap_lambda.sh"
            if not script_path.exists():
                raise FileNotFoundError(f"Bootstrap script not found: {script_path}")

            script = script_path.read_text(encoding="utf-8")
            # Upload and execute script
            sftp = client.open_sftp()
            remote_script = "/tmp/bootstrap_lambda.sh"
            with sftp.file(remote_script, "w") as f:
                f.write(script)
            sftp.chmod(remote_script, 0o755)
            sftp.close()

            result = self._exec(client, f"bash {remote_script}", timeout=600)
            if result["code"] != 0:
                logger.error(f"Bootstrap failed:\n{result['stderr']}")
                raise RuntimeError(f"Bootstrap failed: {result['stderr'][:500]}")

            # Mark as bootstrapped
            self._exec(client, f"sudo touch {BOOTSTRAP_MARKER}", timeout=10)
            logger.info("Bootstrap complete")
            return True
        finally:
            client.close()

    # ------------------------------------------------------------------
    # Remote execution with progress streaming
    # ------------------------------------------------------------------

    def run(self, instance_id: str, command: str) -> dict:
        """Execute a raw command via SSH (required by Provider base class)."""
        client = self._ssh_client(instance_id)
        try:
            return self._exec(client, command)
        finally:
            client.close()

    def run_pipeline(
        self,
        instance_id: str,
        model_id: str,
        format: str,
        variants: list[str],
        hf_token: str = "",
        hf_org: str = "",
        head_bits: int = 8,
    ) -> dict:
        """Start the pipeline remotely and return the background job info."""
        client = self._ssh_client(instance_id)
        try:
            variants_str = ",".join(variants)
            env = f"export HF_TOKEN='{hf_token}' && export EXLLAMAV3_DIR=/opt/exllamav3"
            cmd = (
                f"cd /opt/blockquant/backend && {env} && "
                f"nohup .venv/bin/python -m blockquant.pipeline "
                f"--model {model_id} --format {format} --variants {variants_str} "
                f"--head_bits {head_bits} "
                f"> /tmp/bq-pipeline.log 2>&1 &"
            )
            # Note: using system python3 directly since venv may not exist on remote
            # The bootstrap script installs to system pip3
            cmd = (
                f"cd /opt/blockquant/backend && {env} && "
                f"nohup python3 -m blockquant.pipeline "
                f"--model {model_id} --format {format} --variants {variants_str} "
                f"--head_bits {head_bits} "
                f"> /tmp/bq-pipeline.log 2>&1 &"
            )

            result = self._exec(client, cmd, timeout=30)
            logger.info(f"Remote pipeline started on {instance_id}")
            return {"code": result["code"], "stdout": result["stdout"], "stderr": result["stderr"]}
        finally:
            client.close()

    def get_progress(self, instance_id: str) -> str:
        """Tail the remote progress log."""
        client = self._ssh_client(instance_id)
        try:
            result = self._exec(client, "tail -n 20 /tmp/bq-pipeline.log 2>/dev/null || echo 'NO_LOG'", timeout=10)
            return result["stdout"]
        finally:
            client.close()

    def is_pipeline_running(self, instance_id: str) -> bool:
        """Check if the pipeline process is still active."""
        client = self._ssh_client(instance_id)
        try:
            result = self._exec(client, "pgrep -f 'blockquant.pipeline' > /dev/null && echo running || echo done", timeout=10)
            return result["stdout"].strip() == "running"
        finally:
            client.close()

    # ------------------------------------------------------------------
    # Sync outputs back
    # ------------------------------------------------------------------

    def sync_outputs(self, instance_id: str, local_dir: Path, remote_rel_path: str = "") -> list[Path]:
        """SCP quantized outputs from Lambda to local workspace."""
        client = self._ssh_client(instance_id)
        try:
            sftp = client.open_sftp()
            remote_base = f"/tmp/blockquant-work/{remote_rel_path}"
            downloaded: list[Path] = []

            try:
                sftp.stat(remote_base)
            except FileNotFoundError:
                logger.warning(f"Remote output dir not found: {remote_base}")
                return downloaded

            # Simple recursive download of .safetensors and .json files
            def download_recursive(remote_path: str, local_path: Path):
                sftp.mkdir(str(local_path)) if not local_path.exists() else None
                for item in sftp.listdir(remote_path):
                    r_item = f"{remote_path}/{item}"
                    l_item = local_path / item
                    try:
                        sftp.stat(r_item)
                        # If it's a file, download it
                        if item.endswith((".safetensors", ".json", ".md")):
                            sftp.get(r_item, str(l_item))
                            downloaded.append(l_item)
                    except IOError:
                        # Directory
                        download_recursive(r_item, l_item)

            download_recursive(remote_base, local_dir)
            sftp.close()
            logger.info(f"Downloaded {len(downloaded)} files from Lambda")
            return downloaded
        finally:
            client.close()

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def get_cost_per_hour(self) -> float:
        """Return approximate cost per hour for the default instance type."""
        return 1.10
