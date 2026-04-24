"""Modal.com serverless GPU provider.

Usage:
    1. Sign up at https://modal.com
    2. Create an API token (Settings → API tokens)
    3. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars
    4. Use provider="modal" in your QuantConfig

The provider calls a deployed Modal GPU function that runs a self-contained
pipeline (download → quantize → upload).  No local code is mounted;
the function installs everything it needs inside Modal's container.
"""
import os
import time
from pathlib import Path

from blockquant.providers.base import Provider
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports so this module loads without Modal SDK installed.
_modal = None


def _ensure_modal():
    global _modal
    if _modal is None:
        import modal

        _modal = modal
    return _modal


class ModalProvider(Provider):
    name = "modal"

    def __init__(
        self,
        token_id: str = "",
        token_secret: str = "",
        gpu: str = "A10G",
    ):
        tid = token_id or os.environ.get("MODAL_TOKEN_ID", "")
        tsec = token_secret or os.environ.get("MODAL_TOKEN_SECRET", "")
        if not tid or not tsec:
            raise ValueError(
                "Modal credentials required. Set MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET env vars or pass token_id/token_secret."
            )
        os.environ["MODAL_TOKEN_ID"] = tid
        os.environ["MODAL_TOKEN_SECRET"] = tsec
        self.gpu = gpu
        self._call_id: str | None = None
        self._last_result: dict | None = None
        self._modal = _ensure_modal()

    # ------------------------------------------------------------------
    # Lifecycle (serverless — mostly no-ops)
    # ------------------------------------------------------------------

    def launch(self, config: dict) -> str:
        logger.info(f"Modal provider ready (GPU: {self.gpu})")
        return "modal-serverless"

    def terminate(self, instance_id: str):
        if self._call_id:
            try:
                call = self._modal.FunctionCall.from_id(self._call_id)
                call.cancel()
                logger.info(f"Modal call cancelled: {self._call_id}")
            except Exception as e:
                logger.warning(f"Modal cancel failed: {e}")

    def wait_for_active(self, instance_id: str, timeout: int = 60, interval: int = 5) -> dict:
        return {"status": "active", "id": instance_id}

    def bootstrap(self, instance_id: str) -> bool:
        return True

    def run(self, instance_id: str, command: str) -> dict:
        raise NotImplementedError("ModalProvider uses run_pipeline(), not run()")

    # ------------------------------------------------------------------
    # Remote execution
    # ------------------------------------------------------------------

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
    ) -> dict:
        config_dict = {
            "model_id": model_id,
            "format": format,
            "variants": variants,
            "hf_token": hf_token,
            "hf_org": hf_org,
            "head_bits": head_bits,
            "use_imatrix": use_imatrix,
        }
        logger.info(f"Spawning Modal function for {model_id} ({self.gpu})")
        fn = self._modal.Function.from_name("blockquant", "_run_pipeline_remote")
        function_call = fn.spawn(config_dict)
        self._call_id = function_call.object_id
        logger.info(f"Modal call spawned: {self._call_id}")
        return {"call_id": self._call_id, "status": "spawned"}

    def get_progress(self, instance_id: str) -> str:
        if not self._call_id:
            return "NO_CALL"
        try:
            call = self._modal.FunctionCall.from_id(self._call_id)
            try:
                # timeout=0 returns immediately if result is cached locally;
                # otherwise it fetches from server. Use a short positive
                # timeout so a finished call can be retrieved.
                self._last_result = call.get(timeout=1)
                status = self._last_result.get("status", "unknown")
                return f"COMPLETE status={status}"
            except self._modal.exception.TimeoutError:
                return "RUNNING"
            except TimeoutError:
                # Python's built-in TimeoutError (raised by some Modal paths)
                return "RUNNING"
            except Exception as e:
                return f"ERROR: {type(e).__name__}: {e}"
        except Exception as e:
            return f"STATUS_ERROR: {type(e).__name__}: {e}"

    def is_pipeline_running(self, instance_id: str) -> bool:
        return "RUNNING" in self.get_progress(instance_id)

    def sync_outputs(
        self,
        instance_id: str,
        local_dir: Path,
        remote_rel_path: str = "",
    ) -> list[Path]:
        logger.info("Modal: outputs uploaded to HF by remote function, skipping local sync")
        return []

    def get_result(self) -> dict | None:
        """Return the dict returned by the Modal function (if finished)."""
        return self._last_result

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def get_cost_per_hour(self) -> float:
        costs = {"A10G": 1.10, "A100": 3.50, "A100-80GB": 4.25, "H100": 4.50}
        return costs.get(self.gpu, 3.50)
