"""Provider abstract base class.

Defines the full surface area that ``blockquant.pipeline`` expects from
any cloud provider. Methods that don't apply to a given backend (e.g. a
LocalProvider doesn't need to bootstrap) default to sensible no-ops so
the pipeline can call them unconditionally without ``hasattr`` checks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Provider(ABC):
    name: str = ""

    # ------------------------------------------------------------------
    # Lifecycle — every backend must implement these three.
    # ------------------------------------------------------------------

    @abstractmethod
    def launch(self, config: dict) -> str:
        """Launch the compute instance. Returns an instance_id."""
        ...

    @abstractmethod
    def terminate(self, instance_id: str) -> None:
        """Release the instance. Must be safe to call on a dead instance."""
        ...

    @abstractmethod
    def run(self, instance_id: str, command: str) -> dict:
        """Execute a shell command on the instance.

        Returns a dict with keys ``stdout``, ``stderr``, ``code``.
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks — defaults are safe no-ops so the pipeline can call
    # them unconditionally regardless of backend.
    # ------------------------------------------------------------------

    def wait_for_active(self, instance_id: str) -> dict:
        """Block until the instance is reachable. Default: assume ready."""
        return {"status": "active", "id": instance_id}

    def bootstrap(self, instance_id: str) -> bool:
        """Install dependencies on the instance. Default: nothing to do."""
        return True

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
        """Kick off the remote quantization. Default: not supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support run_pipeline()"
        )

    def get_progress(self, instance_id: str) -> str:
        """Return the tail of the remote progress log. Default: empty."""
        return ""

    def is_pipeline_running(self, instance_id: str) -> bool:
        """True while the remote pipeline process is alive. Default: False."""
        return False

    def sync_outputs(
        self,
        instance_id: str,
        local_dir: Path,
        remote_rel_path: str = "",
    ) -> list[Path]:
        """Pull outputs back to the local workspace. Default: no-op."""
        return []

    def get_result(self) -> dict | None:
        """Return parsed result metadata from the most recent run_pipeline.

        Default: no cached result.
        """
        return None

    def get_cost_per_hour(self) -> float:
        """Hourly USD rate for billing / cost estimates. Default: 0.0."""
        return 0.0
