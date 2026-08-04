"""Runtime configuration for exliberate.

Mirrors heretic's settings style: a single pydantic-settings ``Settings``
class fed by (lowest → highest precedence):

1. dataclass defaults,
2. a TOML file (``exliberate.toml`` in CWD, or the path in the
   ``EXLIBERATE_CONFIG`` environment variable),
3. environment variables prefixed with ``EXLIBERATE_``,
4. CLI flags (only when constructed via :meth:`Settings.from_cli`),
5. explicit constructor keyword arguments.

The class deliberately does *not* parse ``sys.argv`` on plain construction so
that library users (and pytest) can instantiate ``Settings(...)`` safely.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Type

import torch
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _default_toml_file() -> str:
    """TOML config path: ``$EXLIBERATE_CONFIG`` or ``./exliberate.toml``."""
    return os.environ.get("EXLIBERATE_CONFIG", "exliberate.toml")


class Settings(BaseSettings):
    """exliberate runtime settings (TOML + env + CLI sources)."""

    model_config = SettingsConfigDict(
        env_prefix="EXLIBERATE_",
        toml_file=_default_toml_file(),
        extra="ignore",
    )

    # --- model / hardware -------------------------------------------------
    model_id: str = Field(
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        description="Hugging Face model ID or local path.",
    )
    dtype: str = Field(
        default="auto",
        description="auto | float32 | float16 | bfloat16. auto = float32 on CPU, "
        "bfloat16 on CUDA.",
    )
    device: str = Field(
        default="auto",
        description="auto | cpu | cuda | cuda:N. auto = cuda if available else cpu.",
    )
    batch_size: int = Field(default=4, ge=1, description="Prompts per forward batch.")
    trust_remote_code: bool = Field(
        default=False,
        description="Allow models that ship custom modeling code (HF trust_remote_code).",
    )
    seed: int = Field(default=42, description="Global RNG seed (torch/numpy/optuna).")

    # --- math -------------------------------------------------------------
    winsorization_quantile: float = Field(
        default=0.95,
        gt=0.5,
        le=1.0,
        description="Upper quantile for winsorizing residuals (lower = 1 - q).",
    )

    # --- search / generation ----------------------------------------------
    n_trials: int = Field(default=40, ge=1, description="Optuna trials per round.")
    max_new_tokens: int = Field(
        default=64, ge=1, description="Max tokens generated per scored response."
    )
    output_dir: Path = Field(
        default=Path("exliberate-output"),
        description="Where merged models / reports are written.",
    )

    # ------------------------------------------------------------------ api
    def resolve_device(self) -> torch.device:
        """Resolve ``device='auto'`` to a concrete :class:`torch.device`.

        A silent CPU fallback is the single most expensive failure mode here:
        the search still runs, just orders of magnitude slower, and nothing in
        the output says so. Containers make this easy to hit -- a mismatched
        CUDA forward-compat driver makes ``is_available()`` False on a machine
        that plainly has a GPU. Say so, loudly, once.
        """
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        import warnings
        detail = ""
        try:
            if torch.cuda.device_count() > 0:
                detail = " (a CUDA device is present but unusable -- driver/runtime mismatch?)"
        except Exception as exc:
            detail = f" ({exc})"
        warnings.warn(
            "exliberate: CUDA is not available, running on CPU" + detail
            + ". The refusal search will be far slower and --quantize cannot "
            "run. Set device='cpu' explicitly to silence this.",
            RuntimeWarning, stacklevel=2,
        )
        return torch.device("cpu")

    def resolve_dtype(self) -> torch.dtype:
        """Resolve ``dtype='auto'``: float32 on CPU, bfloat16 on CUDA."""
        if self.dtype != "auto":
            mapping = {
                "float32": torch.float32,
                "float": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            try:
                return mapping[self.dtype.lower()]
            except KeyError:
                raise ValueError(f"Unknown dtype: {self.dtype!r}") from None
        return (
            torch.bfloat16
            if self.resolve_device().type == "cuda"
            else torch.float32
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: init kwargs > env > TOML file. TomlConfigSettingsSource
        # silently yields nothing when the file does not exist.
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_cli(cls, argv: list[str] | None = None, **overrides: Any) -> "Settings":
        """Construct settings parsing CLI flags (``--model_id ...`` style —
        pydantic-settings uses field names verbatim, underscores not dashes).

        Kept separate from plain construction so importing/instantiating
        ``Settings`` inside libraries and tests never touches ``sys.argv``.
        """
        if argv is None:
            return cls(_cli_parse_args=True, **overrides)
        return cls(_cli_parse_args=argv, **overrides)
