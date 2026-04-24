"""Stage 2: Convert to FP16 GGUF (GGUF only)."""
import subprocess
import sys
from pathlib import Path

from blockquant.models import QuantConfig, QuantFormat
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)


def run(config: QuantConfig, workspace: Path) -> None:
    """Convert HF model to FP16 GGUF."""
    if config.format != QuantFormat.GGUF:
        return  # EXL3 uses HF format directly

    f16_path = workspace / "model.f16.gguf"
    if f16_path.exists():
        return

    # Try llama.cpp convert script
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        raise FileNotFoundError(
            "llama.cpp/convert_hf_to_gguf.py not found. Run: git clone https://github.com/ggerganov/llama.cpp"
        )

    logger.info("Converting to FP16 GGUF...")
    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(workspace / "model"),
            "--outfile",
            str(f16_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr[:500]}")
