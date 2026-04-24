"""Stage 4b: Quality metrics — KL divergence + Perplexity for EXL3 outputs.

Inspired by ezexl3's measurement pipeline. Runs exllamav3 eval/model_diff.py
to compare the base model against each quantized output.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

from blockquant.models import QuantConfig, QuantFormat, QuantOutput
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)

_KL_RE = re.compile(
    r"KL\s+divergence\s+\(\w+,\s*\w+\):\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
    re.IGNORECASE,
)
_PPL_RE = re.compile(
    r"perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
    re.IGNORECASE,
)


def _find_model_diff_script() -> Path:
    """Locate exllamav3 eval/model_diff.py."""
    exllama_dir = Path(os.environ.get("EXLLAMAV3_DIR", "exllamav3"))
    candidates = [
        exllama_dir / "eval" / "model_diff.py",
        Path("exllamav3/eval/model_diff.py"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("exllamav3 eval/model_diff.py not found. Set EXLLAMAV3_DIR.")


def _run_model_diff(base_dir: Path, quant_dir: Path, rows: int = 100, device: int = 0) -> dict:
    """Run model_diff.py and parse KL divergence + perplexity."""
    script = _find_model_diff_script()
    cmd = [
        sys.executable,
        str(script),
        "-ma", str(base_dir),
        "-mb", str(quant_dir),
        "-r", str(rows),
        "-d", str(device),
    ]
    logger.info(f"Running model_diff: {base_dir.name} vs {quant_dir.name}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if result.returncode != 0:
        logger.error(f"model_diff failed: {result.stderr[:500]}")
        raise RuntimeError(f"model_diff failed (rc={result.returncode}): {result.stderr[:500]}")

    out = result.stdout
    kl_matches = _KL_RE.findall(out)
    ppl_matches = _PPL_RE.findall(out)

    # KL divergence: take the first match (A, B direction)
    kl_div = float(kl_matches[0]) if kl_matches else None

    # Perplexity: model_diff prints A then B. B is the quantized model.
    ppl_base = float(ppl_matches[0]) if len(ppl_matches) >= 1 else None
    ppl_quant = float(ppl_matches[1]) if len(ppl_matches) >= 2 else None

    return {
        "kl_div": kl_div,
        "ppl_base": ppl_base,
        "ppl": ppl_quant,
        "raw_output": out,
    }


def run(config: QuantConfig, workspace: Path, outputs: list[QuantOutput]) -> None:
    """Measure KL divergence and PPL for each EXL3 output."""
    if config.format != QuantFormat.EXL3:
        return

    base_dir = workspace / "model"
    if not base_dir.exists():
        logger.warning("Base model dir missing — skipping quality metrics")
        return

    for output in outputs:
        quant_dir = Path(output.output_path)
        if not quant_dir.exists():
            logger.warning(f"Quant dir missing: {quant_dir}")
            continue

        try:
            metrics = _run_model_diff(base_dir, quant_dir, rows=100, device=0)
            output.quality = {
                "kl_div": metrics["kl_div"],
                "ppl": metrics["ppl"],
                "ppl_base": metrics["ppl_base"],
            }
            logger.info(
                f"Quality [{output.variant}]: KL={metrics['kl_div']}, PPL={metrics['ppl']}"
            )
        except RuntimeError as e:
            # OOM is common when loading base + quant simultaneously on 4090
            if "oom" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(f"Quality check OOM for {output.variant} — skipping: {e}")
            else:
                logger.warning(f"Quality check failed for {output.variant}: {e}")
            output.quality = {"error": str(e)}
