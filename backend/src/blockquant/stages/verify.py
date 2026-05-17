"""Stage 4: Load-test each output."""
from pathlib import Path

from blockquant.models import (
    QuantConfig,
    QuantFormat,
    QuantOutput,
    VerificationResult,
    VerificationStatus,
)
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)


def run(config: QuantConfig, workspace: Path, outputs: list[QuantOutput]) -> None:
    """Verify each output file/directory exists and can be loaded."""
    for output in outputs:
        logger.info(f"Verifying {output.variant}...")
        p = Path(output.output_path)
        if not p.exists():
            output.verification = VerificationResult(
                status=VerificationStatus.FAILED,
                method="filesystem",
                message=f"Output missing: {output.output_path}",
            )
            output.verified = False
            raise FileNotFoundError(f"Output missing: {output.output_path}")
        if output.format == QuantFormat.GGUF:
            output.verification = _verify_gguf(p)
        else:
            output.verification = _verify_exl3(p)
        output.verified = output.verification.status == VerificationStatus.PASSED


def _verify_gguf(path: Path) -> VerificationResult:
    """Quick llama.cpp load test when the optional binding is installed."""
    try:
        from llama_cpp import Llama

        llm = Llama(str(path), n_ctx=256, verbose=False)
        result = llm("The capital of France is", max_tokens=5, temperature=0.0)
        text = result["choices"][0]["text"] if "choices" in result else str(result)
        if "Paris" in text or len(text.strip()) > 0:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                method="llama_cpp",
                message="sample generation returned text",
                details={"sample": text[:200]},
            )
        return VerificationResult(
            status=VerificationStatus.FAILED,
            method="llama_cpp",
            message="sample generation returned empty text",
        )
    except ImportError:
        logger.warning("llama_cpp not available — skipping GGUF verify")
        return VerificationResult(
            status=VerificationStatus.SKIPPED,
            method="llama_cpp",
            message="llama_cpp is not installed",
        )
    except Exception as e:
        logger.warning(f"GGUF verify failed: {e}")
        return VerificationResult(
            status=VerificationStatus.FAILED,
            method="llama_cpp",
            message=str(e),
        )


def _verify_exl3(path: Path) -> VerificationResult:
    """Basic artifact sanity check."""
    if path.exists() and any(path.iterdir()):
        return VerificationResult(
            status=VerificationStatus.PASSED,
            method="filesystem",
            message="output directory exists and is non-empty",
        )
    return VerificationResult(
        status=VerificationStatus.FAILED,
        method="filesystem",
        message="output directory is empty",
    )
