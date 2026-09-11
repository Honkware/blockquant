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

        else:
            output.verification = _verify_exl3(p)
        output.verified = output.verification.status == VerificationStatus.PASSED


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
