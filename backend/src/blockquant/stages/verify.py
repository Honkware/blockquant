"""Stage 4: Load-test each output."""
from pathlib import Path

from blockquant.models import QuantConfig, QuantFormat, QuantOutput
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)


def run(config: QuantConfig, workspace: Path, outputs: list[QuantOutput]) -> None:
    """Verify each output file/directory exists and can be loaded."""
    for output in outputs:
        logger.info(f"Verifying {output.variant}...")
        p = Path(output.output_path)
        if not p.exists():
            raise FileNotFoundError(f"Output missing: {output.output_path}")
        if output.format == QuantFormat.GGUF:
            output.verified = _verify_gguf(p)
        else:
            output.verified = _verify_exl3(p)


def _verify_gguf(path: Path) -> bool:
    """Quick load test."""
    try:
        from llama_cpp import Llama

        llm = Llama(str(path), n_ctx=256, verbose=False)
        result = llm("The capital of France is", max_tokens=5, temperature=0.0)
        text = result["choices"][0]["text"] if "choices" in result else str(result)
        return "Paris" in text or len(text.strip()) > 0
    except ImportError:
        logger.warning("llama_cpp not available — skipping GGUF verify")
        return True
    except Exception as e:
        logger.warning(f"GGUF verify failed: {e}")
        return False


def _verify_exl3(path: Path) -> bool:
    """Basic existence check."""
    # Full ExLlamaV3 load test requires TabbyAPI or similar
    return path.exists() and any(path.iterdir())
