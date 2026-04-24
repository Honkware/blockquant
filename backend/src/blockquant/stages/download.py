"""Stage 1: Download model from HuggingFace Hub with live progress."""
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from blockquant.models import QuantConfig
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_ARCHS = [
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "CohereForCausalLM",
]


def _get_total_size(repo_id: str, token: str | None = None) -> int:
    """Query HF API for total expected file size (bytes)."""
    try:
        api = HfApi(token=token)
        info = api.model_info(repo_id, files_metadata=True)
        return sum(
            sibling.size or 0
            for sibling in info.siblings
            if sibling.size and sibling.rfilename not in {".gitattributes", "README.md", "config.json"}
        )
    except Exception as e:
        logger.warning(f"Could not query total model size: {e}")
        return 0


def _poll_download_progress(
    model_path: Path,
    total_size: int,
    progress_callback,
    stop_event: threading.Event,
):
    """Poll directory size every 2s and emit progress via callback."""
    last_pct = -1
    while not stop_event.is_set():
        if model_path.exists():
            current_size = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            )
            if total_size > 0:
                pct = min(100, int(current_size / total_size * 100))
            else:
                pct = 0
            if pct != last_pct:
                last_pct = pct
                # Map download stage to 5% → 15% of overall pipeline
                overall_pct = 5 + int(pct * 0.10)
                msg = (
                    f"downloading {current_size / (1024**3):.2f} GiB / "
                    f"{total_size / (1024**3):.2f} GiB ({pct}%)"
                )
                progress_callback("download", overall_pct, msg)
        time.sleep(2)


def run(config: QuantConfig, workspace: Path, progress_callback=None) -> None:
    """Download model from HF Hub and validate architecture."""
    model_path = workspace / "model"
    if not (model_path / "config.json").exists():
        logger.info("Downloading model from HuggingFace Hub...")
        total_size = _get_total_size(config.model_id, config.hf_token or None)
        logger.info(f"Expected download size: {total_size / (1024**3):.2f} GiB")

        stop_event = threading.Event()
        monitor_thread = None
        if progress_callback and total_size > 0:
            monitor_thread = threading.Thread(
                target=_poll_download_progress,
                args=(model_path, total_size, progress_callback, stop_event),
                daemon=True,
            )
            monitor_thread.start()

        try:
            snapshot_download(
                repo_id=config.model_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=config.hf_token or None,
            )
        finally:
            stop_event.set()
            if monitor_thread:
                monitor_thread.join(timeout=3)

    # Validate architecture
    import json

    with open(model_path / "config.json") as f:
        cfg = json.load(f)
    arch = cfg.get("architectures", ["Unknown"])[0]
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported architecture: {arch}")

    logger.info(f"Model ready: {arch}")
