"""Stage 6: Upload to HF Hub via huggingface_hub."""
import os
from pathlib import Path

from huggingface_hub import create_repo, HfApi

from blockquant.models import QuantConfig, QuantFormat, QuantOutput
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)


def run(config: QuantConfig, workspace: Path, outputs: list[QuantOutput]) -> None:
    """Upload outputs to HuggingFace Hub."""
    hf_token = config.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.warning("No HF_TOKEN — skipping upload")
        return

    for output in outputs:
        model_name = config.model_id.split("/")[-1]
        if config.hf_org:
            repo_id = (
                f"{config.hf_org}/{model_name}-{output.variant}bpw-exl3"
                if output.format == QuantFormat.EXL3
                else f"{config.hf_org}/{model_name}-{output.variant}-GGUF"
            )
        else:
            repo_id = (
                f"{model_name}-{output.variant}bpw-exl3"
                if output.format == QuantFormat.EXL3
                else f"{model_name}-{output.variant}-GGUF"
            )

        # Create repo
        create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)

        # Upload files
        api = HfApi(token=hf_token)
        if output.format == QuantFormat.EXL3:
            # Upload directory
            for f in Path(output.output_path).rglob("*"):
                if f.is_file():
                    api.upload_file(
                        path_or_fd=str(f),
                        path_in_repo=f.name,
                        repo_id=repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
        else:
            # Upload single GGUF file
            api.upload_file(
                path_or_fd=output.output_path,
                path_in_repo=Path(output.output_path).name,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token,
            )

            # Upload README
            readme = Path(output.output_path).parent / f"README-{output.variant}.md"
            if readme.exists():
                api.upload_file(
                    path_or_fd=str(readme),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_token,
                )

        output.hf_repo_id = repo_id
        output.hf_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Uploaded: {output.hf_url}")
