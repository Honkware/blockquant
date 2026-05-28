"""Stage 6: Upload to HF Hub via huggingface_hub."""
import os
from pathlib import Path

from huggingface_hub import create_repo, HfApi

from blockquant import cards
from blockquant.models import QuantConfig, QuantFormat, QuantOutput
from blockquant.utils.logger import get_logger

logger = get_logger(__name__)


def run(config: QuantConfig, workspace: Path, outputs: list[QuantOutput]) -> None:
    """Upload outputs to HuggingFace Hub."""
    hf_token = config.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.warning("No HF_TOKEN — skipping upload")
        return

    model_name = config.model_id.split("/")[-1]
    owner = config.hf_org
    if not owner:
        try:
            owner = HfApi(token=hf_token).whoami().get("name", "")
        except Exception:
            owner = ""

    for output in outputs:
        if output.format == QuantFormat.EXL3:
            repo_id = cards.exl3_repo_id(owner, model_name, output.variant)
        else:
            slug = f"{model_name}-{output.variant}-GGUF"
            repo_id = f"{owner}/{slug}" if owner else slug

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

        if output.format == QuantFormat.EXL3 and owner:
            cards.ensure_collection(
                owner=owner, base_name=model_name, token=hf_token,
                item_repo_ids=[repo_id],
            )
