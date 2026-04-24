#!/usr/bin/env python3
"""Post-publish fix-up for an EXL3 quant repo.

  1. Patches config.json -> quantization_config:
     - coerces ``bits`` to an integer (HF Transformers validator wants
       int even though EXL3 supports fractional bpw)
     - adds an explicit ``bits_per_weight`` field with the true value
  2. Replaces README.md with a proper model card following the convention
     used at blockblockblock/Qwen3.5-9B-...-4bpw-exl3 (YAML frontmatter,
     base-model link, summary table, inference notes, license, then the
     upstream README appended verbatim).

Usage:
    fix_repo_card_and_config.py \
        --repo blockblockblock/Qwen3.6-...-exl3-4.5bpw \
        --base lordx64/Qwen3.6-...-Distilled \
        --bpw 4.5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass


def _fetch_upstream_readme(api, base_repo: str) -> str:
    """Pull the base model's README.md verbatim. Empty string on miss."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(
            repo_id=base_repo, filename="README.md", token=api.token, repo_type="model"
        )
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[card] upstream README fetch failed ({e}); appending nothing.")
        return ""


def _strip_frontmatter(md: str) -> tuple[str, str]:
    """Split README into (frontmatter_yaml, body) — frontmatter without fences."""
    if md.startswith("---\n"):
        end = md.find("\n---\n", 4)
        if end != -1:
            return md[4:end], md[end + 5:]
    return "", md


def _build_card(
    repo_id: str,
    base_repo: str,
    bpw_str: str,
    upstream_readme: str = "",  # kept for signature compat; intentionally unused
    head_bits: int = 8,
    cal_rows: int = 250,
    size_str: str = "",
) -> str:
    """Generate a model card in the trimmed convention used by turboderp /
    ArtusDev / bartowski. No operator metadata, no appendix, no version
    pinning — just what readers need to use the model.

    `upstream_readme` is accepted but not used: established quanters link
    to the base model rather than copy-pasting its README, which goes
    stale fast.
    """
    base_name = base_repo.split("/")[-1]
    short_name = repo_id.split("/")[-1]

    yaml_block = (
        "license: other\n"
        f"base_model: {base_repo}\n"
        "base_model_relation: quantized\n"
        f"quantized_by: {repo_id.split('/')[0]}\n"
        "library_name: exllamav3\n"
        "pipeline_tag: text-generation\n"
        "tags:\n"
        "  - exl3\n"
        "  - exllamav3\n"
        "  - quantized\n"
        "  - mixture-of-experts\n"
        "quantization_format: exl3\n"
        f"bits_per_weight: {bpw_str}\n"
    )

    intro = (
        f"## EXL3 quants of `{base_name}`\n\n"
        f"[ExLlamaV3](https://github.com/turboderp-org/exllamav3) builds of "
        f"[`{base_repo}`](https://huggingface.co/{base_repo}) for single-GPU "
        f"inference.\n\n"
        f"This repo contains the **{bpw_str} bpw** build.\n"
    )

    size_cell = size_str if size_str else "—"
    quants_table = (
        "## Quants\n\n"
        "| BPW | Head bits | Calibration rows | Size |\n"
        "| :--- | :--- | :--- | ---: |\n"
        f"| **{bpw_str}** | {head_bits} | {cal_rows} | {size_cell} |\n"
    )

    inference = (
        "## Inference\n\n"
        "- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) — OpenAI-compatible HTTP server\n"
        "- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) "
        "with the *ExLlamaV3* loader\n"
        "- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) directly via its Python API\n\n"
        f"VRAM rule of thumb at {bpw_str} bpw: model on disk + ~2 GB context overhead.\n"
    )

    download = (
        "## Download\n\n"
        "```bash\n"
        'pip install -U "huggingface_hub[cli]"\n\n'
        "huggingface-cli download \\\n"
        f"  {repo_id} \\\n"
        f"  --local-dir ./{short_name}\n"
        "```\n"
    )

    license_section = (
        "## License & use\n\n"
        f"Use and license follow the [base model]"
        f"(https://huggingface.co/{base_repo}). The quantization adds no "
        "additional restrictions.\n"
    )

    credits = (
        "## Credits\n\n"
        "Quantized with [BlockQuant](https://github.com/Honkware/blockquant).\n"
    )

    return (
        "---\n"
        + yaml_block
        + "---\n\n"
        + intro
        + "\n"
        + quants_table
        + "\n"
        + inference
        + "\n"
        + download
        + "\n"
        + license_section
        + "\n"
        + credits
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Target HF repo_id")
    p.add_argument("--base", required=True, help="Base model HF repo_id")
    p.add_argument("--bpw", required=True, help="Bits per weight (e.g. 4.5)")
    p.add_argument("--size", default="",
                   help="Optional size string for the Quants table (e.g. '~20.2 GB')")
    p.add_argument("--card-file", default="",
                   help="Path to a hand-edited Markdown file to upload verbatim "
                        "instead of generating one. Useful for one-off polish.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN missing"); sys.exit(1)

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)

    # ---- 1. Fix config.json ----
    print(f"[fix] downloading {args.repo}/config.json ...")
    cfg_path = hf_hub_download(repo_id=args.repo, filename="config.json", token=token)
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    qc = cfg.get("quantization_config", {})
    old_bits = qc.get("bits")
    qc["bits"] = int(float(args.bpw))            # int — satisfies HF validator
    qc["bits_per_weight"] = float(args.bpw)      # truth — what EXL3 actually used
    cfg["quantization_config"] = qc

    print(f"[fix] quantization_config.bits {old_bits} -> {qc['bits']}; "
          f"+ bits_per_weight={qc['bits_per_weight']}")

    # ---- 2. Model card: hand-rolled file if given, otherwise generate ----
    if args.card_file:
        card_path = Path(args.card_file)
        if not card_path.exists():
            print(f"ERROR: --card-file {card_path} not found"); sys.exit(2)
        card = card_path.read_text(encoding="utf-8")
        print(f"[fix] using hand-rolled card from {card_path} ({len(card)} chars)")
    else:
        # Generated cards mirror what the live quantization_config actually says.
        head_bits = int(qc.get("head_bits", 8))
        cal_rows = int(qc.get("calibration", {}).get("rows", 250))
        card = _build_card(
            repo_id=args.repo,
            base_repo=args.base,
            bpw_str=args.bpw,
            head_bits=head_bits,
            cal_rows=cal_rows,
            size_str=args.size or "",
        )
        print(f"[fix] generated card is {len(card)} chars (trimmed convention)")

    if args.dry_run:
        print("[fix] --dry-run, not pushing.")
        print("--- card preview (first 1500 chars) ---")
        print(card[:1500])
        return

    # ---- 3. Push both ----
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (tdp / "README.md").write_text(card, encoding="utf-8")

        print("[fix] uploading config.json ...")
        api.upload_file(
            path_or_fileobj=str(tdp / "config.json"),
            path_in_repo="config.json",
            repo_id=args.repo,
            repo_type="model",
            commit_message="fix: coerce quantization_config.bits to int + add bits_per_weight",
        )
        print("[fix] uploading README.md ...")
        api.upload_file(
            path_or_fileobj=str(tdp / "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            commit_message="docs: proper model card with base-model link + inference notes",
        )

    print(f"[fix] done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
