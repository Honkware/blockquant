#!/usr/bin/env python3
"""Post-upload publisher for a single EXL3 quant variant.

Run this AFTER an ``[upload] complete`` event for ``--bpw <X.X>``. It:

  1. Queries HuggingFace for the actual repo size (no more guessing).
  2. Renders the card from ``backend/templates/card_template.md``, building
     the Quants table dynamically from which sibling repos exist on HF.
  3. Pushes the card + coerces ``quantization_config.bits`` to int via
     ``fix_repo_card_and_config.py --card-file``.
  4. Adds the repo to the user's HF collection (idempotent).
  5. Re-pushes already-shipped sibling cards so their Quants tables
     stay in sync (the new variant becomes "link" instead of "queued").

Usage::

    python backend/scripts/publish_quant.py --bpw 3.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
from blockquant import cards  # noqa: E402  (after sys.path setup)

FIX_SCRIPT = REPO_ROOT / "backend" / "scripts" / "fix_repo_card_and_config.py"
COLLECTION_SLUG = (
    "blockblockblock/qwen36-35b-a3b-claude-47-opus-reasoning-distilled-exl3"
    "-69ece89b196fa4ae78d37550"
)

# Variants we plan to ship for this model. Cal-rows reflects the --profile
# that was used per-variant: fast=128 for the lower bpws, balanced=250 for
# the higher bpws.
KNOWN_VARIANTS = ["3.0", "4.0", "4.5", "5.0", "6.0"]
CAL_ROWS = {"3.0": 128, "4.0": 128, "4.5": 250, "5.0": 250, "6.0": 250}

def _real_size_gb(api, repo_id: str) -> float | None:
    """Sum of every file in the repo. Returns None when the repo doesn't
    exist yet OR is mid-upload (config.json not yet pushed) — either way
    we treat it as 'not published' for table-rendering purposes."""
    try:
        info = api.model_info(repo_id, files_metadata=True)
        names = {s.rfilename for s in info.siblings}
        # The in-pod uploader pushes safetensors before config.json.
        # If config.json is missing, the upload is still in flight.
        if "config.json" not in names:
            return None
        return sum((s.size or 0) for s in info.siblings) / 1e9
    except Exception:
        return None


def _quant_rows(api, base_name: str, hf_org: str) -> list[dict]:
    """Build the Quants-table rows from which sibling repos exist on HF."""
    rows = []
    for v in KNOWN_VARIANTS:
        repo_id = f"{hf_org}/{base_name}-exl3-{v}bpw"
        real = _real_size_gb(api, repo_id)
        rows.append({
            "variant": v, "head_bits": 8, "cal_rows": CAL_ROWS.get(v, 250),
            "size_gb": real,
            "url": f"https://huggingface.co/{repo_id}",
        })
    return rows


def _load_base_config(base_repo: str, token: str) -> dict:
    """Fetch the base model's config.json for architecture facts."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(base_repo, "config.json", token=token or None)
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _push_card(repo_id: str, base_repo: str, bpw: str, card_text: str) -> None:
    """Hand the rendered card to fix_repo_card_and_config.py via --card-file
    so config.json's bits coercion stays in one place."""
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8",
                                     suffix=".md") as tf:
        tf.write(card_text)
        tmp_path = tf.name
    try:
        cmd = [
            sys.executable, str(FIX_SCRIPT),
            "--repo", repo_id,
            "--base", base_repo,
            "--bpw", bpw,
            "--card-file", tmp_path,
        ]
        print(f"[publish] -> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(tmp_path)


def _add_to_collection(repo_id: str, token: str) -> None:
    from huggingface_hub import add_collection_item
    try:
        add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=repo_id,
            item_type="model",
            token=token,
            exists_ok=True,
        )
        print(f"[publish] collection: + {repo_id}", flush=True)
    except Exception as e:
        print(f"[publish] collection add WARN: {type(e).__name__}: {e}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpw", required=True, help="Variant just uploaded, e.g. 3.0")
    ap.add_argument("--base", default="lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled")
    ap.add_argument("--hf-org", default="blockblockblock")
    ap.add_argument(
        "--also-resync", default="3.0,4.0,4.5,5.0,6.0",
        help="Comma-list of sibling bpws whose cards should be re-rendered"
             " so their Quants tables show the newly-shipped variant as link.",
    )
    ap.add_argument("--title", default=None,
                    help="Hand-curated card heading; auto-derived from the name when omitted.")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env", override=True)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN missing"); sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    base_name = args.base.split("/")[-1]
    model_config = _load_base_config(args.base, token)
    license_id = cards.fetch_license(args.base, token)
    collection_url = f"https://huggingface.co/collections/{COLLECTION_SLUG}"
    quant_rows = _quant_rows(api, base_name, args.hf_org)

    # The complete list of bpws we want re-synced — current + siblings that
    # are already published so their tables flip "queued" -> "link".
    targets = [b.strip() for b in args.also_resync.split(",") if b.strip()]
    if args.bpw not in targets:
        targets.insert(0, args.bpw)

    # Skip targets that aren't published yet (still queued).
    published = []
    for v in targets:
        repo_id = f"{args.hf_org}/{base_name}-exl3-{v}bpw"
        if v == args.bpw or _real_size_gb(api, repo_id) is not None:
            published.append(v)

    print(f"[publish] re-syncing cards for: {', '.join(published)}", flush=True)

    for v in published:
        repo_id = f"{args.hf_org}/{base_name}-exl3-{v}bpw"
        size_gb = _real_size_gb(api, repo_id)
        if size_gb is None:
            # Should not happen since we filtered, but guard anyway.
            print(f"[publish] skip {v} — repo not on HF yet", flush=True)
            continue

        rendered = cards.render_exl3_card(
            base_repo=args.base, repo_id=repo_id, variant=v,
            head_bits=8, cal_rows=CAL_ROWS.get(v, 250), size_gb=size_gb,
            model_config=model_config, quant_rows=quant_rows,
            collection_url=collection_url, license_id=license_id,
            quantized_by=args.hf_org, title_override=args.title,
        )
        _push_card(repo_id, args.base, v, rendered)
        _add_to_collection(repo_id, token)
        print(f"[publish] {v} bpw DONE -> https://huggingface.co/{repo_id}",
              flush=True)


if __name__ == "__main__":
    main()
