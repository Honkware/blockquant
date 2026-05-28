#!/usr/bin/env python3
"""Finalize EXL3 cards and the collection for a model.

Run this once after the quants have been uploaded (one variant per pod in a
parallel run, or all variants in a single run). For the given base model it:

  1. Queries HuggingFace for which ``-exl3-{bpw}bpw`` repos exist and their
     real sizes.
  2. Renders each card from ``backend/templates/card_template.md`` with the
     full cross-linked Quants table.
  3. Pushes the card and coerces ``quantization_config.bits`` to int via
     ``fix_repo_card_and_config.py --card-file``.
  4. Creates or reuses the per-model collection and adds every repo to it.

This is model-agnostic and idempotent, so it doubles as a resync: re-run it
any time to bring every card's table back in sync.

Usage::

    python backend/scripts/publish_quant.py \\
        --base huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated \\
        --hf-org blockblockblock
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

DEFAULT_VARIANTS = "3.0,4.0,4.5,5.0,6.0"


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


def _quant_rows(api, base_name, hf_org, variants, cal_rows, head_bits) -> list[dict]:
    """Build the Quants-table rows from which sibling repos exist on HF."""
    rows = []
    for v in variants:
        repo_id = f"{hf_org}/{base_name}-exl3-{v}bpw"
        rows.append({
            "variant": v, "head_bits": head_bits, "cal_rows": cal_rows,
            "size_gb": _real_size_gb(api, repo_id),
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


def main():
    ap = argparse.ArgumentParser(description="Finalize EXL3 cards + collection for a model.")
    ap.add_argument("--base", required=True,
                    help="Base model repo, e.g. huihui-ai/Huihui-Qwen3.6-35B-A3B-...-abliterated")
    ap.add_argument("--hf-org", default="",
                    help="Namespace the quants live under (default: the token owner).")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help="Comma list of bpws to consider; only those present on HF are rendered.")
    ap.add_argument("--collection", default="auto",
                    help="'auto' to create/reuse a per-model collection, a collection slug to "
                         "add to, or 'off' to skip collections.")
    ap.add_argument("--cal-rows", type=int, default=512,
                    help="Calibration rows shown in the recipe table.")
    ap.add_argument("--head-bits", type=int, default=8)
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
    hf_org = args.hf_org or api.whoami().get("name", "")
    if not hf_org:
        print("ERROR: could not resolve HF org (pass --hf-org)"); sys.exit(1)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    model_config = _load_base_config(args.base, token)
    license_id = cards.fetch_license(args.base, token)
    quant_rows = _quant_rows(api, base_name, hf_org, variants, args.cal_rows, args.head_bits)

    # Resolve the collection slug once, up front.
    if args.collection == "off":
        slug = ""
    elif args.collection == "auto":
        slug = cards.create_model_collection(owner=hf_org, base_name=base_name, token=token)
    else:
        slug = args.collection
    coll_url = cards.collection_url(slug) or f"https://huggingface.co/{hf_org}"

    published = [r["variant"] for r in quant_rows if r["size_gb"] is not None]
    if not published:
        print("[publish] no uploaded variants found on HF for this model yet.")
        sys.exit(1)
    print(f"[publish] finalizing cards for: {', '.join(published)}", flush=True)

    for v in published:
        repo_id = f"{hf_org}/{base_name}-exl3-{v}bpw"
        size_gb = _real_size_gb(api, repo_id)
        rendered = cards.render_exl3_card(
            base_repo=args.base, repo_id=repo_id, variant=v,
            head_bits=args.head_bits, cal_rows=args.cal_rows, size_gb=size_gb,
            model_config=model_config, quant_rows=quant_rows,
            collection_url=coll_url, license_id=license_id,
            quantized_by=hf_org, title_override=args.title,
        )
        _push_card(repo_id, args.base, v, rendered)
        cards.add_to_collection(slug, repo_id, token)
        print(f"[publish] {v} bpw done -> https://huggingface.co/{repo_id}", flush=True)
    if slug:
        print(f"[publish] collection: {coll_url}", flush=True)


if __name__ == "__main__":
    main()
