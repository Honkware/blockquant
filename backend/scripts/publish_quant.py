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
TEMPLATE = REPO_ROOT / "backend" / "templates" / "card_template.md"
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

# Per-bpw VRAM positioning copy.
POSITIONING = {
    "3.0": "the tightest fit — sized for 16&nbsp;GB consumer cards while leaving usable context room",
    "4.0": "the tight&#8209;fit build, sized to leave generous context room on a 24&nbsp;GB consumer GPU and to load on 16&nbsp;GB cards at workable context lengths",
    "4.5": "the quality-leaning sweet spot: comfortable on a single 24&nbsp;GB consumer GPU, effectively indistinguishable from FP16 on most reasoning tasks",
    "5.0": "the quality build — fits a 24&nbsp;GB card with reduced context, ideal headroom on 32&nbsp;GB cards",
    "6.0": "near-lossless reference quality — designed for 32&nbsp;GB+ cards (V100, A100, RTX&nbsp;6000)",
}

VRAM_HINT = {
    "3.0": "**VRAM at 3.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Fits a 16&nbsp;GB card with workable context, comfortable on 24&nbsp;GB with very long context.",
    "4.0": "**VRAM at 4.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Comfortable on a single 24&nbsp;GB card with room for ~24k tokens of context; fits a 16&nbsp;GB card with a ~4&ndash;6k token window.",
    "4.5": "**VRAM at 4.5&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Comfortable on a single 24&nbsp;GB card with room for ~16k tokens of context; fits a 16&nbsp;GB card with a reduced context window.",
    "5.0": "**VRAM at 5.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Tight on 24&nbsp;GB (limited context); comfortable on 32&nbsp;GB+.",
    "6.0": "**VRAM at 6.0&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead. Best on 32&nbsp;GB+ cards (V100, A100, RTX&nbsp;6000) where there's room for long context.",
}


def _est_size_gb(bpw: float) -> float:
    """Coarse pre-publish estimate. ~35B params plus head_bits=8 overhead."""
    return 35.0 * bpw / 8.0 + 1.5


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


def _build_quants_table(api, base_name: str, hf_org: str, current_bpw: str) -> str:
    rows = []
    for v in KNOWN_VARIANTS:
        repo_id = f"{hf_org}/{base_name}-exl3-{v}bpw"
        real = _real_size_gb(api, repo_id)
        is_current = (v == current_bpw)
        if real is not None:
            size_str = f"{real:.1f}&nbsp;GB"
            status = ("<kbd>this repo</kbd>" if is_current
                      else f"[link](https://huggingface.co/{repo_id})")
        else:
            size_str = f"<i>~{_est_size_gb(float(v)):.0f}&nbsp;GB</i>"
            status = "<sub>queued</sub>"
        if is_current:
            size_str = f"**{size_str.replace('&nbsp;', '&nbsp;')}**"
        bold_open = "**" if is_current else ""
        bold_close = "**" if is_current else ""
        rows.append(
            f"| {bold_open}{v}{bold_close} | 8 | {CAL_ROWS[v]} | {size_str} | {status} |"
        )
    header = (
        "| BPW &nbsp; | &nbsp; Head bits &nbsp; | "
        "&nbsp; Calibration rows &nbsp; | &nbsp; Size &nbsp; | &nbsp; Status |\n"
        "| :---: | :---: | :---: | ---: | :--- |"
    )
    return header + "\n" + "\n".join(rows)


def _render(template: str, ctx: dict) -> str:
    out = template
    for k, v in ctx.items():
        out = out.replace("{{" + k + "}}", v)
    return out


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
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env", override=True)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN missing"); sys.exit(1)

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    base_name = args.base.split("/")[-1]
    template = TEMPLATE.read_text(encoding="utf-8")

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

        ctx = {
            "BASE_REPO": args.base,
            "BASE_BADGE": args.base.replace("/", "%2F").replace("-", "--"),
            "BPW": v,
            "SIZE_GB": f"{size_gb:.1f}",
            "SIZE_GB_BADGE": f"{size_gb:.1f}",
            "REPO_ID": repo_id,
            "SHORT_NAME": f"{base_name}-exl3-{v}bpw",
            "CAL_ROWS": str(CAL_ROWS.get(v, 250)),
            "POSITIONING": POSITIONING.get(v, f"the {v} bpw build"),
            "VRAM_HINT": VRAM_HINT.get(v,
                f"**VRAM at {v}&nbsp;bpw:** weights on disk + ~2&nbsp;GB context overhead."),
            "QUANTS_TABLE": _build_quants_table(api, base_name, args.hf_org, v),
        }
        rendered = _render(template, ctx)
        _push_card(repo_id, args.base, v, rendered)
        _add_to_collection(repo_id, token)
        print(f"[publish] {v} bpw DONE -> https://huggingface.co/{repo_id}",
              flush=True)


if __name__ == "__main__":
    main()
