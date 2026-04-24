#!/usr/bin/env python3
"""
Generate README.md via generate_quant_readme.py and upload to existing EXL3 repos.

Default: Jackrong Qwen3.5-9B Claude Opus reasoning distill → blockblockblock/*-bpw-exl3 cards.

Usage (from BlockQuant root, HF_TOKEN in env or .env):
  py -3 scripts/upload_jackrong_quant_readmes.py

Options:
  --dry-run   Only write README files to ./tmp/readme-preview/ (no upload)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parent.parent
    path = root / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if "#" in val:
            val = val.split("#", 1)[0].strip()
        if key and key not in os.environ:
            os.environ[key] = val


def _sanitize_hf_namespace(raw: str) -> str:
    """HF user/org name only; empty if .env had a comment or invalid value."""
    s = (raw or "").strip()
    if not s or s.startswith("#"):
        return ""
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,94}$", s):
        return ""
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload EXL3 model cards for Jackrong 9B quants")
    parser.add_argument(
        "--source",
        default="Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled",
        help="Upstream Hugging Face model id (quant source)",
    )
    parser.add_argument(
        "--org",
        default="",
        help="Hub org or user (default: HF_ORG env, else token owner)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write READMEs under tmp/readme-preview/ only",
    )
    args = parser.parse_args()

    _load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("HF_TOKEN is required for upload (set in .env or environment).", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    gen_script = root / "scripts" / "generate_quant_readme.py"
    if not gen_script.is_file():
        print(f"Missing {gen_script}", file=sys.stderr)
        sys.exit(1)

    org = _sanitize_hf_namespace(args.org or os.environ.get("HF_ORG") or "")
    if not org and token:
        from huggingface_hub import HfApi

        org = HfApi(token=token).whoami(token=token)["name"]

    targets: list[tuple[str, float]] = [
        ("Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bpw-exl3", 4.0),
        ("Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bpw-exl3", 8.0),
    ]

    preview_dir = root / "tmp" / "readme-preview"
    if args.dry_run:
        preview_dir.mkdir(parents=True, exist_ok=True)

    exe = sys.executable
    env = {**os.environ}
    if token:
        env["HF_TOKEN"] = token

    for repo_name, bpw in targets:
        if args.dry_run:
            out_path = preview_dir / f"{repo_name}-README.md"
        else:
            fd, tmp = tempfile.mkstemp(suffix=".md", prefix="readme_")
            os.close(fd)
            out_path = Path(tmp)

        cmd = [
            exe,
            "-u",
            str(gen_script),
            "--output",
            str(out_path),
            "--source-repo",
            args.source,
            "--repo-name",
            repo_name,
            "--bpw",
            str(bpw),
            "--org",
            org,
        ]
        r = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"generate_quant_readme failed for {repo_name}:\n{r.stderr or r.stdout}", file=sys.stderr)
            if not args.dry_run:
                out_path.unlink(missing_ok=True)
            sys.exit(1)
        tail = (r.stdout or "").strip().splitlines()
        if tail:
            try:
                meta = json.loads(tail[-1])
                if not meta.get("ok"):
                    print(f"generate_quant_readme reported ok=false for {repo_name}: {meta}", file=sys.stderr)
                    sys.exit(1)
            except json.JSONDecodeError:
                pass

        if args.dry_run:
            print(f"Wrote {out_path}")
            continue

        from huggingface_hub import HfApi

        repo_id = f"{org}/{repo_name}"
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Update model card",
        )
        out_path.unlink(missing_ok=True)
        print(f"Uploaded README.md -> https://huggingface.co/{repo_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
