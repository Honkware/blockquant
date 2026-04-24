#!/usr/bin/env python3
"""Watch the in-flight quant log; when the HF upload completes, rename the
repo from the old scheme (`{model}-{format}`) to the new per-bpw scheme
(`{model}-{variant}bpw-{format}`) so iter 9's output lands at the URL the
dashboard advertises.

This is a one-shot post-fact fix for jobs launched before the per-bpw
naming was rolled into the remote quant script.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


# Force stdout to line-buffer so the background output file updates live.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def watch_and_rename(
    log_path: Path,
    model_id: str,
    variant: str,
    hf_org: str,
    poll_sec: int = 30,
) -> int:
    load_dotenv(log_path.parent.parent.parent / ".env", override=True)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        log("ERROR: HF_TOKEN missing from .env")
        return 1

    from huggingface_hub import HfApi
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore

    model_name = model_id.split("/")[-1]
    old_slug = f"{hf_org}/{model_name}-exl3" if hf_org else f"{model_name}-exl3"
    new_slug = (
        f"{hf_org}/{model_name}-exl3-{variant}bpw"
        if hf_org else f"{model_name}-exl3-{variant}bpw"
    )

    upload_done_re = re.compile(r"\[upload\].*complete", re.IGNORECASE)
    status_complete_re = re.compile(r"\"status\":\s*\"complete\"")

    log(f"[rename-watcher] tailing {log_path}")
    log(f"[rename-watcher] will rename:  {old_slug}  ->  {new_slug}")
    log(f"[rename-watcher] poll every {poll_sec}s")

    ticks = 0
    while True:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            text = ""
        if upload_done_re.search(text) or status_complete_re.search(text):
            break
        if "ERROR: bootstrap failed" in text or "Status: failed" in text:
            log("[rename-watcher] job ended in failure; no rename to do.")
            return 0
        ticks += 1
        if ticks % 10 == 0:
            log(f"[rename-watcher] still waiting… ({ticks * poll_sec}s elapsed)")
        time.sleep(poll_sec)

    log("[rename-watcher] upload-complete signal detected — issuing rename")
    api = HfApi(token=token)
    try:
        try:
            api.repo_info(repo_id=old_slug, repo_type="model")
        except RepositoryNotFoundError:
            log(f"[rename-watcher] old repo {old_slug} not found; upload may have "
                "used a different slug. No rename performed.")
            return 0

        try:
            api.repo_info(repo_id=new_slug, repo_type="model")
            log(f"[rename-watcher] new repo {new_slug} already exists — refusing "
                "to clobber. Delete it manually if you want the rename.")
            return 2
        except RepositoryNotFoundError:
            pass

        api.move_repo(from_id=old_slug, to_id=new_slug, repo_type="model")
        log(f"[rename-watcher] ✓ renamed {old_slug} -> {new_slug}")
        log(f"[rename-watcher] URL: https://huggingface.co/{new_slug}")
        return 0
    except Exception as e:
        log(f"[rename-watcher] rename failed: {type(e).__name__}: {e}")
        return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--hf-org", default="")
    p.add_argument("--poll", type=int, default=30)
    args = p.parse_args()
    sys.exit(watch_and_rename(args.log, args.model, args.variant, args.hf_org, args.poll))


if __name__ == "__main__":
    main()
