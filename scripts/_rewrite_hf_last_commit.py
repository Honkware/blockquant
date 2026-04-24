#!/usr/bin/env python3
"""One-off: rewrite last Hub commit message (soft reset + recommit + force push)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote


def git_exe() -> str:
    w = shutil.which("git")
    if w:
        return w
    for p in (
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\bin\git.exe",
    ):
        if Path(p).is_file():
            return p
    raise FileNotFoundError("git not found; install Git for Windows or add git to PATH")


def load_dotenv() -> None:
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


def main() -> int:
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    user = HfApi(token=token).whoami(token=token)["name"]
    repos = [
        "blockblockblock/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bpw-exl3",
        "blockblockblock/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bpw-exl3",
    ]
    u = quote(user, safe="")
    t = quote(token, safe="")
    base = f"https://{u}:{t}@huggingface.co/"
    new_msg = "Update model card"
    git = git_exe()

    for rid in repos:
        d = tempfile.mkdtemp(prefix="hf_git_")
        try:
            url = base + rid + ".git"
            subprocess.run([git, "clone", "--depth", "2", url, d], check=True)
            br = (
                subprocess.run(
                    [git, "-C", d, "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            subprocess.run(
                [git, "-C", d, "config", "user.email", "blockquant@users.noreply.huggingface.co"],
                check=True,
            )
            subprocess.run([git, "-C", d, "config", "user.name", "BlockQuant"], check=True)
            subprocess.run([git, "-C", d, "reset", "--soft", "HEAD~1"], check=True)
            subprocess.run([git, "-C", d, "commit", "-m", new_msg], check=True)
            subprocess.run([git, "-C", d, "push", "--force", "origin", f"HEAD:{br}"], check=True)
            print("OK", rid, "branch", br, "->", new_msg)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
