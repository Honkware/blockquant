#!/usr/bin/env python3
"""Rescue upload — runs upload from a still-alive pod after the in-pod
upload step failed (e.g. wrong repo_id format).

Connects to the pod by ID, uploads a tiny fix script that re-runs
HfApi.upload_folder() with a properly-namespaced repo_id, executes it,
then optionally terminates the pod.

Usage:
    rescue_upload.py --pod kw1vcy4bdrwtmr \\
        --repo blockblockblock/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-exl3-4.5bpw \\
        --remote-dir /workspace/blockquant/output-4.5bpw \\
        --terminate
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# Silence paramiko's transient WARN-level reconnects (handled by retry layer).
logging.getLogger("paramiko.transport").setLevel(logging.ERROR)

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from blockquant.providers.runpod_provider import RunPodProvider


FIX_SCRIPT = r'''#!/usr/bin/env python3
"""One-shot HF upload using a properly-namespaced repo_id."""
import json, os, sys
from huggingface_hub import HfApi, login

cfg = json.loads(open("/root/bq-config.json").read())
hf_token = cfg["hf_token"]
login(token=hf_token)
api = HfApi(token=hf_token)

repo_id = "__REPO_ID__"
folder = "__FOLDER__"

print(f"[rescue] uploading {folder}  ->  {repo_id}", flush=True)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
api.upload_folder(
    folder_path=folder,
    repo_id=repo_id,
    repo_type="model",
)
print(f"[rescue] done — https://huggingface.co/{repo_id}", flush=True)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pod", required=True, help="RunPod pod ID (still alive)")
    p.add_argument("--repo", required=True, help="Full HF repo_id like org/name")
    p.add_argument("--remote-dir", required=True, help="Remote folder containing the quantized output")
    p.add_argument("--terminate", action="store_true", help="Terminate the pod after a successful upload")
    args = p.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY missing"); sys.exit(1)

    provider = RunPodProvider()
    provider._pod_id = args.pod  # skip create — pod already exists

    print(f"[rescue] connecting to pod {args.pod} ...")
    # Probe: pod must be alive and ssh-able
    endpoint = provider._get_ssh_endpoint(args.pod, timeout=120)
    print(f"[rescue] ssh endpoint: {endpoint['host']}:{endpoint['port']}")

    # Confirm output folder exists + has weights
    ls = provider.run(args.pod, f"ls -lh {args.remote_dir} 2>&1 | head -20")
    print(f"[rescue] {args.remote_dir} contents:")
    print(ls["stdout"])
    if ls["code"] != 0 or "No such" in ls["stdout"]:
        print(f"ERROR: remote dir missing"); sys.exit(2)

    # Push fix script
    script = (FIX_SCRIPT
              .replace("__REPO_ID__", args.repo)
              .replace("__FOLDER__", args.remote_dir))
    provider._upload_bytes(args.pod, script.encode("utf-8"), "/root/upload_fix.py")

    # Find the python interpreter that has huggingface_hub
    py_probe = provider.run(
        args.pod,
        "for p in python python3 python3.11 python3.12; do "
        "  command -v $p >/dev/null && $p -c 'import huggingface_hub' 2>/dev/null && echo $p && break; "
        "done"
    )
    py = py_probe["stdout"].strip() or "python3"
    print(f"[rescue] using interpreter: {py}")

    # Run it (this is the long part — multiple GB upload to HF)
    print(f"[rescue] starting upload (may take 5-15 min for ~18GB)...")
    res = provider.run(args.pod, f"{py} /root/upload_fix.py 2>&1")
    print(res["stdout"][-3000:])
    if res["code"] != 0:
        print(f"ERROR: upload failed exit {res['code']}")
        print(res["stderr"][-2000:])
        sys.exit(3)

    print(f"[rescue] ✓ upload OK  →  https://huggingface.co/{args.repo}")

    if args.terminate:
        print(f"[rescue] terminating pod {args.pod}...")
        provider.terminate(args.pod)
    else:
        print(f"[rescue] pod left alive (no --terminate flag). "
              f"Don't forget to call cleanup_pods.py when done.")


if __name__ == "__main__":
    main()
