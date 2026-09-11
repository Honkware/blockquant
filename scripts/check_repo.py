#!/usr/bin/env python3
"""
Inspect an output repo and decide whether it already holds the quant we were
about to make. Returns one JSON line for Node.js parsing.
"""
import argparse
import json
import math
import os
import re
import sys

from huggingface_hub import HfApi, hf_hub_download


def _float_eq(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    except Exception:
        return False


def _quant_config(repo, token, revision):
    """quantization_config out of the repo's config.json.

    This replaced a comparison against blockquant-manifest.json, which no repo
    on the hub has ever had: writeManifest runs after the upload and writes to
    the local output dir, so every lookup came back manifest_missing and the
    "already quantized" short-circuit could never fire. exllamav3 writes
    quantization_config itself, and it holds the fields that actually decide
    what the artifact is.
    """
    kwargs = {"repo_id": repo, "filename": "config.json", "repo_type": "model", "token": token}
    if revision:
        kwargs["revision"] = revision
    with open(hf_hub_download(**kwargs), encoding="utf-8") as f:
        return json.load(f).get("quantization_config") or {}


def _base_model(repo, token, revision):
    """base_model out of the card's YAML front matter, or None.

    Two source repos can share a model name (org-a/Qwen3-8B and org-b/Qwen3-8B
    both slug to Qwen3-8B-exl3-4.0bpw), so the name alone does not prove the
    quant came from the model in front of us. The old manifest carried
    sourceModel for this; the card front matter is the only published place it
    survives. Cards written before this field, or hand-edited ones, give None
    and the check is skipped rather than failed.
    """
    kwargs = {"repo_id": repo, "filename": "README.md", "repo_type": "model", "token": token}
    if revision:
        kwargs["revision"] = revision
    try:
        with open(hf_hub_download(**kwargs), encoding="utf-8") as f:
            head = f.read(4096)
    except Exception:
        return None
    m = re.match(r"---\s*\n(.*?)\n---", head, re.S)
    if not m:
        return None
    m = re.search(r"^base_model:\s*(\S+)\s*$", m.group(1), re.M)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(description="Inspect a published EXL3 repo")
    parser.add_argument("repo_name", help="Repository name without org prefix")
    parser.add_argument("--token", default=None, help="HF API token")
    parser.add_argument("--org", default="", help="Organization name (blank = current user)")
    parser.add_argument("--source_model", default="", help="Expected source model ID")
    parser.add_argument("--profile", default="", help="Ignored; profiles are gone")
    parser.add_argument("--bpw", default="", help="Expected bpw value")
    parser.add_argument(
        "--quant_options_json",
        default="{}",
        help="Settings the request pinned, e.g. {\"headBits\":8,\"codebook\":\"mcg\"}. "
             "Anything absent was left to exllamav3 and is not compared.",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Branch to inspect (e.g. 8.00bpw); default = main",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    result = {
        "repoId": None,
        "url": None,
        "exists": False,
        "settingsMatch": None,
        "reason": None,
        "quantConfig": None,
        "error": None,
    }

    if not token:
        result["error"] = "Missing HF token"
        print(json.dumps(result), flush=True)
        sys.exit(1)

    try:
        pinned = json.loads(args.quant_options_json or "{}")
        if not isinstance(pinned, dict):
            pinned = {}
    except Exception:
        pinned = {}

    api = HfApi()
    try:
        owner = args.org.strip() or api.whoami(token=token).get("name")
        full_repo = f"{owner}/{args.repo_name}"
        result["repoId"] = full_repo
        result["url"] = f"https://huggingface.co/{full_repo}"
    except Exception as e:
        result["error"] = f"Auth failed: {str(e)[:300]}"
        print(json.dumps(result), flush=True)
        sys.exit(1)

    try:
        api.model_info(repo_id=full_repo, token=token)
        result["exists"] = True
    except Exception:
        print(json.dumps(result), flush=True)
        return

    revision = (args.revision or "").strip() or None

    try:
        qc = _quant_config(full_repo, token, revision)
    except Exception:
        # Repo exists but has no readable config.json: mid-upload, or the
        # weights never landed. Not a match, and not a conflict either.
        result["settingsMatch"] = False
        result["reason"] = "config_missing"
        print(json.dumps(result), flush=True)
        return

    result["quantConfig"] = qc
    reasons = []

    if args.bpw != "":
        # bits is the integer floor (4 for a 4.5bpw quant); bits_per_weight is
        # the real number. Fall back for quants old enough to lack it.
        published = qc.get("bits_per_weight", qc.get("bits"))
        if not _float_eq(published, args.bpw):
            reasons.append("bpw_mismatch")

    if args.source_model:
        base = _base_model(full_repo, token, revision)
        if base is not None and base != args.source_model:
            reasons.append("source_model_mismatch")

    # Only what the request actually pinned. Leaving head bits unset means
    # "whatever exllamav3 picks", so an existing repo at this bpw satisfies it
    # regardless of what it was built with -- the requester expressed no opinion.
    for key, field in (("headBits", "head_bits"), ("codebook", "codebook"),
                       ("visionBits", "vision_bits")):
        want = pinned.get(key)
        if want in (None, ""):
            continue
        got = qc.get(field)
        same = _float_eq(got, want) if field != "codebook" else str(got) == str(want)
        if not same:
            reasons.append(f"{field}_mismatch")

    result["settingsMatch"] = not reasons
    if reasons:
        result["reason"] = ",".join(reasons)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
