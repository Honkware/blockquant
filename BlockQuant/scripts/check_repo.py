#!/usr/bin/env python3
"""
Inspect an output repo and compare its manifest settings.
Returns one JSON line for Node.js parsing.
"""
import argparse
import json
import math
import os
import sys

from huggingface_hub import HfApi, hf_hub_download


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _float_eq(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect a HF model repo manifest")
    parser.add_argument("repo_name", help="Repository name without org prefix")
    parser.add_argument("--token", default=None, help="HF API token")
    parser.add_argument("--org", default="", help="Organization name (blank = current user)")
    parser.add_argument("--source_model", default="", help="Expected source model ID")
    parser.add_argument("--profile", default="", help="Expected quant profile")
    parser.add_argument("--bpw", default="", help="Expected bpw value")
    parser.add_argument(
        "--quant_options_json",
        default="{}",
        help="Expected quant options JSON object",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Branch/revision for blockquant-manifest.json (e.g. 8.00bpw); default = main",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    result = {
        "repoId": None,
        "url": None,
        "exists": False,
        "settingsMatch": None,
        "reason": None,
        "manifest": None,
        "error": None,
    }

    if not token:
        result["error"] = "Missing HF token"
        print(json.dumps(result), flush=True)
        sys.exit(1)

    expected_quant_options = {}
    try:
        expected_quant_options = json.loads(args.quant_options_json or "{}")
        if not isinstance(expected_quant_options, dict):
            expected_quant_options = {}
    except Exception:
        expected_quant_options = {}

    api = HfApi()

    try:
        owner = args.org.strip()
        if not owner:
            owner = api.whoami(token=token).get("name")
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

    manifest = None
    try:
        dl_kwargs = {
            "repo_id": full_repo,
            "filename": "blockquant-manifest.json",
            "repo_type": "model",
            "token": token,
        }
        if revision:
            dl_kwargs["revision"] = revision
        manifest_path = hf_hub_download(**dl_kwargs)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        result["manifest"] = manifest
    except Exception:
        result["settingsMatch"] = False
        result["reason"] = "manifest_missing"
        print(json.dumps(result), flush=True)
        return

    checks = []
    reasons = []

    if args.source_model:
        ok = manifest.get("sourceModel") == args.source_model
        checks.append(ok)
        if not ok:
            reasons.append("source_model_mismatch")

    if args.profile:
        ok = manifest.get("profile") == args.profile
        checks.append(ok)
        if not ok:
            reasons.append("profile_mismatch")

    if args.bpw != "":
        ok = _float_eq(manifest.get("bpw"), args.bpw)
        checks.append(ok)
        if not ok:
            reasons.append("bpw_mismatch")

    manifest_quant_options = manifest.get("quantOptions", {})
    ok = _canon(manifest_quant_options) == _canon(expected_quant_options)
    checks.append(ok)
    if not ok:
        reasons.append("quant_options_mismatch")

    result["settingsMatch"] = all(checks) if checks else True
    if reasons:
        result["reason"] = ",".join(reasons)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
