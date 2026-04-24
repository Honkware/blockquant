#!/usr/bin/env python3
"""
List Hugging Face model repos with low/zero downloads and optionally delete them.

Uses HF_TOKEN from the environment. Hub "downloads" is an aggregate counter and
can lag slightly after publish.

Examples:
  Dry-run (default), only BlockQuant-style EXL3 repos with 0 downloads:
    py -3 scripts/hub_cleanup.py --only-exl3

  Preview any repo under your account with downloads <= 0:
    py -3 scripts/hub_cleanup.py --author blockblockblock --max-downloads 0

  Actually delete (requires both flags):
    py -3 scripts/hub_cleanup.py --only-exl3 --delete --yes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Best-effort .env next to BlockQuant root (no python-dotenv dependency)."""
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
        if key and key not in os.environ:
            os.environ[key] = val

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

# Matches `ModelName-exl3` (one repo per model, branches per BPW) and legacy `ModelName-8bpw-exl3`.
EXL3_SUFFIX = re.compile(r"-exl3$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF model repo cleanup by download count")
    parser.add_argument(
        "--author",
        default="",
        help="Hub user or org name (default: token owner, or HF_ORG if set)",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=0,
        help="Select repos with downloads <= this value (default: 0)",
    )
    parser.add_argument(
        "--only-exl3",
        action="store_true",
        help="Only repos whose name ends with -exl3 (BlockQuant EXL3 uploads; includes legacy *-<bpw>bpw-exl3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max repos to scan (0 = no limit)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON lines instead of a table",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete selected repos (requires --yes)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive --delete",
    )
    args = parser.parse_args()

    _load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    author = (args.author or os.environ.get("HF_ORG") or "").strip()
    if not author:
        author = api.whoami(token=token)["name"]

    if args.delete and not args.yes:
        print("Refusing to delete without --yes", file=sys.stderr)
        sys.exit(1)

    candidates = []
    scanned = 0
    for model in api.list_models(author=author, full=True):
        if args.limit and scanned >= args.limit:
            break
        scanned += 1
        repo_id = model.id
        if args.only_exl3 and not EXL3_SUFFIX.search(repo_id.split("/")[-1]):
            continue
        downloads = getattr(model, "downloads", None)
        if downloads is None:
            continue
        if downloads > args.max_downloads:
            continue
        lm = getattr(model, "lastModified", None) or getattr(model, "last_modified", None)
        candidates.append(
            {
                "id": repo_id,
                "downloads": downloads,
                "last_modified": lm,
            }
        )

    if args.json:
        for c in candidates:
            print(json.dumps(c), flush=True)
    else:
        print(f"Author: {author}")
        print(f"Criteria: downloads <= {args.max_downloads}" + ("; --only-exl3" if args.only_exl3 else ""))
        print(f"Matches: {len(candidates)}")
        for c in candidates:
            lm = c["last_modified"]
            lm_s = lm if lm is not None else "?"
            print(f"  {c['id']}\tdownloads={c['downloads']}\tlastModified={lm_s}")

    if not args.delete:
        if not args.json and candidates:
            print("\nDry-run only. To delete, run again with --delete --yes")
        return

    errors = 0
    for c in candidates:
        rid = c["id"]
        try:
            api.delete_repo(repo_id=rid, repo_type="model")
            print(f"Deleted {rid}", flush=True)
        except Exception as e:
            errors += 1
            print(f"Failed {rid}: {e}", file=sys.stderr)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
