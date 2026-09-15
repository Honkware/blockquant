#!/usr/bin/env python3
"""Find a quant to generate a self-calibration trace from.

The trace is the model sampling itself, so the donor has to be a quant of that
same model -- sc_rfn_probe walks both module trees together and sc_measure runs
the trace through the fp16's embedding, so anything else fails, in two
different ways. Highest bitrate at or above the target wins: least damaged
sample source.

Plain quants only. An SC quant was itself built from a trace, and calibrating
on one compounds whatever that trace got wrong.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Model name without the org")
    ap.add_argument("--min-bpw", type=float, required=True)
    ap.add_argument("--org", default="")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    out = {"donor": None, "error": None}
    try:
        from huggingface_hub import HfApi
        from blockquant import cards

        api = HfApi(token=token or None)
        org = args.org.strip() or api.whoami()["name"]
        found = []
        for m in api.list_models(author=org, limit=1000):
            rid = getattr(m, "id", None) or getattr(m, "modelId", "") or ""
            owner, _, slug = rid.partition("/")
            if owner != org:
                continue
            parsed = cards.parse_exl3_slug(slug, args.base)
            if not parsed or parsed["sc"]:
                continue
            if float(parsed["variant"]) >= args.min_bpw:
                found.append((float(parsed["variant"]), rid))
        if found:
            out["donor"] = max(found)[1]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
