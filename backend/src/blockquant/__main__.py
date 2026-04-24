"""CLI for local testing — mirrors the existing bot's behavior.

Usage:
    bq-pipeline --model mistralai/Mistral-7B --format exl3 --variants 4.0,5.0
    bq-pipeline --model Qwen/Qwen2.5-7B --format gguf --variants q4_k_m,q5_k_m
"""
import argparse
import json
import os
import sys
from pathlib import Path

from blockquant.models import QuantConfig, QuantFormat
from blockquant.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="BlockQuant backend pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bq-pipeline --model mistralai/Mistral-7B --format exl3 --variants 4.0
  bq-pipeline --model Qwen/Qwen2.5-7B --format gguf --variants q4_k_m,q5_k_m
        """,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", choices=["gguf", "exl3"], default="exl3")
    parser.add_argument("--variants", default="4.0")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--hf-org", default=os.environ.get("HF_ORG", ""))
    parser.add_argument("--workspace", default="/tmp/blockquant-work")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--head-bits", type=int, default=8)

    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    config = QuantConfig(
        model_id=args.model,
        format=QuantFormat(args.format),
        variants=variants,
        provider=args.provider,
        hf_org=args.hf_org,
        hf_token=os.environ.get("HF_TOKEN", ""),
        workspace_dir=Path(args.workspace),
        head_bits=args.head_bits,
    )

    if args.dry_run:
        print(f"DRY RUN: {config.model_id}")
        print(f"  format={config.format.value} variants={config.variants}")
        print(f"  provider={config.provider}")
        sys.exit(0)

    result = run_pipeline(config)

    # Print JSON for the Node.js bot to parse
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    sys.exit(0 if result.status == "complete" else 1)


if __name__ == "__main__":
    main()
