#!/usr/bin/env python3
"""Submit a quantization job to RunPod and wait for completion.

Example:
    python run_runpod_job.py \
        --model lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled \
        --variants 4.5 \
        --gpu "NVIDIA H100 80GB HBM3" \
        --hf-org blockblockblock
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Windows cp1252 console blows up on non-ASCII in remote stderr. Force UTF-8.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockquant.providers.runpod_provider import RunPodProvider
from dotenv import load_dotenv

# override=True so .env always wins over stale values in the ambient shell
# environment. Otherwise a forgotten HF_TOKEN set months ago will silently
# ride along to the remote pod and fail there.
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)


def main():
    parser = argparse.ArgumentParser(description="Run EXL3 quantization on RunPod")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--variants", default="4.5", help="Comma-separated BPW values")
    parser.add_argument("--gpu", default="NVIDIA H100 80GB HBM3", help="Preferred RunPod GPU type")
    parser.add_argument(
        "--gpu-fallback",
        default="NVIDIA H100 NVL,NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB",
        help=(
            "Comma-separated GPU types to try if --gpu is out of stock "
            "(on 'no instances available' only; other errors abort)."
        ),
    )
    parser.add_argument("--cloud", default="COMMUNITY", help="COMMUNITY or SECURE")
    parser.add_argument(
        "--image",
        default="",
        help=(
            "Pod image override. Use ghcr.io/honkware/blockquant:latest "
            "to skip bootstrap (~5 min faster start). Defaults to RunPod's "
            "pytorch base, which triggers the in-pod bootstrap path."
        ),
    )
    parser.add_argument("--hf-org", default="", help="HF org for upload")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HF token")
    parser.add_argument("--runpod-api-key", default=os.environ.get("RUNPOD_API_KEY", ""), help="RunPod API key")
    parser.add_argument("--head-bits", type=int, default=8, help="Head bits for quantization")
    parser.add_argument(
        "--local-exllama",
        type=Path,
        default=Path(__file__).parent.parent.parent / "exllamav3",
        help="Path to local exllamav3 fork (skip to use PyPI)",
    )
    parser.add_argument("--skip-local-exllama", action="store_true", help="Install exllamav3 from PyPI")
    parser.add_argument(
        "--install-flash-attn",
        action="store_true",
        help="Attempt flash-attn install (off by default; SDPA is fine for quant)",
    )
    parser.add_argument("--keep-pod", action="store_true", help="Don't terminate pod after completion")
    parser.add_argument("--poll-interval", type=int, default=15, help="Progress poll interval in seconds")
    args = parser.parse_args()

    if not args.hf_token:
        print("ERROR: HF_TOKEN required (set env var or pass --hf-token)")
        sys.exit(1)
    if not args.runpod_api_key:
        print("ERROR: RUNPOD_API_KEY required (set env var or pass --runpod-api-key)")
        sys.exit(1)

    # Header line consumed by log_dashboard.py's parser.
    print(
        f"[job] model={args.model} variants={args.variants} format=exl3 "
        f"head_bits={args.head_bits} hf_org={args.hf_org or '(personal)'}",
        flush=True,
    )

    gpu_candidates = [args.gpu] + [g.strip() for g in args.gpu_fallback.split(",") if g.strip()]
    # De-dup while preserving order
    seen = set()
    gpu_candidates = [g for g in gpu_candidates if not (g in seen or seen.add(g))]

    provider = None
    instance_id = None
    chosen_gpu = None
    hourly = 0.0
    last_err = None
    for candidate in gpu_candidates:
        attempt = RunPodProvider(
            api_key=args.runpod_api_key,
            gpu_type=candidate,
            cloud_type=args.cloud,
            install_flash_attn=args.install_flash_attn,
            image=args.image,
        )
        rate = attempt.get_cost_per_hour()
        print(f"[1/6] Trying {candidate}  (~${rate:.2f}/hr {args.cloud})...", flush=True)
        try:
            instance_id = attempt.launch({})
        except Exception as e:
            msg = str(e).lower()
            # Match any "no(t) ... instances available" / "no capacity" phrasing RunPod throws.
            if ("instances available" in msg and ("no " in msg or "not " in msg)) or "capacity" in msg:
                print(f"      out of stock ({type(e).__name__}), falling through", flush=True)
                last_err = e
                continue
            raise
        provider = attempt
        chosen_gpu = candidate
        hourly = rate
        break

    if provider is None or instance_id is None:
        print(f"ERROR: all GPU candidates out of stock. Last error: {last_err}")
        sys.exit(1)
    print(f"      Pod ID: {instance_id}  (GPU: {chosen_gpu})")

    t_launch = time.time()
    try:
        print(f"[2/6] Waiting for SSH (up to 10 min)...")
        active = provider.wait_for_active(instance_id)
        if active["status"] != "active":
            print(f"ERROR: pod did not become active: {active}")
            sys.exit(1)
        ssh = active["ssh"]
        print(f"      SSH ready at {ssh['host']}:{ssh['port']}")

        print(f"[3/6] Bootstrapping (PyTorch, transformers, exllamav3, flash-attn)...")
        local_exl = None if args.skip_local_exllama else args.local_exllama
        if local_exl is not None and not local_exl.exists():
            print(f"      Local exllamav3 not found at {local_exl} — falling back to PyPI")
            local_exl = None
        if not provider.bootstrap(instance_id, exllamav3_local_dir=local_exl):
            print("ERROR: bootstrap failed")
            sys.exit(1)
        print("      Bootstrap complete")

        print(f"[4/6] Starting remote quantization...")
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        launch_result = provider.run_pipeline(
            instance_id=instance_id,
            model_id=args.model,
            format="exl3",
            variants=variants,
            hf_token=args.hf_token,
            hf_org=args.hf_org,
            head_bits=args.head_bits,
        )
        if launch_result.get("status") != "started":
            print(f"ERROR: run_pipeline failed: {launch_result}")
            sys.exit(1)
        print(f"      Remote script started")

        print(f"[5/6] Polling progress every {args.poll_interval}s...")
        last_tail = ""
        while provider.is_pipeline_running(instance_id):
            time.sleep(args.poll_interval)
            tail = provider.get_progress(instance_id)
            if tail and tail != last_tail:
                # Print only the new tail lines
                new = tail[len(last_tail):] if tail.startswith(last_tail) else tail
                sys.stdout.write(new if new.endswith("\n") else new + "\n")
                sys.stdout.flush()
                last_tail = tail
        # One deep final drain (last ~500 lines) so the local log captures
        # the last batch of remote output (final quantize layers, the
        # upload-complete line, status sentinel) — the routine 30-line
        # poll skips most of these when the run wraps up between ticks.
        final_tail = provider.get_progress(instance_id, lines=500)
        if final_tail and final_tail != last_tail:
            new = final_tail[len(last_tail):] if final_tail.startswith(last_tail) else final_tail
            sys.stdout.write(new if new.endswith("\n") else new + "\n")
            sys.stdout.flush()

        print(f"\n[6/6] Fetching result...")
        result = provider.get_result()
        elapsed = time.time() - t_launch
        cost = (elapsed / 3600) * hourly
        if result is None:
            print("ERROR: no result file on pod — check log tail above")
            sys.exit(1)
        status = result.get("status", "unknown")
        print(f"      Status: {status}")
        if status == "complete":
            for out in result.get("outputs", []):
                url = out.get("hf_url", "(no upload)")
                print(f"      {out['variant']} bpw  →  {url}")
            print(f"      Remote time: {result.get('total_time', 0):.0f}s")
            print(f"      Total time (incl. pod lifecycle): {elapsed:.0f}s  ≈  ${cost:.2f}")
        else:
            print(f"      Error: {result.get('error', 'unknown')}")
            sys.exit(1)

    finally:
        if not args.keep_pod:
            print(f"Terminating pod {instance_id}...")
            provider.terminate(instance_id)
        else:
            print(f"Pod {instance_id} kept alive (--keep-pod). Remember to terminate manually.")


if __name__ == "__main__":
    main()
