#!/usr/bin/env python3
"""Run CatBench for one model on RunPod and tear the pod down.

Two prompts and two pictures — not a quantization. So this is deliberately
leaner than run_runpod_job.py: cheapest card that fits, no calibration, no
upload, and the pod dies the moment the result JSON is in hand.

    python run_catbench_job.py --model Qwen/Qwen3-8B --out /tmp/cb.json
    python run_catbench_job.py --model Qwen/Qwen3-8B --preflight

--preflight does the format and size gates alone and prints one JSON line. It
creates no pod, so the bot can reject an oversized model, or one in a format no
loader here reads, instantly.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from blockquant.providers.runpod_provider import RunPodProvider
from blockquant.poll import poll_remote
from blockquant.providers.runpod.constants import REMOTE_LOG, REMOTE_RESULT
# Same GPU catalogue, orphan sweep, arch registry and failure-line parser /quant
# uses. Importing is the point: one implementation of "which cards exist and
# what do they cost", "which architectures exllamav3 reads", and "what actually
# went wrong on the pod".
from run_runpod_job import (
    _auto_gpu_ids, _terminate_stray_pods, _last_exception, _resolve_arch,
    _image_missing,
)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

REMOTE_CB = "/root/catbench.py"
CB_SOURCE = Path(__file__).parent.parent / "src" / "blockquant" / "remote" / "catbench.py"

# A single H100 is the ceiling. bf16 weights are ~1 byte per byte of the HF
# download, so the download size IS the VRAM bill; the rest of the 80 GB is
# KV cache, activations and CUDA overhead.
DEFAULT_MAX_GB = 64.0
VRAM_HEADROOM_GB = 8.0


class _CatBenchProvider(RunPodProvider):
    """RunPodProvider that knows our script is catbench.py, not quant.py.

    is_pipeline_running greps for the quant script paths, so without this
    override poll_remote would call every CatBench run finished on the first
    tick. Subclassed rather than patched so /quant's provider is untouched.
    """

    def is_pipeline_running(self, instance_id: str) -> bool:
        pat = f"[{REMOTE_CB[0]}]{REMOTE_CB[1:]}"
        result = self.run(
            instance_id,
            f"command -v pgrep >/dev/null 2>&1 || {{ echo running; exit 0; }}; "
            f"pgrep -f '{pat}' >/dev/null && echo running || echo done",
        )
        return result["stdout"].strip() == "running"


def repo_format(model_id: str, token: str) -> str:
    """What kind of weights the repo holds: "exl3", "gguf", another
    quant_method, or "" for plain fp16/bf16 safetensors.

    Read from config.json and the file list, never the repo name.
    """
    from huggingface_hub import HfApi, hf_hub_download
    names: list[str] = []
    try:
        info = HfApi(token=token or None).model_info(model_id)
        names = [s.rfilename for s in (info.siblings or [])]
    except Exception:
        pass
    try:
        cfg = json.loads(Path(hf_hub_download(
            model_id, "config.json", token=token or None)).read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    method = str((cfg.get("quantization_config") or {}).get("quant_method", "")).lower()
    if method:
        return method
    # exllamav3 also drops a standalone quantization_config.json; an older quant
    # can have that and nothing in config.json.
    if "quantization_config.json" in names:
        return "exl3"
    # Only when there is nothing else to load: plenty of repos ship a GGUF
    # convenience copy alongside the real safetensors.
    if (any(n.lower().endswith(".gguf") for n in names)
            and not any(n.endswith(".safetensors") for n in names)):
        return "gguf"
    return ""


def format_check(model_id: str, token: str) -> dict:
    """Gate the weight format BEFORE anything is provisioned.

    The pod has two loaders, transformers and exllamav3, so a GGUF/AWQ/GPTQ repo
    can only die at "loading weights" six minutes and one pod in. Reject it here
    instead. For EXL3 the architecture also has to be one exllamav3 knows.
    """
    fmt = repo_format(model_id, token)
    if fmt and fmt != "exl3":
        return {"ok": False, "format": fmt,
                "error": f"`{model_id}` holds {fmt.upper()} weights. CatBench loads fp16/bf16 "
                         f"safetensors or an EXL3 quant; nothing on the pod reads {fmt.upper()}."}
    if fmt == "exl3":
        arch, supported, ok = _resolve_arch(model_id, token)
        if ok and not supported:
            return {"ok": False, "format": fmt,
                    "error": f"`{model_id}` is an EXL3 quant of `{arch}`, an architecture "
                             "exllamav3 does not support. Only exllamav3 can read EXL3, so "
                             "there is nothing here that can load it."}
    return {"ok": True, "format": fmt, "error": None}


def size_check(model_id: str, token: str, max_gb: float) -> dict:
    """Gate on model size BEFORE anything is provisioned.

    Reuses RunPodProvider._base_download_gb, the same HF lookup that drives
    /quant's GPU selection. It sums the actual sibling byte counts, so a
    quantized repo measures as its own size, not a bf16 guess. An unreadable
    size is a rejection: a model we cannot measure is a model we cannot promise
    fits one H100.
    """
    gb = RunPodProvider._base_download_gb(model_id, token)
    if gb is None:
        return {"ok": False, "gb": None,
                "error": f"could not read the size of `{model_id}` from HuggingFace "
                         "(gated, missing, or no safetensors)"}
    if gb > max_gb:
        return {"ok": False, "gb": round(gb, 1),
                "error": f"`{model_id}` is ~{gb:.0f} GB. CatBench runs on a single H100, "
                         f"so the limit is {max_gb:.0f} GB of weights (~{max_gb / 2:.0f}B "
                         "params at bf16). Too big to bench."}
    return {"ok": True, "gb": round(gb, 1), "error": None}


def failure_reason(provider, instance_id, outcome: str) -> str:
    """The real reason a run ended badly, for the [joberror] line.

    poll_remote only ever sees "Traceback (most recent call last):". The
    exception itself is several lines below it, and the marker-filtered progress
    tail drops those, which is how this reported a bare "run failed". Drain the
    RAW tail and run it through /quant's _last_exception.
    """
    try:
        tail = provider.get_progress(instance_id, lines=500, raw=True) or ""
    except Exception:
        tail = ""
    lines = [ln.rstrip() for ln in tail.splitlines() if ln.strip()]
    exc = _last_exception(lines)
    # The pod catches its own exceptions and prints [joberror] <what broke>, so
    # fall back to that before falling back to whatever the last line was.
    if not exc:
        exc = next((ln.split("[joberror]", 1)[1].strip()
                    for ln in reversed(lines) if "[joberror]" in ln), "")
    if outcome == "failed":
        return exc or (lines[-1][-300:] if lines else "the pod log was unreadable")
    return (f"hit the '{outcome}' watchdog limit "
            f"({exc or (lines[-1][-200:] if lines else 'no output')})")


def main():
    p = argparse.ArgumentParser(description="Run CatBench for one model on RunPod")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--out", default="", help="Where to write the result JSON")
    p.add_argument("--preflight", action="store_true",
                   help="Size gate only: print one JSON line and exit, no pod")
    p.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB)
    p.add_argument("--min-vram", type=int, default=0,
                   help="Override the VRAM floor derived from the model size")
    p.add_argument("--max-price", type=float, default=2.60, help="$/hr ceiling")
    p.add_argument("--image", default=os.environ.get("RUNPOD_IMAGE", ""))
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--runpod-api-key", default=os.environ.get("RUNPOD_API_KEY", ""))
    p.add_argument("--cloud", default="", help="SECURE or COMMUNITY (default: try both)")
    p.add_argument("--launch-retries", type=int, default=12)
    p.add_argument("--launch-retry-delay", type=int, default=20)
    p.add_argument("--poll-interval", type=int, default=15)
    # Two prompts on a loaded model is minutes. An hour is already generous;
    # past that something is wrong and the pod should stop billing.
    p.add_argument("--max-runtime", type=int, default=3600)
    p.add_argument("--stall-timeout", type=int, default=1200)
    p.add_argument("--keep-pod", action="store_true")
    args = p.parse_args()

    # Format first: "we cannot read GGUF" is more use than "that is 300 GB".
    fmt = format_check(args.model, args.hf_token)
    gate = (size_check(args.model, args.hf_token, args.max_gb) if fmt["ok"]
            else {"ok": False, "gb": None, "error": fmt["error"]})
    gate["format"] = fmt["format"] or "bf16"
    if args.preflight:
        print(json.dumps(gate), flush=True)
        sys.exit(0 if gate["ok"] else 1)
    if not gate["ok"]:
        print(f"[joberror] {gate['error']}", flush=True)
        sys.exit(1)
    if not args.hf_token or not args.runpod_api_key:
        print("[joberror] HF_TOKEN and RUNPOD_API_KEY are required", flush=True)
        sys.exit(1)

    base_gb = gate["gb"]
    # Weights + KV/activations, rounded to a card size. Passing base_gb=None to
    # _auto_gpu_ids keeps the ordering cheapest-first: this job is bandwidth and
    # download bound, so a faster card buys nothing.
    min_vram = args.min_vram or int(min(80, max(24, base_gb + VRAM_HEADROOM_GB) + 0.999))
    # Weights, the HF cache copy, and room to breathe. Nothing is quantized or
    # written back, so this is far below /quant's 120 GB floor.
    container_gb = max(60, int(base_gb * 2 + 30))

    print(f"[gpu] model ~{base_gb:.0f} GB ({gate['format']}) -> need >= {min_vram} GB VRAM, "
          f"disk {container_gb} GB, cap ${args.max_price:.2f}/hr", flush=True)

    if args.image and _image_missing(args.image):
        print(f"[joberror] image tag is not in the registry: {args.image}. "
              f"Nothing can boot from it; fix the pin before renting a card.", flush=True)
        sys.exit(1)

    gpu_candidates = _auto_gpu_ids(args.runpod_api_key, min_vram, None)
    if not gpu_candidates:
        print(f"[joberror] no GPUs with >= {min_vram} GB VRAM available", flush=True)
        sys.exit(1)
    print(f"[gpu] {len(gpu_candidates)} candidates, cheapest first", flush=True)

    run_tag = f"bq-{os.getpid()}-{int(time.time()) % 100000}"
    cloud_order = [args.cloud] if args.cloud else ["SECURE", "COMMUNITY"]
    pairs = [(g, c) for c in cloud_order for g in gpu_candidates]

    provider = None
    instance_id = None
    hourly = 0.0
    price_cache: dict = {}
    for sweep in range(1, max(1, args.launch_retries) + 1):
        for gpu, cloud in pairs:
            attempt = _CatBenchProvider(
                api_key=args.runpod_api_key, gpu_type=gpu, cloud_type=cloud,
                container_disk_gb=container_gb, volume_gb=10,
                image=args.image, name_prefix=run_tag,
            )
            rate = price_cache.get((gpu, cloud))
            if rate is None:
                try:
                    rate = attempt.get_cost_per_hour()
                except Exception:
                    rate = 0.0
                price_cache[(gpu, cloud)] = rate
            if args.max_price and rate and rate > args.max_price:
                continue
            print(f"[1/5] Trying {gpu} (~${rate:.2f}/hr {cloud})...", flush=True)
            try:
                instance_id = attempt.launch({})
            except Exception as e:
                print(f"      unavailable ({type(e).__name__}: {str(e)[:80]})", flush=True)
                continue
            print(f"      Pod ID: {instance_id}", flush=True)
            print("[2/5] Waiting for SSH...", flush=True)
            try:
                active = attempt.wait_for_active(instance_id)
            except Exception as e:
                active = {"status": "error", "error": str(e)}
            if active.get("status") != "active":
                print(f"      pod never came up "
                      f"({active.get('error') or active.get('status')}); next card", flush=True)
                try:
                    attempt.terminate(instance_id)
                except Exception:
                    pass
                instance_id = None
                continue
            provider, hourly = attempt, rate
            break
        if provider is not None:
            break
        _terminate_stray_pods(args.runpod_api_key, run_tag)
        if sweep < args.launch_retries:
            print(f"[1/5] No GPU free (sweep {sweep}); retrying in "
                  f"{args.launch_retry_delay}s...", flush=True)
            time.sleep(args.launch_retry_delay)

    if provider is None or instance_id is None:
        _terminate_stray_pods(args.runpod_api_key, run_tag)
        print("[joberror] no GPU could be secured for CatBench", flush=True)
        sys.exit(1)
    _terminate_stray_pods(args.runpod_api_key, run_tag, keep_id=instance_id)

    t0 = time.time()
    try:
        print("[3/5] Bootstrapping...", flush=True)
        if not provider.bootstrap(instance_id, exllamav3_local_dir=None):
            print("[joberror] bootstrap failed", flush=True)
            sys.exit(1)

        print("[4/5] Starting CatBench...", flush=True)
        cfg = {
            "model_id": args.model,
            "hf_token": args.hf_token,
            "pod_id": instance_id,
            "runpod_api_key": args.runpod_api_key,
            "keep_pod": bool(args.keep_pod),
            # Backstop only fires if THIS controller dies without terminating.
            "backstop_seconds": args.max_runtime + 600,
        }
        # 0600: it carries both secrets. catbench.py unlinks it after parsing.
        provider._upload_bytes(instance_id, json.dumps(cfg).encode("utf-8"),
                               "/root/bq-catbench-config.json", mode=0o600)
        provider._upload_bytes(instance_id, CB_SOURCE.read_bytes(), REMOTE_CB)
        provider.run(instance_id, f"chmod +x {REMOTE_CB}")
        launched = provider.run_detached(
            instance_id,
            f"rm -f {REMOTE_RESULT} {REMOTE_LOG} && "
            f"nohup setsid {provider._remote_py} {REMOTE_CB} "
            f"> {REMOTE_LOG} 2>&1 < /dev/null & echo $!",
        )
        print(f"      started (pid={launched['stdout'].strip() or '?'})", flush=True)

        print(f"[5/5] Polling every {args.poll_interval}s...", flush=True)
        last = ""

        def _print_new(tail):
            nonlocal last
            new = tail[len(last):] if tail.startswith(last) else tail
            sys.stdout.write(new if new.endswith("\n") else new + "\n")
            sys.stdout.flush()
            last = tail

        outcome = poll_remote(
            provider, instance_id, poll_interval=args.poll_interval,
            max_runtime=args.max_runtime, stall_timeout=args.stall_timeout,
            on_progress=_print_new,
        )
        if outcome != "done":
            print(f"[joberror] {failure_reason(provider, instance_id, outcome)}; "
                  "terminating pod", flush=True)
            sys.exit(3)

        result = None
        for _ in range(5):
            try:
                result = provider.get_result()
                if result is not None:
                    break
            except Exception as e:
                print(f"      result read retry ({e})", flush=True)
            time.sleep(6)

        elapsed = time.time() - t0
        cost = (elapsed / 3600) * hourly
        if result is None:
            print("[joberror] the pod produced no result", flush=True)
            sys.exit(1)
        if result.get("status") != "complete":
            print(f"[joberror] {result.get('error', 'unknown')}", flush=True)
            sys.exit(1)

        result["cost_usd"] = round(cost, 3)
        result["wall_seconds"] = round(elapsed)
        if args.out:
            Path(args.out).write_text(json.dumps(result), encoding="utf-8")
            print(f"[catbench] result {args.out}", flush=True)
        else:
            print(json.dumps(result))
        print(f"[catbench] done in {elapsed:.0f}s ~ ${cost:.2f}", flush=True)

    finally:
        if not args.keep_pod:
            print(f"Terminating pod {instance_id}...", flush=True)
            if not provider.terminate(instance_id):
                print(f"\n  !! POD {instance_id} MAY STILL BE BILLING !!\n"
                      f"     https://www.runpod.io/console/pods\n",
                      file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
