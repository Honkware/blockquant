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
from blockquant.poll import poll_remote, DEFAULT_MAX_RUNTIME_S, DEFAULT_STALL_TIMEOUT_S
from dotenv import load_dotenv

# override=True so .env always wins over stale values in the ambient shell
# environment. Otherwise a forgotten HF_TOKEN set months ago will silently
# ride along to the remote pod and fail there.
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)


# Blackwell-class GPUs (sm_100/sm_120) need CUDA 12.8+ / torch >= 2.7; our
# torch 2.6 + cu124 has no kernels for them. Matched as substrings of the
# RunPod GPU id (e.g. "NVIDIA RTX PRO 4500 Blackwell", "NVIDIA GeForce RTX 5090").
_BLACKWELL_EXCLUDE = ("Blackwell", "B200", "B300", "RTX 5090", "RTX 5080", "RTX 5070")

# Cards too weak to reliably quantize a large model (low compute / VRAM-marginal
# for big MoE layers). Exact GPU-id match. The L4 froze mid-quant on the 35B MoE.
_WEAK_FOR_QUANT = {"NVIDIA L4"}

# exllamav3 is CUDA-only (no ROCm), so AMD cards can't compile or run the
# extension at all. RunPod lists e.g. "AMD Instinct MI300X OAM" whose huge VRAM
# sorts to the top of the big-model capable-first order, so a big quant would
# preferentially land on a card it cannot use and fail the health check. Matched
# as substrings of the RunPod GPU id.
_NON_CUDA_EXCLUDE = ("AMD", "Instinct", "Radeon")

# exllamav3 version is chosen by architecture. The stable release (0.0.37, what
# the bootstrap path installs) handles the proven models incl. Qwen3.6. The
# master build (0.0.38) adds newer archs like LFM2 but REGRESSES others
# (Qwen3.6 segfaults loading the first layer), so we only reach for the master
# image when the model's architecture actually needs it.
_MASTER_IMAGE = os.environ.get("RUNPOD_MASTER_IMAGE", "ghcr.io/honkware/blockquant:v0.1.3")
_MASTER_ONLY_ARCH_MARKERS = ("lfm2",)

# Qwen3.5 (qwen3_5 / qwen3_5_moe) and Qwen3-Next use gated-delta linear attention,
# which exllamav3 only handles in 0.0.43 via flash-linear-attention -- and fla's
# triton kernels only import on python 3.12 (triton #5224). So route ONLY those
# archs to the 0.0.43 / py3.12 image. 0.0.43 regresses proven models (Qwen3.6),
# so everyone else stays on 0.0.38. (ministral3's layer-0 crash was NOT this -- it
# was a float config field; remote/quant.py _sanitize_config fixes it on any image.)
# Markers are substrings of the lowercased arch+model_type; "qwen3_5" matches the
# dense AND MoE variant but NOT Qwen3.6 (qwen3 / qwen3_moe).
_EXL3_043_IMAGE = os.environ.get("RUNPOD_EXL3_043_IMAGE", "ghcr.io/honkware/blockquant:qwen35-exl3-0.0.43-py312")
_EXL3_043_ARCH_MARKERS = ("qwen3_5", "qwen3_next")


def _arch_markers(model_id: str, token: str) -> str:
    """Lowercased 'architectures + model_type' from config.json, '' on failure."""
    try:
        import json as _json
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(model_id, "config.json", token=token or None)
        with open(p) as f:
            cfg = _json.load(f)
        archs = " ".join(cfg.get("architectures") or [])
        return f"{archs} {cfg.get('model_type', '')}".lower()
    except Exception:
        return ""


def _arch_needs_master(model_id: str, token: str) -> bool:
    """True if the model's architecture is only supported on exllamav3 master."""
    hay = _arch_markers(model_id, token)
    return any(m in hay for m in _MASTER_ONLY_ARCH_MARKERS)


def _arch_needs_exl3_043(model_id: str, token: str) -> bool:
    """True for archs needing the 0.0.43/py3.12 image (qwen3_5*, qwen3_next -- linear attn + fla)."""
    hay = _arch_markers(model_id, token)
    return any(m in hay for m in _EXL3_043_ARCH_MARKERS)


# --- arch-support registry (the committed single source of truth) -------------
# Routing + the pre-flight gate read backend/arch_support.json (generated from
# exllamav3 by gen_arch_support.py). The marker funcs above stay only as a
# fallback for when the registry file is somehow missing.
_IMAGE_BY_NAME = {"exl3_043": _EXL3_043_IMAGE, "master": _MASTER_IMAGE, "stable": ""}


def _load_arch_support() -> dict:
    """{arch_string_lower: entry} from arch_support.json, or {} if absent."""
    import json as _json
    p = Path(__file__).parent.parent / "arch_support.json"
    try:
        d = _json.loads(p.read_text())
        return {a["arch"].lower(): a for a in d.get("architectures", [])}
    except Exception:
        return {}


def _resolve_arch(model_id: str, token: str):
    """(arch, registry_entry|None, config_read_ok). entry is the arch_support
    record (tier/image/note) when the arch is supported by exllamav3."""
    import json as _json
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(model_id, "config.json", token=token or None)
        cfg = _json.loads(Path(p).read_text())
    except Exception:
        return "", None, False
    reg = _load_arch_support()
    for a in (cfg.get("architectures") or []):
        if a.lower() in reg:
            return a, reg[a.lower()], True
    archs = cfg.get("architectures") or []
    return (archs[0] if archs else cfg.get("model_type", "?")), None, True


def _variant_uploaded(model_id: str, variants, hf_org: str, token: str) -> bool:
    """True if a requested variant's exl3 repo is already on HF with a
    config.json. Used as a fallback when the result file can't be read over a
    dropped SSH connection -- the pod uploads to HF on its own, so an upload may
    have succeeded even though the controller lost contact."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or None)
        base = model_id.split("/")[-1]
        org = hf_org or (api.whoami() or {}).get("name", "")
        for v in variants:
            try:
                info = api.model_info(f"{org}/{base}-exl3-{v}bpw")
                if "config.json" in {s.rfilename for s in (info.siblings or [])}:
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _recommend_max_price(base_gb: float | None) -> float:
    """Price cap scaled to model size. The quant is compute-bound, so a big
    model finishes ~3x faster on an A100/H100 for roughly the same TOTAL cost,
    while a small model is plenty fast on the cheap tier. Tiers by HF download
    GB (35B ~= 72 GB, 8B ~= 17 GB)."""
    if not base_gb:
        return 1.5
    if base_gb <= 20:
        return 0.80   # <= ~10B: cheap cards are fast enough
    if base_gb <= 50:
        return 1.30   # ~10-25B: RTX 5000 Ada / A40 / A6000 tier
    if base_gb <= 100:
        return 1.80   # ~25-50B: A100 tier (worth it, ~3x faster)
    return 2.80       # 50B+: A100 80GB / H100


def _auto_gpu_ids(api_key: str, min_vram_gb: int, base_gb: float | None = None) -> list[str]:
    """GPU type ids with at least min_vram_gb, ordered by the model size.

    Small models go cheapest-first (cheap cards quantize them fast). Big models
    are compute-bound and would crawl on a cheap card, so they go capable-first
    (priciest within the cap) and fall back to cheaper cards on a stock-out, so
    they finish far faster for ~the same total cost without ever getting stuck.
    """
    from blockquant.providers.runpod.pricing import static_price
    import runpod
    runpod.api_key = api_key
    cards = []
    for g in runpod.get_gpus():
        gid = (g.get("id") or "").strip()
        mem = g.get("memoryInGb") or 0
        if not gid or gid == "unknown" or "MIG" in gid or mem < min_vram_gb:
            continue
        # torch 2.6 / cu124 (the baked image AND the bootstrap path) has no
        # kernels for Blackwell (sm_100/sm_120): the quant dies mid-run with
        # "no kernel image is available for execution on the device". Skip
        # Blackwell-class cards until the stack moves to a cu128 torch. Plenty
        # of cheap Ada/Ampere/Hopper stock remains under the price cap.
        if any(tok in gid for tok in _BLACKWELL_EXCLUDE):
            continue
        # exllamav3 has no ROCm support, so AMD GPUs can't run the CUDA extension
        # (the baked .so is CUDA-only and the JIT recompile has no nvcc). Without
        # this, an AMD MI300X -- top of the big-model capable-first list by VRAM
        # -- fails the import/health check and kills the variant with no retry.
        if any(tok in gid for tok in _NON_CUDA_EXCLUDE):
            continue
        # Skip cards too weak to reliably quantize a large model. The L4 is a
        # low-power inference card (~72W) that froze mid-quant on the 35B MoE
        # while an RTX 5000 Ada on the same job finished fine. Exact match so we
        # don't also drop the capable L40 / L40S. Plenty of cheap, more capable
        # stock remains (3090, 4090, A5000, A40, RTX 5000 Ada...).
        if gid in _WEAK_FOR_QUANT:
            continue
        cards.append((mem, gid))
    # Big model (> ~25 GB download, ~12B+) -> capable-first: sort by price (then
    # VRAM) DESCENDING so the fastest allowed card is tried first, falling back
    # to cheaper ones. Small model -> cheapest/smallest first.
    big = bool(base_gb and base_gb > 25)
    cards.sort(key=lambda c: (static_price(c[1]), c[0]), reverse=big)
    return [gid for _, gid in cards]


def _terminate_stray_pods(api_key: str, prefix: str, keep_id: str = "") -> list[str]:
    """Kill pods named ``{prefix}-*`` except keep_id.

    RunPod's create_pod can create a pod AND still raise, so a failed launch
    attempt can orphan a billing pod the controller never learned the id of.
    The prefix is unique per run, so this only ever touches THIS run's strays,
    never pods from other concurrent jobs. Uses the REST API (the GraphQL one
    is flaky for rpa_ keys and has been timing out).
    """
    import json
    import urllib.request as u
    killed: list[str] = []
    try:
        req = u.Request("https://rest.runpod.io/v1/pods",
                        headers={"Authorization": f"Bearer {api_key}"})
        data = json.load(u.urlopen(req, timeout=30))
        pods = data if isinstance(data, list) else data.get("pods", [])
    except Exception:
        return killed
    for p in pods:
        pid = p.get("id")
        name = p.get("name") or ""
        if pid and pid != keep_id and name.startswith(prefix + "-"):
            try:
                u.urlopen(u.Request(f"https://rest.runpod.io/v1/pods/{pid}", method="DELETE",
                                    headers={"Authorization": f"Bearer {api_key}"}), timeout=20)
                killed.append(pid)
            except Exception:
                pass
    return killed


def main():
    parser = argparse.ArgumentParser(description="Run EXL3 quantization on RunPod")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--variants", default="4.5", help="Comma-separated BPW values")
    parser.add_argument(
        "--test-prompt", default=None,
        help="Optional prompt to run on each finished quant; the reply is echoed for the bot.",
    )
    parser.add_argument(
        "--gpu", default="NVIDIA H100 80GB HBM3",
        help="RunPod GPU type, or 'auto' to pick the cheapest in-stock card with enough VRAM.",
    )
    parser.add_argument(
        "--gpu-fallback",
        default="NVIDIA H100 NVL,NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB",
        help="Comma-separated GPU types to try if --gpu is out of stock. Ignored when --gpu auto.",
    )
    parser.add_argument(
        "--min-vram", type=int, default=24,
        help="With --gpu auto, only consider cards with at least this many GB of VRAM "
             "(quant is layer-by-layer and peaks ~4GB, so small cards are fine).",
    )
    parser.add_argument(
        "--max-price", default="auto",
        help="With --gpu auto, skip any card over this $/hr. 'auto' scales the "
             "cap to the model size (small models -> cheap cards; a big model -> "
             "A100/H100 since it's compute-bound and ~3x faster for ~the same "
             "total cost). A number pins it; 0 disables the cap.",
    )
    parser.add_argument(
        "--container-disk", default="auto",
        help="Container disk in GB, holding OS + torch/deps + the quantized "
             "outputs and conversion work dir (the unquantized model lives on "
             "the /workspace volume). 'auto' sizes from the model + variants; a "
             "number pins it.",
    )
    parser.add_argument(
        "--volume-disk", default="auto",
        help="/workspace volume in GB, or 'auto' (default) to size it from the "
             "model download plus the sum of all variant outputs plus a work "
             "dir. This is the disk that actually fills. Pass a number to pin it.",
    )
    parser.add_argument(
        "--launch-retries", type=int, default=6,
        help="How many times to re-sweep all GPU candidates when none are free. "
             "RunPod stock blips in and out per-GPU over minutes, so one pass is brittle.",
    )
    parser.add_argument(
        "--launch-retry-delay", type=int, default=30,
        help="Seconds to wait between GPU sweeps when everything was out of stock.",
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
    parser.add_argument(
        "--max-runtime", type=int, default=DEFAULT_MAX_RUNTIME_S,
        help="Hard cap in seconds on the remote run before the pod is terminated (default 8h).",
    )
    parser.add_argument(
        "--stall-timeout", type=int, default=DEFAULT_STALL_TIMEOUT_S,
        help="Terminate if no new log output arrives for this many seconds (default 60m).",
    )
    # Speedup tuning surface
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help=(
            "Speedup preset. Sets cloud + cal_rows + GPU preference. "
            "Per-knob flags below override the preset."
        ),
    )
    parser.add_argument(
        "--cal-rows", type=int, default=None,
        help="Override calibration rows (preset: fast=128 / balanced=250 / quality=512).",
    )
    parser.add_argument(
        "--cal-cols", type=int, default=None,
        help="Override calibration sequence length (default 2048).",
    )
    parser.add_argument(
        "--network-volume-id", default="",
        help="Mount a RunPod network volume for the work_dir (~5-10%% I/O speedup).",
    )
    parser.add_argument(
        "--data-center-id", default="",
        help="Pin the pod to a data center (required when --network-volume-id is set).",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Print resolved config + cost estimate and exit. Does not launch a pod.",
    )
    args = parser.parse_args()

    # ---- Resolve --profile + per-knob overrides -------------------------
    # Only apply preset's cloud/GPU when the user didn't pass theirs.
    cli_passed_cloud = "--cloud" in sys.argv
    cli_passed_gpu = "--gpu" in sys.argv
    cli_passed_fallback = "--gpu-fallback" in sys.argv

    profile_cfg = RunPodProvider.resolve_profile(
        args.profile,
        cal_rows=args.cal_rows,
        cal_cols=args.cal_cols,
    )
    cal_rows = profile_cfg["cal_rows"]
    cal_cols = profile_cfg["cal_cols"]
    if not cli_passed_cloud:
        args.cloud = profile_cfg["cloud_type"]
    if not cli_passed_gpu and not cli_passed_fallback:
        # Promote the profile's GPU list into --gpu + --gpu-fallback.
        prefs = profile_cfg["gpu_preference"]
        args.gpu = prefs[0]
        args.gpu_fallback = ",".join(prefs[1:])

    if not args.hf_token:
        print("ERROR: HF_TOKEN required (set env var or pass --hf-token)")
        sys.exit(1)
    if not args.runpod_api_key:
        print("ERROR: RUNPOD_API_KEY required (set env var or pass --runpod-api-key)")
        sys.exit(1)

    # Model, cache, work, and outputs ALL live on the LOCAL container disk now --
    # RunPod's /workspace volume is network-backed (mfs) in some DCs and throws
    # IO errors under big-model load. recommend_container_gb sizes the bounded
    # serial peak (model + one output + one work + one kl-stage); the volume is
    # a small unused stub.
    _variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if str(args.container_disk).strip().lower() == "auto":
        args.container_disk = RunPodProvider.recommend_container_gb(
            args.model, _variants, args.hf_token)
    else:
        args.container_disk = int(args.container_disk)
    if str(args.volume_disk).strip().lower() == "auto":
        args.volume_disk = 10
    else:
        args.volume_disk = int(args.volume_disk)
    print(f"[disk] all on local NVMe container -> /quant ({args.container_disk} GB) | "
          f"/workspace volume {args.volume_disk} GB stub", flush=True)

    # Model size (HF download GB) drives the price cap AND the card ordering:
    # big models go to a capable card (compute-bound, ~3x faster for ~same total
    # cost), small ones stay cheap. Computed once, reused for both.
    _base_gb = RunPodProvider._base_download_gb(args.model, args.hf_token)
    if str(args.max_price).strip().lower() == "auto":
        args.max_price = _recommend_max_price(_base_gb)
    else:
        args.max_price = float(args.max_price)
    print(f"[gpu] model ~{_base_gb or 0:.0f} GB -> price cap ${args.max_price:.2f}, "
          f"{'capable-first' if (_base_gb and _base_gb > 25) else 'cheapest-first'}", flush=True)

    # Architecture routing + pre-flight gate, both driven by the arch-support
    # registry (the single source of truth). Gate an unknown arch BEFORE spending
    # a pod, and force the right image for archs that need a non-default one --
    # overriding the bot-pinned RUNPOD_IMAGE. A `stable` arch keeps its pinned
    # image; proven models (incl. Qwen3.6) never move. Falls back to the marker
    # funcs only if arch_support.json is missing.
    _reg = _load_arch_support()
    if _reg:
        _arch, _entry, _ok = _resolve_arch(args.model, args.hf_token)
        if _ok and _entry is None:
            print(f"[joberror] unsupported architecture '{_arch}' -- not in the exllamav3 "
                  f"{len(_reg)}-arch support set. Refusing before launch (no pod, no download).",
                  flush=True)
            sys.exit(2)
        if _entry and _entry["image"] != "stable":
            args.image = _IMAGE_BY_NAME[_entry["image"]]
            _note = f" -- {_entry['note']}" if _entry.get("note") else ""
            print(f"[image] {args.model} ({_arch}, {_entry['tier']}) -> {args.image}{_note}", flush=True)
        elif not args.image:
            print("[image] bootstrap path (exllamav3 0.0.38, stable)", flush=True)
    elif _arch_needs_exl3_043(args.model, args.hf_token):
        args.image = _EXL3_043_IMAGE
        print(f"[image] {args.model} needs exllamav3 0.0.43; forcing {args.image} (py3.12 fresh-ext)", flush=True)
    elif not args.image:
        if _arch_needs_master(args.model, args.hf_token):
            args.image = _MASTER_IMAGE
            print(f"[image] {args.model} needs exllamav3 master; using {args.image}", flush=True)
        else:
            print("[image] bootstrap path (exllamav3 0.0.38, stable)", flush=True)

    # ---- --tune: read-only diagnostic ---------------------------------
    if args.tune:
        # Look up live pricing for the preferred GPU so the cost band is honest.
        try:
            tune_provider = RunPodProvider(
                api_key=args.runpod_api_key,
                gpu_type=args.gpu,
                cloud_type=args.cloud,
            )
            rate = tune_provider.get_cost_per_hour()
        except Exception as e:
            rate = 0.0
            print(f"(could not look up live price: {e})")
        # Estimate walltime band: yesterday's run was ~3h41m on COMMUNITY
        # NVL with cal_rows=250 — use that as the baseline.
        baseline_h = 3.7
        wt_factor = RunPodProvider.PROFILES[args.profile]["_walltime_factor"]
        # If user picked SXM, knock another ~10% off; SECURE adds ~5% more
        # consistency (fewer slowdowns) so net wash with COMMUNITY+SXM.
        gpu_speedup = 0.9 if "HBM3" in args.gpu else 1.0
        eta_low_h = baseline_h * wt_factor * gpu_speedup * 0.85
        eta_high_h = baseline_h * wt_factor * gpu_speedup * 1.10
        cost_low = eta_low_h * rate
        cost_high = eta_high_h * rate

        print()
        print(f"  PROFILE: {args.profile}  ({RunPodProvider.PROFILES[args.profile]['_summary']})")
        print(f"  GPU:     {args.gpu}   ${rate:.2f}/hr ({args.cloud.lower()})")
        print(f"  CAL:     {cal_rows} rows × {cal_cols} cols")
        if args.network_volume_id:
            print(f"  VOL:     {args.network_volume_id} (DC: {args.data_center_id or 'unset!'})")
        else:
            print(f"  VOL:     none (use --network-volume-id <id> for ~5-10% I/O speedup)")
        print(f"  IMAGE:   {args.image or '(default RunPod pytorch base — bootstraps in-pod)'}")
        print(f"  ETA:     ~{eta_low_h:.1f}h - ~{eta_high_h:.1f}h walltime  "
              f"(cost band: ${cost_low:.0f} - ${cost_high:.0f})")
        print()
        print(f"  LAUNCH:  re-run without --tune to start")
        print()
        return

    # Header line consumed by log_dashboard.py's parser. Skip the hf_org
    # field entirely when unset so the dashboard doesn't render the literal
    # placeholder "(personal)" as if it were a real account name.
    header = (
        f"[job] model={args.model} variants={args.variants} format=exl3 "
        f"head_bits={args.head_bits}"
    )
    if args.hf_org:
        header += f" hf_org={args.hf_org}"
    print(header, flush=True)

    if args.gpu.strip().lower() == "auto":
        gpu_candidates = _auto_gpu_ids(args.runpod_api_key, args.min_vram, _base_gb)
        if not gpu_candidates:
            print(f"ERROR: no GPUs with >= {args.min_vram}GB VRAM found")
            sys.exit(1)
        print(f"[gpu] auto: {len(gpu_candidates)} candidates >= {args.min_vram}GB, cheapest first")
    else:
        gpu_candidates = [args.gpu] + [g.strip() for g in args.gpu_fallback.split(",") if g.strip()]
    # De-dup while preserving order
    seen = set()
    gpu_candidates = [g for g in gpu_candidates if not (g in seen or seen.add(g))]

    # Unique pod-name prefix for THIS run, so orphan cleanup only ever touches
    # our own create-then-raise strays, never another concurrent job's pods.
    run_tag = f"bq-{os.getpid()}-{int(time.time()) % 100000}"

    # Try SECURE first (more reliable stock + network), then COMMUNITY, unless
    # the user pinned a cloud. The --max-price cap already keeps both clouds on
    # the cheap VRAM-appropriate tier, so SECURE-first costs nothing extra and
    # just lands a pod faster.
    cloud_order = [args.cloud] if cli_passed_cloud else ["SECURE", "COMMUNITY"]
    candidate_pairs = [(g, c) for c in cloud_order for g in gpu_candidates]

    provider = None
    instance_id = None
    chosen_gpu = None
    chosen_cloud = None
    hourly = 0.0
    ssh = None
    last_err = None
    _price_cache: dict = {}
    sweeps = max(1, args.launch_retries)
    for sweep in range(1, sweeps + 1):
        for candidate, cloud in candidate_pairs:
            attempt = RunPodProvider(
                api_key=args.runpod_api_key,
                gpu_type=candidate,
                cloud_type=cloud,
                container_disk_gb=args.container_disk,
                volume_gb=args.volume_disk,
                install_flash_attn=args.install_flash_attn,
                image=args.image,
                network_volume_id=args.network_volume_id,
                data_center_id=args.data_center_id,
                name_prefix=run_tag,
            )
            # Price cap (cached per gpu+cloud so retries stay fast). Skip cards
            # over --max-price so a stock-out can't push us onto an idle
            # H100/A100 at several times the cost of a capable cheap card.
            ckey = (candidate, cloud)
            rate = _price_cache.get(ckey)
            if rate is None:
                try:
                    rate = attempt.get_cost_per_hour()
                except Exception:
                    rate = 0.0
                _price_cache[ckey] = rate
            if args.max_price and rate and rate > args.max_price:
                print(f"      {candidate} ({cloud}) ${rate:.2f}/hr over cap "
                      f"${args.max_price:.2f}, skipping", flush=True)
                continue
            print(f"[1/6] Trying {candidate}  (~${rate:.2f}/hr {cloud})...", flush=True)
            try:
                instance_id = attempt.launch({})
            except Exception as e:
                msg = str(e).lower()
                # Out-of-stock is expected. But transient/rate-limit errors are
                # also common when several controllers hammer the API at once,
                # and re-raising here would crash the whole controller (and
                # orphan any pod a create-then-raise leaked). So NEVER re-raise
                # mid-sweep: log and try the next candidate. A genuinely fatal
                # condition just exhausts the sweeps and exits cleanly below.
                unavailable = (
                    ("instances available" in msg and ("no " in msg or "not " in msg))
                    or "capacity" in msg
                    or "does not have the resources" in msg
                    or "insufficient" in msg
                )
                label = "unavailable" if unavailable else "error"
                print(f"      {label} ({type(e).__name__}: {str(e)[:80]}), falling through", flush=True)
                last_err = e
                continue
            # Got a pod, but RunPod occasionally hands out a dead host that never
            # exposes SSH. That used to be fatal (sys.exit after a 10-min wait);
            # instead, confirm it becomes active here and, if not, terminate it
            # and fall through to the next card. So one bad host can't kill the
            # run while other stock is free, and the launch budget covers dead
            # hosts as well as stock-outs.
            print(f"      Pod ID: {instance_id}  (GPU: {candidate}, {cloud}, ~${rate:.2f}/hr)", flush=True)
            print("[2/6] Waiting for SSH (up to 10 min)...", flush=True)
            try:
                active = attempt.wait_for_active(instance_id)
            except Exception as e:  # noqa: BLE001
                active = {"status": "error", "error": str(e)}
            if active.get("status") != "active":
                print(f"      pod {instance_id} did not become active "
                      f"({active.get('status')}); terminating and trying the next card",
                      flush=True)
                try:
                    attempt.terminate(instance_id)
                except Exception:  # noqa: BLE001
                    pass
                last_err = RuntimeError(f"pod did not become active: {active}")
                instance_id = None
                continue
            provider = attempt
            chosen_gpu = candidate
            chosen_cloud = cloud
            hourly = rate
            ssh = active["ssh"]
            print(f"      SSH ready at {ssh['host']}:{ssh['port']}", flush=True)
            break
        if provider is not None:
            break
        # Whole sweep came up empty. Clean any create-then-raise orphans from
        # this run, then wait for stock to blip back before re-sweeping.
        strays = _terminate_stray_pods(args.runpod_api_key, run_tag)
        if strays:
            print(f"      cleaned {len(strays)} stray pod(s): {strays}", flush=True)
        if sweep < sweeps:
            print(f"[1/6] No GPU free (sweep {sweep}/{sweeps}); retrying in "
                  f"{args.launch_retry_delay}s...", flush=True)
            time.sleep(args.launch_retry_delay)

    if provider is None or instance_id is None:
        _terminate_stray_pods(args.runpod_api_key, run_tag)
        print(f"ERROR: all GPU candidates out of stock after {sweeps} sweeps. "
              f"Last error: {last_err}")
        sys.exit(1)
    # Defensive: a create-then-raise earlier in the sweep may have orphaned a
    # pod we never tracked; kill any of our strays that isn't the live one.
    strays = _terminate_stray_pods(args.runpod_api_key, run_tag, keep_id=instance_id)
    if strays:
        print(f"      cleaned {len(strays)} stray pod(s): {strays}", flush=True)
    # hourly was already cached during the sweep; only look it up if somehow unset.
    if not hourly:
        try:
            hourly = provider.get_cost_per_hour()
        except Exception:
            hourly = 0.0
    # The chosen pod's "Pod ID" and "SSH ready" lines were already printed during
    # the sweep, where SSH is now confirmed before a card is accepted.

    t_launch = time.time()
    try:
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
            cal_rows=cal_rows,
            cal_cols=cal_cols,
            keep_pod=args.keep_pod,
            test_prompt=args.test_prompt,
        )
        if launch_result.get("status") != "started":
            print(f"ERROR: run_pipeline failed: {launch_result}")
            sys.exit(1)
        print(f"      Remote script started")

        print(f"[5/6] Polling progress every {args.poll_interval}s...")
        last_tail = ""

        def _print_new(tail):
            nonlocal last_tail
            new = tail[len(last_tail):] if tail.startswith(last_tail) else tail
            sys.stdout.write(new if new.endswith("\n") else new + "\n")
            sys.stdout.flush()
            last_tail = tail

        outcome = poll_remote(
            provider, instance_id,
            poll_interval=args.poll_interval,
            max_runtime=args.max_runtime,
            stall_timeout=args.stall_timeout,
            on_progress=_print_new,
        )
        if outcome != "done":
            print(
                f"\n[watchdog] remote run hit the '{outcome}' limit; terminating pod.",
                flush=True,
            )
            sys.exit(3)  # finally still terminates the pod

        # One deep final drain (last ~500 lines) so the local log captures
        # the last batch of remote output (final quantize layers, the
        # upload-complete line, status sentinel) — the routine 30-line
        # poll skips most of these when the run wraps up between ticks.
        try:
            final_tail = provider.get_progress(instance_id, lines=500, raw=True)
            if final_tail and final_tail != last_tail:
                new = final_tail[len(last_tail):] if final_tail.startswith(last_tail) else final_tail
                sys.stdout.write(new if new.endswith("\n") else new + "\n")
                sys.stdout.flush()
        except Exception as e:
            print(f"      (final drain skipped: {e})", flush=True)

        print(f"\n[6/6] Fetching result...")
        # Reading the result is over SSH and can hit a transient reset right at
        # the end; retry a few times before giving up.
        result = None
        for _attempt in range(5):
            try:
                result = provider.get_result()
                if result is not None:
                    break
            except Exception as e:
                print(f"      result read retry ({e})", flush=True)
            time.sleep(8)
        elapsed = time.time() - t_launch
        cost = (elapsed / 3600) * hourly
        if result is None:
            # SSH unreachable / pod self-terminated. The pod uploads to HF on its
            # own, so check there before declaring failure.
            if _variant_uploaded(args.model, _variants, args.hf_org, args.hf_token):
                print("      result unreadable over SSH, but the variant is on HF "
                      "-> treating as complete", flush=True)
                sys.exit(0)
            print("ERROR: no result file on pod and nothing on HF — check log tail above")
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
            if not provider.terminate(instance_id):
                print(
                    f"\n  !! POD {instance_id} MAY STILL BE RUNNING AND BILLING !!\n"
                    f"     Could not confirm termination. Kill it manually now:\n"
                    f"     https://www.runpod.io/console/pods\n",
                    file=sys.stderr, flush=True,
                )
                sys.exit(2)
        else:
            print(f"Pod {instance_id} kept alive (--keep-pod). Remember to terminate manually.")


if __name__ == "__main__":
    main()
