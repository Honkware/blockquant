#!/usr/bin/env python3
"""Quantize several bpw variants in parallel (one pod each), then finalize.

Fans out one ``run_runpod_job.py`` per variant; each grabs its own cheap pod
via ``--gpu auto`` and self-manages (watchdog + terminate + backstop). After
all finish, runs ``publish_quant.py`` once to cross-link every card's Quants
table and populate the collection, including variants quantized earlier.

Usage::

    python backend/scripts/run_parallel_quants.py \\
        --model huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated \\
        --variants 3.0,4.0,5.0,6.0
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_JOB = REPO_ROOT / "backend" / "scripts" / "run_runpod_job.py"
PUBLISH = REPO_ROOT / "backend" / "scripts" / "publish_quant.py"
SYNCER = REPO_ROOT / "backend" / "scripts" / "remote_log_syncer.py"

_POD_ID_RE = re.compile(r"Pod ID:\s*(\S+)")


def _feed_dashboard(variant: str, controller_log: Path, dash_log: Path,
                    syncers: list, stop: threading.Event) -> None:
    """Wait for the pod id to appear in the controller log, then start a
    header-seeded syncer so the dashboard sees complete logs (controller
    metadata + live pod stream) with no manual step. The pod's own remote log
    only carries the quant stream, so without the header the dashboard can't
    show runtime, spend, GPU, or the pod id.
    """
    pod_id = None
    while not stop.is_set():
        try:
            for line in controller_log.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                m = _POD_ID_RE.search(line)
                if m:
                    pod_id = m.group(1)
                    break
        except OSError:
            pass
        if pod_id:
            break
        stop.wait(5)
    if not pod_id:
        return
    dash_log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, str(SYNCER), "--pod", pod_id,
         "--local-log", str(dash_log), "--header-from", str(controller_log),
         "--interval", "20"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    syncers.append(proc)
    print(f"[parallel] {variant} bpw: dashboard syncer started (pod {pod_id})",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description="Parallel multi-bpw EXL3 quant + finalize.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True,
                    help="Comma list, one pod each, e.g. 3.0,4.0,5.0,6.0")
    ap.add_argument("--hf-org", default="")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--min-vram", type=int, default=24)
    ap.add_argument("--container-disk", default="auto",
                    help="GB, or 'auto' (default): each pod sizes its disk from "
                         "the model download plus its bpw output.")
    ap.add_argument("--cal-rows", type=int, default=250)
    ap.add_argument("--cloud", default="COMMUNITY")
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--log-dir", default="/tmp/blockquant-parallel")
    ap.add_argument("--dashboard-dir", default=str(REPO_ROOT / "backend" / "logs"),
                    help="Where to write complete (header + stream) logs for log_dashboard.py.")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="Skip the per-variant dashboard syncer.")
    ap.add_argument("--finalize-variants", default="3.0,4.0,4.5,5.0,6.0",
                    help="Full set the finalize step cross-links (include bpws quantized earlier).")
    ap.add_argument("--no-finalize", action="store_true")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dash_dir = Path(args.dashboard_dir)

    procs = []
    syncers: list = []
    stop = threading.Event()
    feeders: list[threading.Thread] = []
    for v in variants:
        log = log_dir / f"quant-{v}.log"
        cmd = [
            sys.executable, str(RUN_JOB),
            "--model", args.model, "--variants", v,
            "--profile", args.profile, "--skip-local-exllama",
            "--cal-rows", str(args.cal_rows), "--cloud", args.cloud,
            "--gpu", args.gpu, "--min-vram", str(args.min_vram),
            "--container-disk", str(args.container_disk),
        ]
        if args.hf_org:
            cmd += ["--hf-org", args.hf_org]
        fh = open(log, "w", encoding="utf-8")
        print(f"[parallel] launching {v} bpw -> {log}", flush=True)
        procs.append((v, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
        if not args.no_dashboard:
            t = threading.Thread(
                target=_feed_dashboard,
                args=(v, log, dash_dir / f"runpod-{v}.log", syncers, stop),
                daemon=True,
            )
            t.start()
            feeders.append(t)
        time.sleep(3)  # stagger pod creation so they don't all grab the same card

    print(f"[parallel] {len(procs)} jobs launched; waiting for all to finish...", flush=True)
    for v, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        note = "" if rc == 0 else "  (rc!=0 is usually the cosmetic wrap-up issue; verify by repo)"
        print(f"[parallel] {v} bpw finished rc={rc}{note}", flush=True)

    # Jobs are done, so stop the dashboard feeders and their syncers so we
    # don't leave background tails running after the pods are gone.
    stop.set()
    for proc in syncers:
        proc.terminate()

    if args.no_finalize:
        print("[parallel] skipping finalize (--no-finalize)", flush=True)
        return
    print("[parallel] finalizing cards + collection across all variants...", flush=True)
    fin = [sys.executable, str(PUBLISH), "--base", args.model,
           "--variants", args.finalize_variants, "--cal-rows", str(args.cal_rows)]
    if args.hf_org:
        fin += ["--hf-org", args.hf_org]
    subprocess.run(fin)


if __name__ == "__main__":
    main()
