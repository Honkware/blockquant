#!/usr/bin/env python3
"""Periodic remote → local log sync for stalled poll loops.

When run_runpod_job.py's poll loop hangs on a stuck SSH call (rare but
real — paramiko can wedge if the underlying TCP gets weird), this script
keeps the dashboard fed by tailing the remote log on its own schedule
and appending only the new lines locally.

Idempotent: tracks the last byte we wrote and only appends new content
on each tick. Safe to leave running indefinitely; exits cleanly on
KeyboardInterrupt.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# Silence paramiko's transient WARN-level reconnects — our retry layer
# in RunPodProvider.run handles them and they only add ledger noise.
logging.getLogger("paramiko.transport").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from blockquant.providers.runpod_provider import RunPodProvider


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pod", required=True)
    p.add_argument("--local-log", required=True, type=Path)
    p.add_argument("--remote-log", default="/root/bq.log")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--lines", type=int, default=300,
                   help="How many remote lines to fetch per tick.")
    args = p.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY missing"); sys.exit(1)

    prov = RunPodProvider()
    prov._pod_id = args.pod

    # Track what we've already mirrored locally.
    seen: set[str] = set()
    if args.local_log.exists():
        for line in args.local_log.read_text(encoding="utf-8", errors="replace").splitlines():
            seen.add(line)

    print(f"[syncer] pod={args.pod}  local={args.local_log}  every {args.interval}s")
    while True:
        try:
            res = prov.run(args.pod, f"tail -n {args.lines} {args.remote_log} 2>/dev/null")
            text = res.get("stdout") or ""
            new_lines = []
            for line in text.splitlines():
                if line not in seen:
                    seen.add(line)
                    new_lines.append(line)
            if new_lines:
                with open(args.local_log, "a", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                # Status line every tick where we actually appended something
                last_meaningful = next(
                    (ln for ln in reversed(new_lines)
                     if "Quantized:" in ln or "[upload]" in ln
                     or "Estimated remaining" in ln or "huggingface.co" in ln),
                    new_lines[-1] if new_lines else "",
                )
                print(f"[syncer] +{len(new_lines)} lines · last: {last_meaningful[:120]}",
                      flush=True)
            else:
                print(f"[syncer] no new lines this tick", flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[syncer] error this tick (will retry): {type(e).__name__}: {e}",
                  flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
