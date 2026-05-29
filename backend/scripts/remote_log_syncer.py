#!/usr/bin/env python3
"""Periodic remote → local log sync for stalled poll loops.

When run_runpod_job.py's poll loop hangs on a stuck SSH call (rare but
real — paramiko can wedge if the underlying TCP gets weird), this script
keeps the dashboard fed by tailing the remote log on its own schedule
and appending only new lines locally.

Idempotent by line-count offset: on each tick, it fetches remote lines
starting from the next unread position (``tail -n +N``) and appends
them. This correctly handles repeated log lines and does not grow any
in-memory state without bound.
Safe to leave running indefinitely; exits cleanly on KeyboardInterrupt.
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
    p.add_argument(
        "--header-from", type=Path, default=None,
        help="Prepend this controller log (GPU/pod/cost/timestamps) to the "
             "local log so it is self-complete for the dashboard. The pod's "
             "remote log only carries the live quant stream, so without this "
             "the dashboard can't show runtime, spend, or the pod id.",
    )
    args = p.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY missing"); sys.exit(1)

    prov = RunPodProvider()

    # Offset of remote lines already mirrored locally, tracked independently
    # of the local file's line count, because an optional header makes the two
    # diverge. tail -n +N is 1-based, so we fetch from remote_offset + 1.
    # Resume-aware: we never re-download the already-synced stream (a full
    # tail of a 50k-line remote log over one SSH exec comes back empty/truncated
    # anyway), so a restart picks up exactly where the prior run stopped.
    remote_offset = 0
    args.local_log.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if args.local_log.exists():
        existing = args.local_log.read_text(encoding="utf-8", errors="replace")

    if args.header_from and args.header_from.exists():
        header = args.header_from.read_text(
            encoding="utf-8", errors="replace"
        ).rstrip("\n") + "\n"
        header_lines = header.count("\n")
        if existing.startswith(header):
            # Already seeded on a prior run, so resume from the end of the
            # stream portion (everything past the static header).
            remote_offset = len(existing.splitlines()) - header_lines
        else:
            # Empty, or a legacy pure-mirror log with no header. Treat existing
            # content as already-synced remote lines and prepend the header
            # once, in place, without losing or re-downloading anything.
            remote_offset = len(existing.splitlines())
            args.local_log.write_text(header + existing, encoding="utf-8")
    elif existing:
        # No header: the local log is a pure remote mirror, so resume from
        # wherever it left off.
        remote_offset = len(existing.splitlines())

    print(f"[syncer] pod={args.pod}  local={args.local_log}  every {args.interval}s"
          + (f"  header={args.header_from}" if args.header_from else ""))
    while True:
        try:
            # Fetch only lines we haven't seen yet by starting from the
            # next unread line (tail line numbers are 1-based).
            # Note: if the remote log is rotated or truncated between ticks,
            # local_line_count will exceed the remote line count and tail
            # will return nothing until new lines are appended past that
            # offset. This is acceptable for a best-effort syncer.
            res = prov.run(args.pod, f"tail -n +{remote_offset + 1} {args.remote_log} 2>/dev/null")
            text = res.get("stdout") or ""
            new_lines = text.splitlines()
            if new_lines:
                with open(args.local_log, "a", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                remote_offset += len(new_lines)
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
