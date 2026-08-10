#!/usr/bin/env python3
"""Generate backend/arch_support.json -- the single source of truth for which
model architectures blockquant can quantize, derived straight from exllamav3.

exllamav3 declares one `arch_string = "<X>ForCausalLM"` per supported arch. We
grep them at the ref the image is baked from, and that set is the whole file:
one image serves every architecture, so there is nothing to route. The
pre-flight gate, the /architectures command and the stay-current check all read
the resulting JSON.

Usage:
    gen_arch_support.py --exllamav3 /root/blockquant/exllamav3 [--out ...]
    gen_arch_support.py --exllamav3 ... --check   # diff master vs committed JSON
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# The ref the image is built at (docker/Dockerfile.runpod EXLLAMAV3_REF). Bump
# both together, behind an end-to-end validation pass.
IMAGE_REF = "4f8ad0121f483ba66a5336244a4c3b6d7210385e"

# What --check measures the committed file against.
UPSTREAM_REF = "origin/master"


def _arch_strings(exl_dir: Path, ref: str) -> set[str]:
    out = subprocess.run(
        ["git", "grep", "-h", "arch_string = ", ref, "--", "exllamav3/architecture"],
        cwd=exl_dir, capture_output=True, text=True,
    ).stdout
    return set(re.findall(r'arch_string\s*=\s*"([^"]+)"', out))


def _version(exl_dir: Path, ref: str) -> str:
    out = subprocess.run(["git", "show", f"{ref}:exllamav3/version.py"],
                         cwd=exl_dir, capture_output=True, text=True).stdout
    m = re.search(r'(\d+\.\d+\.\d+)', out)
    return m.group(1) if m else "?"


def _short(exl_dir: Path, ref: str) -> str:
    return subprocess.run(["git", "rev-parse", "--short", ref], cwd=exl_dir,
                          capture_output=True, text=True).stdout.strip()


def build(exl_dir: Path, ref: str = IMAGE_REF) -> dict:
    subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=exl_dir)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exllamav3": {"ref": _short(exl_dir, ref), "version": _version(exl_dir, ref)},
        "architectures": sorted(_arch_strings(exl_dir, ref)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exllamav3", required=True, help="path to an exllamav3 git checkout")
    p.add_argument("--out", default=str(Path(__file__).parent.parent / "arch_support.json"))
    p.add_argument("--check", action="store_true",
                   help="diff exllamav3 master against the committed JSON; exit 1 if new archs")
    args = p.parse_args()
    exl = Path(args.exllamav3)

    if args.check:
        upstream = build(exl, UPSTREAM_REF)
        known = set(json.loads(Path(args.out).read_text())["architectures"]) \
            if Path(args.out).exists() else set()
        new = sorted(a for a in upstream["architectures"] if a not in known)
        if new:
            print(f"NEW architectures in exllamav3 {upstream['exllamav3']['version']} "
                  f"not yet in {Path(args.out).name}:")
            for a in new:
                print("  +", a)
            raise SystemExit(1)
        print(f"up to date: {len(known)} archs, exllamav3 "
              f"{upstream['exllamav3']['version']}")
        return

    data = build(exl)
    Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}: {len(data['architectures'])} archs "
          f"(exllamav3 {data['exllamav3']['version']})")


if __name__ == "__main__":
    main()
