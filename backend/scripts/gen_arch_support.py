#!/usr/bin/env python3
"""Generate backend/arch_support.json -- the single source of truth for which
model architectures blockquant can quantize, derived straight from exllamav3.

exllamav3 declares one `arch_string = "<X>ForCausalLM"` per supported arch. We
grep them at two refs (the stable image's ref and exllamav3 master) and bucket
each arch into a tier: stable (works on the proven image), latest_only (only in
master -> needs the 0.0.43 image), or special (needs a specific image + has a
known quirk). The pre-flight gate, the /architectures command, and the
stay-current check all read the resulting JSON.

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

# Stable image's exllamav3 ref (Dockerfile.runpod EXLLAMAV3_REF) and the master
# ref the 0.0.43 image is built at. Bump these when the images are rebuilt.
STABLE_REF = "c18e9b40bed1d2bd9c30fbbc0ba0eeb19b4fbda1"
LATEST_REF = "origin/master"

# Logical image -> ghcr tag (matches run_runpod_job.py routing).
IMAGES = {
    "stable": "ghcr.io/honkware/blockquant (0.0.38, default)",
    "exl3_043": "ghcr.io/honkware/blockquant:qwen35-exl3-0.0.43-py312",
    "master": "ghcr.io/honkware/blockquant:v0.1.3",
}

# Archs that are in the registry but need a specific image + carry a quirk. Keep
# in sync with run_runpod_job.py's routing markers.
SPECIAL = {
    "Qwen3_5ForCausalLM": ("exl3_043", "gated-delta linear attn; needs flash-linear-attention + python 3.12"),
    "Qwen3_5ForConditionalGeneration": ("exl3_043", "gated-delta linear attn; needs flash-linear-attention + python 3.12"),
    "Qwen3_5MoeForCausalLM": ("exl3_043", "MoE + gated-delta; needs flash-linear-attention + python 3.12"),
    "Qwen3_5MoeForConditionalGeneration": ("exl3_043", "MoE + gated-delta; needs flash-linear-attention + python 3.12"),
    "Qwen3NextForCausalLM": ("exl3_043", "linear attn; needs flash-linear-attention + python 3.12"),
    "Lfm2MoeForCausalLM": ("master", "LFM2 MoE -- exllamav3 master image"),
}


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


def build(exl_dir: Path) -> dict:
    subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=exl_dir)
    stable = _arch_strings(exl_dir, STABLE_REF)
    latest = _arch_strings(exl_dir, LATEST_REF)

    archs = []
    for a in sorted(latest):
        if a in SPECIAL:
            image, note = SPECIAL[a]
            tier = "special"
        elif a in stable:
            image, note, tier = "stable", "", "stable"
        else:
            image, note, tier = "exl3_043", "new in exllamav3 0.0.43", "latest_only"
        archs.append({"arch": a, "tier": tier, "image": image, "note": note})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exllamav3": {
            "stable": {"ref": _short(exl_dir, STABLE_REF), "version": _version(exl_dir, STABLE_REF)},
            "latest": {"ref": _short(exl_dir, LATEST_REF), "version": _version(exl_dir, LATEST_REF)},
        },
        "images": IMAGES,
        "architectures": archs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exllamav3", required=True, help="path to an exllamav3 git checkout")
    p.add_argument("--out", default=str(Path(__file__).parent.parent / "arch_support.json"))
    p.add_argument("--check", action="store_true",
                   help="diff exllamav3 master against the committed JSON; exit 1 if new archs")
    args = p.parse_args()
    exl = Path(args.exllamav3)

    data = build(exl)
    if args.check:
        known = {a["arch"] for a in json.loads(Path(args.out).read_text())["architectures"]} \
            if Path(args.out).exists() else set()
        new = sorted(a["arch"] for a in data["architectures"] if a["arch"] not in known)
        if new:
            print(f"NEW architectures in exllamav3 {data['exllamav3']['latest']['version']} "
                  f"not yet in {Path(args.out).name}:")
            for a in new:
                print("  +", a)
            raise SystemExit(1)
        print(f"up to date: {len(data['architectures'])} archs, exllamav3 "
              f"{data['exllamav3']['latest']['version']}")
        return

    Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}: {len(data['architectures'])} archs "
          f"(stable {data['exllamav3']['stable']['version']} / "
          f"latest {data['exllamav3']['latest']['version']})")


if __name__ == "__main__":
    main()
