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
IMAGE_REF = "0740edc2da569fb99174023c1d2988b1e98cb41e"

# What --check measures the committed file against: the branch releases are cut
# from, matching what the image pins to.
UPSTREAM_REF = "origin/master"


def _arch_files(exl_dir: Path, ref: str) -> dict[str, str]:
    """{filename: source} for every architecture module at `ref`."""
    names = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{ref}:exllamav3/architecture"],
        cwd=exl_dir, capture_output=True, text=True,
    ).stdout.split()
    out = {}
    for n in names:
        if not n.endswith(".py"):
            continue
        out[n] = subprocess.run(["git", "show", f"{ref}:exllamav3/architecture/{n}"],
                                cwd=exl_dir, capture_output=True, text=True).stdout
    return out


def _arch_strings(exl_dir: Path, ref: str) -> set[str]:
    found = set()
    for src in _arch_files(exl_dir, ref).values():
        found.update(re.findall(r'arch_string\s*=\s*"([^"]+)"', src))
    return found


def _vision_bits(files: dict[str, str]) -> dict[str, int]:
    """{arch_string: default_vision_bits} for the towers exllamav3 will quantize.

    A tower's bitrate under `auto` is whatever its vision model class declares
    in caps, and absent means 16 -- the tower is copied whole. Grepping each
    file for the constant is not enough: the MoE variants (qwen3_vl_moe,
    glm4v_moe) declare their own arch_string but import the vision class from
    the dense sibling, so the number lives one module away. Follow the import.
    """
    local = {n: int(m.group(1))
             for n, src in files.items()
             if (m := re.search(r'"default_vision_bits"\s*:\s*(\d+)', src))}
    out: dict[str, int] = {}
    for name, src in files.items():
        bits = local.get(name)
        if bits is None:
            # e.g. `from .qwen3_vl import ..., Qwen3VLVisionModel`
            for mod in re.findall(r"from\s+\.(\w+)\s+import\s+([^\n]+)", src):
                if "VisionModel" in mod[1]:
                    bits = local.get(f"{mod[0]}.py")
                    if bits is not None:
                        break
        if bits is None:
            continue
        for arch in re.findall(r'arch_string\s*=\s*"([^"]+)"', src):
            out[arch] = bits
    return out


def _head_bits(exl_dir: Path, ref: str) -> int:
    """What `auto` resolves --head_bits to. Scraped, not typed in, because the
    bot has to name an SC quant -H{n} before a conversion exists to read it off,
    and a hardcoded 6 here goes quietly wrong the release upstream moves it."""
    src = subprocess.run(["git", "show", f"{ref}:exllamav3/conversion/convert_model.py"],
                         cwd=exl_dir, capture_output=True, text=True).stdout
    m = re.search(r'\(\s*"head_bits"\s*,\s*\w+\s*,\s*[\w.]*\s*or\s*(\d+)\s*\)', src)
    if not m:
        raise SystemExit("could not find the head_bits default in convert_model.py")
    return int(m.group(1))


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
        # What `auto` resolves --vision_bits to, per architecture. Anything not
        # listed has no validated tower and stays fp16. Lets the bot name a
        # quantized tower -V{n} without asking, instead of discovering it after
        # the upload.
        "default_vision_bits": dict(sorted(_vision_bits(_arch_files(exl_dir, ref)).items())),
        "default_head_bits": _head_bits(exl_dir, ref),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exllamav3", required=True, help="path to an exllamav3 git checkout")
    p.add_argument("--ref", default=IMAGE_REF,
                   help="ref to read arch strings at (default: the baked image ref)")
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

    data = build(exl, args.ref)
    Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}: {len(data['architectures'])} archs "
          f"(exllamav3 {data['exllamav3']['version']})")


if __name__ == "__main__":
    main()
