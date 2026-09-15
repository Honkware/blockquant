#!/usr/bin/env python3
"""
Pre-flight check: validates HF token write access and model existence.
Outputs a single JSON line to stdout so Node can parse it.
"""
import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

# Same allowlist the launcher gates on (backend/scripts/run_runpod_job.py), so a
# model is accepted or refused identically here and at launch. gen_arch_support.py
# regenerates it from exllamav3 at the ref the image is baked from. The
# hand-written copy that used to live here had drifted 16 architectures behind,
# and every one of them was turned away with "exllamav3 does not support it".
_ARCH_SUPPORT = Path(__file__).resolve().parent.parent / "backend" / "arch_support.json"


def auto_vision_bits(arch: str) -> int:
    """What --vision_bits `auto` resolves to for this architecture.

    exllamav3 takes the number off the vision model's caps and falls back to 16
    -- copy the tower whole -- for anything with no validated tower. Knowing it
    here is what lets the published name say -V{n} without asking the requester,
    instead of the tower state being discoverable only from config.json after
    the upload.
    """
    try:
        table = json.loads(_ARCH_SUPPORT.read_text(encoding="utf-8")).get("default_vision_bits", {})
    except Exception:
        return 16
    return int(table.get(arch, 16))


def supported_archs() -> set:
    """Arch strings from arch_support.json, or an empty set if it is unreadable.
    Empty means no gate here; the launcher checks again before renting a pod, so
    the cost of a missing file is a later error, not a wasted GPU."""
    try:
        return set(json.loads(_ARCH_SUPPORT.read_text(encoding="utf-8"))["architectures"])
    except Exception:
        return set()


# Expert counts, in the spelling each arch family happens to use. A multimodal
# repo puts the LM's geometry under text_config, so both scopes get checked --
# same thing run_runpod_job._default_codebook does when it picks mcg for MoE.
_EXPERT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")


def model_facts(cfg: dict) -> dict:
    """Vision tower and MoE, straight off config.json.

    Cheap because the caller already downloaded the file for the arch gate.
    A vision tower means --vision_bits decides whether the tower is quantized
    or copied at fp16, and that lands in the published repo name, so the
    request has to say which -- hence surfacing it before the job is approved.
    """
    scopes = [cfg, cfg.get("text_config") or {}]
    experts = next((s[k] for s in scopes for k in _EXPERT_KEYS if s.get(k)), None)
    return {
        "hasVision": bool(cfg.get("vision_config")
                          or (cfg.get("text_config") or {}).get("vision_config")),
        "isMoe": experts is not None,
        "numExperts": experts,
    }


def _weight_gb(model_id: str, token: str, subfolder: str = "") -> float:
    """Total weight-file size in GB, or 0 when it cannot be read.

    The bot needs this before a pod exists: the cost band was a flat per-variant
    number that knew nothing about the model, so a 0.8B and a 70B quoted the
    same figure. Only the weights count -- tokenizer and config files are noise
    at this scale.
    """
    try:
        from huggingface_hub import HfApi
        total = 0
        for e in HfApi().list_repo_tree(model_id, token=token or None,
                                        path_in_repo=subfolder or None, recursive=True):
            if type(e).__name__ != "RepoFile":
                continue
            if e.path.endswith((".safetensors", ".bin", ".pt", ".gguf")):
                total += getattr(e, "size", 0) or 0
        return round(total / (1024 ** 3), 2)
    except Exception:
        return 0.0


def _top_level_dirs(model_id: str, token: str) -> list:
    """Directory names at the repo root, for telling someone where to look."""
    try:
        from huggingface_hub import HfApi
        return sorted(
            e.path for e in HfApi().list_repo_tree(model_id, token=token or None)
            if type(e).__name__ == "RepoFolder"
        )
    except Exception:
        return []


def _subfolder_models(model_id: str, token: str) -> list:
    """Subdirectories that hold a model, one entry each.

    Some repos ship several formats side by side -- BF16/, FP8/, GGUF/, NVFP4/ --
    with nothing at the root. Only an unquantized one is a usable source: an FP8
    or NVFP4 copy has already been rounded once, and quantizing that again
    compounds the error rather than measuring it.
    """
    from huggingface_hub import hf_hub_download

    out = []
    for d in _top_level_dirs(model_id, token):
        try:
            path = hf_hub_download(model_id, f"{d}/config.json", token=token)
            cfg = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        archs = cfg.get("architectures") or []
        out.append({
            "path": d,
            "architecture": archs[0] if archs else None,
            "quantized": bool(cfg.get("quantization_config")),
            **model_facts(cfg),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default=None, help='HuggingFace API token')
    parser.add_argument('--model', default=None, help='Model ID to check (org/name)')
    args = parser.parse_args()
    token = args.token or os.environ.get('HF_TOKEN')

    result = {'canWrite': False, 'modelExists': None, 'username': None, 'error': None}
    if not token:
        result['error'] = 'Missing HF token'
        print(json.dumps(result), flush=True)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        # Check token validity and permissions
        info = api.whoami(token=token)
        result['username'] = info.get('name', info.get('fullname', 'unknown'))

        # Check write access: whoami succeeds with a write token, but we can
        # also check the auth section.  Fine-grained tokens expose 'auth'.
        # For classic tokens, if whoami succeeds that's enough.
        auth = info.get('auth', {})
        access_token_role = auth.get('accessToken', {}).get('role', None)
        if access_token_role and access_token_role == 'read':
            result['canWrite'] = False
            result['error'] = 'Token has read-only access. A write token is required.'
        else:
            result['canWrite'] = True

        # Check existence, gated access, and arch support in one pass: fetch
        # config.json (403s on a gated repo the token can't read) and inspect
        # architectures[0]. Sets flags the Node side turns into clear errors.
        if args.model:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import (
                EntryNotFoundError, GatedRepoError, RepositoryNotFoundError,
            )
            try:
                cfg_path = hf_hub_download(args.model, 'config.json', token=token)
                result['modelExists'] = True
                with open(cfg_path, encoding='utf-8') as f:
                    cfg = json.load(f)
                archs = cfg.get('architectures') or []
                arch = archs[0] if archs else None
                supported = supported_archs()
                result['architecture'] = arch
                result['archSupported'] = (arch in supported) if (arch and supported) else None
                if arch and supported and arch not in supported:
                    result['error'] = (f"exllamav3 does not support the '{arch}' architecture, "
                                       f"so this model cannot be quantized to EXL3.")
                result.update(model_facts(cfg))
                result["sizeGb"] = _weight_gb(args.model, token)
                if result.get("hasVision"):
                    result["visionBitsAuto"] = auto_vision_bits(arch)
            except GatedRepoError:
                result['modelExists'] = True
                result['accessDenied'] = True
                result['error'] = (f"No access to {args.model}: it is gated and your token lacks "
                                   f"permission. Accept the license on the model page, then retry.")
            except RepositoryNotFoundError:
                result['modelExists'] = False
                result['error'] = f"Model {args.model} not found, or your token cannot access it."
            except EntryNotFoundError:
                # No config.json at the repo root. Usually a repo that ships
                # several formats in subdirectories (BF16/, FP8/, GGUF/ ...),
                # which the downloader pulls whole -- one of these cost a 433 GB
                # download and 13 minutes of pod before the converter looked for
                # a config that was never going to be there. The generic handler
                # below used to swallow this and let the job through, because it
                # cannot tell a missing file from a network blip.
                result['modelExists'] = True
                subs = _subfolder_models(args.model, token)
                result['subfolders'] = subs
                usable = [x for x in subs
                          if not x["quantized"] and x["architecture"] in supported_archs()]
                if len(usable) == 1:
                    # One obvious source, so do not make anyone name it.
                    pick = usable[0]
                    result['subfolder'] = pick["path"]
                    result['architecture'] = pick["architecture"]
                    result['archSupported'] = True
                    result.update({k: pick[k] for k in ("hasVision", "isMoe", "numExperts")})
                    if pick["hasVision"]:
                        result["visionBitsAuto"] = auto_vision_bits(pick["architecture"])
                else:
                    result['archSupported'] = False
                    if usable:
                        names = ", ".join(x["path"] for x in usable)
                        result['error'] = (
                            f"{args.model} keeps several quantizable models in "
                            f"subdirectories ({names}). Pass one as `subfolder`."
                        )
                    elif subs:
                        why = ", ".join(
                            f"{x['path']} ({'already quantized' if x['quantized'] else x['architecture'] or 'no architecture'})"
                            for x in subs
                        )
                        result['error'] = (
                            f"{args.model} has no quantizable model in it: {why}. An "
                            f"already-quantized copy is a bad source -- it has been "
                            f"rounded once and quantizing it again compounds that."
                        )
                    else:
                        dirs = _top_level_dirs(args.model, token)
                        where = f" It has subdirectories ({', '.join(dirs)})." if dirs else ""
                        result['error'] = (
                            f"{args.model} has no config.json at its root, so there is "
                            f"nothing to quantize there.{where}"
                        )
            except Exception:
                # Could not read config (network, missing file): fall back to a
                # plain existence check rather than hard-failing the preflight.
                try:
                    api.model_info(repo_id=args.model, token=token)
                    result['modelExists'] = True
                except Exception:
                    result['modelExists'] = False

    except Exception as e:
        result['error'] = str(e)[:300]

    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
