#!/usr/bin/env python3
"""
Pre-flight check: validates HF token write access and model existence.
Outputs a single JSON line to stdout so Node can parse it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Same allowlist the launcher gates on (backend/scripts/run_runpod_job.py), so a
# model is accepted or refused identically here and at launch. gen_arch_support.py
# regenerates it from exllamav3 at the ref the image is baked from. The
# hand-written copy that used to live here had drifted 16 architectures behind,
# and every one of them was turned away with "exllamav3 does not support it".
_ARCH_SUPPORT = Path(__file__).resolve().parent.parent / "backend" / "arch_support.json"


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
            from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
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
            except GatedRepoError:
                result['modelExists'] = True
                result['accessDenied'] = True
                result['error'] = (f"No access to {args.model}: it is gated and your token lacks "
                                   f"permission. Accept the license on the model page, then retry.")
            except RepositoryNotFoundError:
                result['modelExists'] = False
                result['error'] = f"Model {args.model} not found, or your token cannot access it."
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
