#!/usr/bin/env python3
"""
Pre-flight check: validates HF token write access and model existence.
Outputs a single JSON line to stdout so Node can parse it.
"""
import argparse
import json
import os
import sys

# Architectures exllamav3 can quantize, keyed by config.json architectures[0].
# Mirrors exllamav3/architecture/architectures.py plus the arches our fork image
# carries (Mellum, LocateAnything). Update when the baked exllamav3 gains one.
# Lets us reject an unsupported model at /quant time instead of booting a pod
# that downloads the weights and then fails on an unknown architecture.
SUPPORTED_ARCHS = {
    "AfmoeForCausalLM", "ApertusForCausalLM", "ArceeForCausalLM", "Cohere2ForCausalLM",
    "CohereForCausalLM", "DFlashDraftModel", "DeciLMForCausalLM", "Dots1ForCausalLM",
    "Ernie4_5_ForCausalLM", "Ernie4_5_MoeForCausalLM", "Exaone4ForCausalLM",
    "Gemma2ForCausalLM", "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration",
    "Gemma4ForConditionalGeneration", "Glm4ForCausalLM", "Glm4MoeForCausalLM",
    "Glm4vForConditionalGeneration", "Glm4vMoeForConditionalGeneration",
    "HCXVisionV2ForCausalLM", "HyperCLOVAXForCausalLM", "IQuestCoderForCausalLM",
    "Lfm2MoeForCausalLM", "LlamaForCausalLM", "LocateAnythingForConditionalGeneration",
    "MellumForCausalLM", "MiMoForCausalLM", "MiniMaxM2ForCausalLM", "Ministral3ForCausalLM",
    "Mistral3ForConditionalGeneration", "MistralForCausalLM", "MixtralForCausalLM",
    "NanoChatForCausalLM", "Olmo3ForCausalLM", "OlmoHybridForCausalLM", "Phi3ForCausalLM",
    "Qwen2ForCausalLM", "Qwen2_5_VLForConditionalGeneration", "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM", "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration", "SeedOssForCausalLM",
    "SmolLM3ForCausalLM", "SolarOpenForCausalLM", "Step3p5ForCausalLM",
    "Step3p7ForConditionalGeneration",
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
                result['architecture'] = arch
                result['archSupported'] = (arch in SUPPORTED_ARCHS) if arch else None
                if arch and arch not in SUPPORTED_ARCHS:
                    result['error'] = (f"exllamav3 does not support the '{arch}' architecture, "
                                       f"so this model cannot be quantized to EXL3.")
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
