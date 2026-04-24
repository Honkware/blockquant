#!/usr/bin/env python3
"""
Pre-flight check: validates HF token write access and model existence.
Outputs a single JSON line to stdout so Node can parse it.
"""
import argparse
import json
import os
import sys

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

        # Check model existence
        if args.model:
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
