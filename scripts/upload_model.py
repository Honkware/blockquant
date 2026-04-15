#!/usr/bin/env python3
"""
Upload a quantized model folder to HuggingFace Hub.
Creates the repo if it doesn't exist, then uploads all files.
Outputs a JSON result line for Node.js to parse.
"""
import argparse
import json
import os
import sys
import time

# Upload path is more stable without forcing hf_transfer.
os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
# Avoid experimental multi-commit / PR path (can exit non-zero on some hub versions).
os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')


def log_status(stage, completion, status):
    msg = json.dumps({'stage': stage, 'completion': str(completion), 'status': status})
    print(f'[STATUS]{msg}[/STATUS]', flush=True)


def is_retryable_error(text):
    s = (text or '').lower()
    retry_tokens = (
        'timeout',
        'timed out',
        'connection reset',
        'connection aborted',
        'connection error',
        'temporarily unavailable',
        '502',
        '503',
        '504',
        '429',
    )
    return any(t in s for t in retry_tokens)


def main():
    parser = argparse.ArgumentParser(description='Upload model to HuggingFace Hub')
    parser.add_argument('folder_path', help='Path to folder to upload')
    parser.add_argument('repo_name', help='Repository name (without org prefix)')
    parser.add_argument('--token', default=None, help='HF API token')
    parser.add_argument('--org', default='', help='Organization (blank = personal account)')
    parser.add_argument(
        '--revision',
        default='',
        help='Branch/revision to upload to (e.g. 8.00bpw for multi-branch EXL3 repos)',
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print(json.dumps({'error': 'Missing HF token'}), flush=True)
        sys.exit(1)

    api = HfApi()

    # Determine owner
    try:
        user_info = api.whoami(token=token)
        owner = args.org if args.org else user_info['name']
    except Exception as e:
        print(json.dumps({'error': f'Auth failed: {str(e)[:300]}'}), flush=True)
        sys.exit(1)

    full_repo = f'{owner}/{args.repo_name}'
    log_status('Uploading', 0, f'Creating repo {full_repo}')

    # Create repo (idempotent)
    try:
        api.create_repo(
            repo_id=full_repo,
            token=token,
            repo_type='model',
            exist_ok=True,
        )
    except Exception as e:
        print(json.dumps({'error': f'Create repo failed: {str(e)[:300]}'}), flush=True)
        sys.exit(1)

    rev = (args.revision or '').strip() or None
    log_status(
        'Uploading',
        0.1,
        f'Uploading files to {full_repo}' + (f' (revision {rev})' if rev else ''),
    )

    # Single-commit upload_folder (stable). multi_commits uses experimental PR API and has failed in production.
    upload_kwargs = {
        'folder_path': args.folder_path,
        'repo_id': full_repo,
        'repo_type': 'model',
        'token': token,
        'commit_message': 'Upload EXL3 quantized model',
    }
    if rev:
        upload_kwargs['revision'] = rev

    last_error = None
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            api.upload_folder(**upload_kwargs)
            last_error = None
            break
        except Exception as e:
            last_error = e

        if last_error is None:
            break
        if attempt < max_attempts - 1 and is_retryable_error(str(last_error)):
            wait_s = 2 ** attempt
            log_status('Uploading', 0.1, f'Retrying upload after transient error ({attempt + 1}/{max_attempts})')
            time.sleep(wait_s)
            continue
        break

    if last_error is not None:
        err_type = last_error.__class__.__name__
        err_text = str(last_error)
        print(json.dumps({'error': f'Upload failed ({err_type}): {err_text[:1000]}'}), flush=True)
        sys.exit(1)

    repo_url = f'https://huggingface.co/{full_repo}'
    log_status('Uploading', 1.0, f'Upload complete: {repo_url}')

    tree_url = f'{repo_url}/tree/{rev}' if rev else repo_url
    print(
        json.dumps(
            {
                'status': 'success',
                'url': repo_url,
                'revision': rev or 'main',
                'tree_url': tree_url,
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
