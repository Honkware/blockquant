#!/usr/bin/env python3
"""
Download a HuggingFace model to a local directory.
Uses snapshot_download for reliability and resume support.
Emits [STATUS] lines for the Node.js parent process.
"""
import argparse
import json
import os
import sys

try:
    import hf_transfer  # noqa: F401
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
except Exception:
    # Fall back to default transport if hf_transfer is unavailable.
    os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)


def log_status(stage, completion, status):
    msg = json.dumps({'stage': stage, 'completion': str(completion), 'status': status})
    print(f'[STATUS]{msg}[/STATUS]', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Download a HuggingFace model')
    parser.add_argument('model_id', help='e.g. meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('output_dir', help='Local directory to save files')
    parser.add_argument('--token', default=None, help='HF API token')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    token = args.token or os.environ.get('HF_TOKEN')

    log_status('Downloading', 0, f'Starting download of {args.model_id}')

    try:
        snapshot_download(
            repo_id=args.model_id,
            local_dir=args.output_dir,
            revision=args.revision,
            token=token,
            local_dir_use_symlinks=False,
            ignore_patterns=['*.md', '*.txt', '.gitattributes'],
        )
    except Exception as e:
        log_status('Error', 0, 'Download failed')
        print(f'Download failed: {str(e)[:500]}', file=sys.stderr)
        sys.exit(1)

    log_status('Downloading', 1.0, 'Download complete')
    print('Download complete.', flush=True)


if __name__ == '__main__':
    main()
