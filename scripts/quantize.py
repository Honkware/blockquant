#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exllamav3'))

from huggingface_hub import snapshot_download
from exllamav3.conversion.convert_model import convert_model
from exllamav3.conversion.quant_config import EXL3QuantConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='HF repo id or local path')
    parser.add_argument('--output', required=True)
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--cache', default=os.getenv('HF_CACHE_PATH', './.tmp/cache'))
    args = parser.parse_args()
    
    print(f"downloading {args.input}...")
    
    # download if it's an HF repo
    if not os.path.exists(args.input):
        model_path = snapshot_download(
            repo_id=args.input,
            cache_dir=args.cache,
            local_dir=os.path.join(args.cache, args.input.replace('/', '--'))
        )
    else:
        model_path = args.input
    
    print(f"quantizing to {args.bits}bit...")
    
    config = EXL3QuantConfig(bpw_override=args.bits)
    convert_model(model_path, args.output, config)
    
    print(f"done: {args.output}")

if __name__ == '__main__':
    main()
