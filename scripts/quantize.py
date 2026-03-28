#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exllamav3'))

from exllamav3.conversion.convert_model import convert_model
from exllamav3.conversion.quant_config import EXL3QuantConfig

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bits', type=int, default=8)
    args = parser.parse_args()
    
    config = EXL3QuantConfig(bpw_override=args.bits)
    convert_model(args.input, args.output, config)
