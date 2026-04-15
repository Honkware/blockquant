#!/usr/bin/env python3
"""
Pre-flight check: validates HF token write access and model existence.
Detects GGUF models and determines convertibility.
Outputs a single JSON line to stdout so Node can parse it.
"""
import argparse
import json
import os
import struct
import sys


# GGUF file type constants (from llama.cpp gguf.py)
GGUF_FILE_TYPE = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    6: "MOSTLY_Q5_0",
    7: "MOSTLY_Q5_1",
    8: "MOSTLY_Q8_0",
    9: "MOSTLY_Q2_K",
    10: "MOSTLY_Q3_K_S",
    11: "MOSTLY_Q3_K_M",
    12: "MOSTLY_Q3_K_L",
    13: "MOSTLY_Q4_K_S",
    14: "MOSTLY_Q4_K_M",
    15: "MOSTLY_Q5_K_S",
    16: "MOSTLY_Q5_K_M",
    17: "MOSTLY_Q6_K",
    18: "MOSTLY_IQ2_XXS",
    19: "MOSTLY_IQ2_XS",
    20: "MOSTLY_Q2_K_S",
    21: "MOSTLY_IQ3_XS",
    22: "MOSTLY_IQ3_XXS",
    23: "MOSTLY_IQ1_S",
    24: "MOSTLY_IQ4_NL",
    25: "MOSTLY_IQ3_M",
    26: "MOSTLY_IQ2_S",
    27: "MOSTLY_IQ2_M",
    28: "MOSTLY_IQ4_XS",
    29: "MOSTLY_IQ1_M",
    30: "MOSTLY_BF16",
    31: "MOSTLY_Q6_0",
    32: "MOSTLY_Q8_1",
    33: "MOSTLY_Q8_K",
    34: "MOSTLY_Q8_KV",
}

# File types that are suitable for conversion (full precision or 8-bit)
CONVERTIBLE_TYPES = {
    'ALL_F32', 'MOSTLY_F16', 'MOSTLY_BF16', 'MOSTLY_Q8_0', 'MOSTLY_Q8_1', 
    'MOSTLY_Q8_K', 'MOSTLY_Q8_KV'
}

# Quantized types (reject)
QUANTIZED_TYPES = {
    'MOSTLY_Q4_0', 'MOSTLY_Q4_1', 'MOSTLY_Q5_0', 'MOSTLY_Q5_1',
    'MOSTLY_Q2_K', 'MOSTLY_Q2_K_S', 'MOSTLY_Q3_K_S', 'MOSTLY_Q3_K_M', 'MOSTLY_Q3_K_L',
    'MOSTLY_Q4_K_S', 'MOSTLY_Q4_K_M', 'MOSTLY_Q5_K_S', 'MOSTLY_Q5_K_M', 'MOSTLY_Q6_K',
    'MOSTLY_Q6_0', 'MOSTLY_IQ2_XXS', 'MOSTLY_IQ2_XS', 'MOSTLY_IQ2_S', 'MOSTLY_IQ2_M',
    'MOSTLY_IQ3_XS', 'MOSTLY_IQ3_XXS', 'MOSTLY_IQ3_M', 'MOSTLY_IQ1_S', 'MOSTLY_IQ1_M',
    'MOSTLY_IQ4_NL', 'MOSTLY_IQ4_XS'
}


def read_gguf_header_from_file(file_path):
    """
    Read GGUF file header and return file type.
    Only reads the first ~64KB to avoid downloading entire file.
    """
    try:
        with open(file_path, 'rb') as f:
            # Read magic and version
            magic = f.read(4)
            if magic != b'GGUF':
                return None
            
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read tensor count and metadata KV count
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            
            # Parse metadata to find general.file_type
            for _ in range(metadata_kv_count):
                # Read key length and key
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8')
                
                # Read value type
                value_type = struct.unpack('<I', f.read(4))[0]
                
                # Read value based on type
                if value_type == 0:  # UINT8
                    f.read(1)
                elif value_type == 1:  # INT8
                    f.read(1)
                elif value_type == 2:  # UINT16
                    f.read(2)
                elif value_type == 3:  # INT16
                    f.read(2)
                elif value_type == 4:  # UINT32
                    val = struct.unpack('<I', f.read(4))[0]
                    if key == 'general.file_type':
                        return GGUF_FILE_TYPE.get(val, f'UNKNOWN_{val}')
                elif value_type == 5:  # INT32
                    f.read(4)
                elif value_type == 6:  # FLOAT32
                    f.read(4)
                elif value_type == 7:  # UINT64
                    f.read(8)
                elif value_type == 8:  # INT64
                    f.read(8)
                elif value_type == 9:  # FLOAT64
                    f.read(8)
                elif value_type == 10:  # BOOL
                    f.read(1)
                elif value_type == 11:  # STRING
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    f.read(str_len)
                elif value_type == 12:  # ARRAY
                    arr_type = struct.unpack('<I', f.read(4))[0]
                    arr_len = struct.unpack('<Q', f.read(8))[0]
                    # Skip array data (we don't need it)
                    for _ in range(arr_len):
                        if arr_type == 4:  # UINT32
                            f.read(4)
                        elif arr_type == 7:  # UINT64
                            f.read(8)
                        elif arr_type == 11:  # STRING
                            s_len = struct.unpack('<Q', f.read(8))[0]
                            f.read(s_len)
                        else:
                            break  # Unknown array type, skip
                else:
                    break  # Unknown type, stop parsing
                    
            return None  # general.file_type not found
    except Exception:
        return None


def detect_gguf_precision_from_file(repo_id, gguf_file, token=None):
    """
    Download just the header of a GGUF file and detect its precision.
    Returns tuple of (precision_type, can_convert, reason).
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Download only first 64KB to read header
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=gguf_file,
            token=token,
            local_dir=os.path.join(os.path.expanduser('~'), '.cache', 'blockquant_gguf_headers'),
            local_dir_use_symlinks=False
        )
        
        file_type = read_gguf_header_from_file(local_path)
        
        if file_type is None:
            return 'unknown', True, None
        
        if file_type in CONVERTIBLE_TYPES:
            precision = 'bf16' if 'BF16' in file_type else ('q8_0' if 'Q8' in file_type else 'fp16')
            return precision, True, None
        elif file_type in QUANTIZED_TYPES:
            return 'quantized', False, f'File is {file_type} (quantized). Need BF16, F16, FP32, or Q8_0 for conversion.'
        else:
            return 'unknown', True, None
            
    except Exception as e:
        return 'unknown', True, None


def detect_gguf_source(files, repo_id=None, token=None):
    """
    Detect GGUF files in repo and determine if conversion is possible.
    Uses pattern matching + header inspection for unknown files.
    
    Returns dict with:
    - isGguf: bool
    - ggufFile: str or None (best file to use)
    - ggufPrecision: str or None ('bf16', 'q8_0', 'quantized', None)
    - canConvert: bool
    - rejectReason: str or None
    """
    gguf_files = [f for f in files if f.endswith('.gguf')]
    
    if not gguf_files:
        return {
            'isGguf': False,
            'ggufFile': None,
            'ggufPrecision': None,
            'canConvert': True,
            'rejectReason': None
        }
    
    # Categorize files by precision patterns
    bf16_patterns = ['bf16', 'bfloat16', 'BF16', 'BFloat16', 'f16', 'F16', 'fp16', 'FP16']
    q8_patterns = ['q8_0', 'Q8_0', 'q8-', 'Q8-', '-8bit', '_8bit', '8-bit']
    quantized_patterns = ['q2_', 'q3_', 'q4_', 'q5_', 'q6_', 'Q2_', 'Q3_', 'Q4_', 'Q5_', 'Q6_',
                          '-q2-', '-q3-', '-q4-', '-q5-', '-q6-', '-Q2-', '-Q3-', '-Q4-', '-Q5-', '-Q6-',
                          'q2_k', 'q3_k', 'q4_k', 'q5_k', 'q6_k', 'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K']
    
    bf16_files = []
    q8_files = []
    quantized_files = []
    unknown_files = []
    
    for f in gguf_files:
        f_lower = f.lower()
        if any(p.lower() in f_lower for p in bf16_patterns):
            bf16_files.append(f)
        elif any(p.lower() in f_lower for p in q8_patterns):
            q8_files.append(f)
        elif any(p.lower() in f_lower for p in quantized_patterns):
            quantized_files.append(f)
        else:
            unknown_files.append(f)
    
    # Priority: BF16 > Q8_0 > Unknown (inspect header) > Quantized (reject)
    if bf16_files:
        # Prefer files with "bf16" or "BF16" in the name, then sort by size (largest first, likely full model)
        bf16_files.sort(key=lambda x: ('bf16' in x.lower(), len(x)), reverse=True)
        return {
            'isGguf': True,
            'ggufFile': bf16_files[0],
            'ggufPrecision': 'bf16',
            'canConvert': True,
            'rejectReason': None
        }
    
    if q8_files:
        q8_files.sort(key=lambda x: len(x), reverse=True)
        return {
            'isGguf': True,
            'ggufFile': q8_files[0],
            'ggufPrecision': 'q8_0',
            'canConvert': True,
            'rejectReason': None
        }
    
    if quantized_files:
        # Only quantized files found - cannot convert
        return {
            'isGguf': True,
            'ggufFile': quantized_files[0],
            'ggufPrecision': 'quantized',
            'canConvert': False,
            'rejectReason': (
                f'Only quantized GGUF files found ({quantized_files[0]}). '
                'Need BF16/bfloat16, F16/FP16, or Q8_0 for conversion. '
                'Double-quantization would cause severe quality loss.'
            )
        }
    
    # Unknown GGUF files - skip header download since GGUF is unsupported regardless
    if unknown_files:
        return {
            'isGguf': True,
            'ggufFile': unknown_files[0],
            'ggufPrecision': 'unknown',
            'canConvert': False,
            'rejectReason': 'GGUF files are not supported. Use a standard HuggingFace model repo with safetensors/bin weights.'
        }
    
    # Shouldn't reach here, but fallback
    return {
        'isGguf': True,
        'ggufFile': gguf_files[0],
        'ggufPrecision': 'unknown',
        'canConvert': True,
        'rejectReason': None
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', default=None, help='HuggingFace API token')
    parser.add_argument('--model', default=None, help='Model ID to check (org/name)')
    args = parser.parse_args()
    token = args.token or os.environ.get('HF_TOKEN')

    result = {
        'canWrite': False,
        'modelExists': None,
        'username': None,
        'error': None,
        'isGguf': False,
        'ggufFile': None,
        'ggufPrecision': None,
        'canConvert': True,
        'rejectReason': None
    }
    
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

        # Check model existence and GGUF status
        if args.model:
            try:
                api.model_info(repo_id=args.model, token=token)
                result['modelExists'] = True
                
                # List repo files to detect GGUF
                try:
                    files = api.list_repo_files(repo_id=args.model, token=token)
                    gguf_info = detect_gguf_source(files, repo_id=args.model, token=token)
                    result.update(gguf_info)
                except Exception as e:
                    # Non-fatal: model exists but we can't list files
                    result['isGguf'] = False
                    
            except Exception:
                result['modelExists'] = False

    except Exception as e:
        result['error'] = str(e)[:300]

    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
