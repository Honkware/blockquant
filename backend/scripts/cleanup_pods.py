#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import runpod
runpod.api_key = os.environ.get("RUNPOD_API_KEY", "")

# Keep the most recent blockquant pod alive (user's current job)
keep = sys.argv[1] if len(sys.argv) > 1 else None

pods = runpod.get_pods()
for p in pods:
    pod_id = p["id"]
    name = p.get("name", "?")
    if keep and pod_id == keep:
        print(f"KEEPING {pod_id} ({name})")
        continue
    print(f"TERMINATING {pod_id} ({name})...")
    try:
        runpod.terminate_pod(pod_id)
        print(f"  terminated")
    except Exception as e:
        print(f"  error: {e}")
