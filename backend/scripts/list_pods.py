#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import runpod
runpod.api_key = os.environ.get("RUNPOD_API_KEY", "")

pods = runpod.get_pods()
print(f"Total pods: {len(pods)}")
for p in pods:
    print(f"  {p['id']} | {p.get('name','?')} | {p.get('desiredStatus','?')} | {p.get('gpu','?')}")
