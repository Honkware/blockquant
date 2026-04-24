#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import runpod
runpod.api_key = os.environ.get("RUNPOD_API_KEY", "")

gpus = runpod.get_gpus()
for g in gpus:
    print(f"{g.get('id'):35s} | {g.get('displayName'):25s} | {g.get('memoryInGb')} GB | ${g.get('communityPrice', 'N/A')}/hr")
