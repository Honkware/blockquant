"""Stub exllamav3's FormatronFilter for a quant-only image.

exllamav3's top-level __init__ hard-imports FormatronFilter, and formatron is
fragile against pinned pydantic in many base images, so `import exllamav3` can
fail before any quantization runs. The quant path never builds a filter, so we
replace exllamav3/generator/filter/formatron.py with a stub that imports
cleanly and only raises if something actually instantiates it.

Locates the package via find_spec (no import, no GPU), so it runs on a
GPU-less CI builder. Mirrors the in-pod bootstrap's formatron handling.
"""

import importlib.util
import os
import sys

STUB = (
    "class FormatronFilter:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        raise RuntimeError('FormatronFilter stubbed out (quant-only image)')\n"
)

spec = importlib.util.find_spec("exllamav3")
if not spec or not spec.origin:
    print("exllamav3 not found", file=sys.stderr)
    sys.exit(1)

path = os.path.join(os.path.dirname(spec.origin), "generator", "filter", "formatron.py")
if not os.path.exists(path):
    # Nothing to stub (layout changed upstream); leave it to the runtime probe.
    print(f"formatron.py not found at {path}; skipping stub")
    sys.exit(0)

with open(path, "w") as f:
    f.write(STUB)
print(f"stubbed {path}")
