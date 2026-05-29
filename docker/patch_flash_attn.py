"""Make exllamav3 import and run without flash-attn, falling back to SDPA.

exllamav3 0.0.37 imports flash_attn unconditionally in several modules (the
attention dispatch, sliding-window attention, ...) and its dispatch wrappers
call the flash functions whenever arg shapes match. With flash-attn absent
that crashes at import and again during the calibration forward pass.

This walks the installed exllamav3 package and guards every
``from flash_attn import ...`` so the import never fails, and adds an early
``return None`` to the three dispatch wrappers in flash_attn_2.py so the
dispatcher falls through to exllamav3's own torch-SDPA backend. If some other
code path still calls a flash function directly it raises a clear TypeError
(the name is None) rather than silently producing wrong results.

Idempotent. Pass an explicit file or dir as argv[1] for testing; otherwise the
exllamav3 package is located via importlib.
"""
import importlib.util
import os
import re
import sys

IMPORT_RE = re.compile(r"^from flash_attn import (.+)$", re.MULTILINE)

DEFS = (
    "def fn_flash_attn_with_kvcache(args: AttnArgs) -> torch.Tensor | None:",
    "def fn_flash_attn_func(args: AttnArgs) -> torch.Tensor | None:",
    "def fn_flash_attn_varlen_func(args: AttnArgs) -> torch.Tensor | None:",
)
GUARD = "    if not HAS_FLASH_ATTN:\n        return None"


def _guarded_import(match: "re.Match") -> str:
    names = match.group(1).strip()
    # Names assigned to None on the fallback path so a later call fails loudly
    # instead of importing a missing symbol.
    targets = [n.split(" as ")[-1].strip() for n in names.split(",") if n.strip()]
    none_assign = " = ".join(targets) + " = None"
    return (
        "try:\n"
        f"    from flash_attn import {names}\n"
        "    HAS_FLASH_ATTN = True\n"
        "except ImportError:\n"
        "    HAS_FLASH_ATTN = False\n"
        f"    {none_assign}"
    )


def patch_text(text: str) -> str:
    if "HAS_FLASH_ATTN" in text or not IMPORT_RE.search(text):
        return text
    text = IMPORT_RE.sub(_guarded_import, text, count=1)
    for d in DEFS:
        if d in text:
            text = text.replace(d + "\n", d + "\n" + GUARD + "\n", 1)
    return text


def _iter_files(target: str):
    if os.path.isfile(target):
        yield target
        return
    for root, _dirs, files in os.walk(target):
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def main() -> int:
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        spec = importlib.util.find_spec("exllamav3")
        target = os.path.dirname(spec.origin) if spec and spec.origin else ""
    if not target or not os.path.exists(target):
        print(f"exllamav3 not found at {target!r}", file=sys.stderr)
        return 1

    patched = []
    for path in _iter_files(target):
        try:
            with open(path, encoding="utf-8") as f:
                original = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if "from flash_attn import" not in original:
            continue
        new = patch_text(original)
        if new != original:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            patched.append(path)

    if patched:
        print("patched: " + ", ".join(os.path.relpath(p, target) for p in patched))
    else:
        print("no unguarded flash_attn imports found (already patched or absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
