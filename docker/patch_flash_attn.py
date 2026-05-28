"""Make exllamav3 fall back to SDPA when flash-attn is absent.

exllamav3 0.0.37 imports flash_attn unconditionally in
``modules/attention_fn/flash_attn_2.py`` and its dispatch wrappers call the
flash functions whenever the argument shapes match, so a missing or stubbed
flash-attn crashes during the calibration forward pass. This guards the import
and makes the three wrappers return None when flash-attn is unavailable; the
dispatcher then falls through to exllamav3's own torch-SDPA backend.

Idempotent: re-running is a no-op once patched. Pass the target file as argv[1]
for testing; otherwise it is located via importlib.
"""
import importlib.util
import os
import sys

IMPORT_OLD = "from flash_attn import flash_attn_func, flash_attn_with_kvcache, flash_attn_varlen_func"
IMPORT_NEW = (
    "try:\n"
    "    from flash_attn import flash_attn_func, flash_attn_with_kvcache, flash_attn_varlen_func\n"
    "    HAS_FLASH_ATTN = True\n"
    "except ImportError:\n"
    "    HAS_FLASH_ATTN = False\n"
    "    flash_attn_func = flash_attn_with_kvcache = flash_attn_varlen_func = None"
)
DEFS = (
    "def fn_flash_attn_with_kvcache(args: AttnArgs) -> torch.Tensor | None:",
    "def fn_flash_attn_func(args: AttnArgs) -> torch.Tensor | None:",
    "def fn_flash_attn_varlen_func(args: AttnArgs) -> torch.Tensor | None:",
)
GUARD = "    if not HAS_FLASH_ATTN:\n        return None"


def _target_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    spec = importlib.util.find_spec("exllamav3")
    if not spec or not spec.origin:
        return ""
    return os.path.join(
        os.path.dirname(spec.origin), "modules", "attention_fn", "flash_attn_2.py"
    )


def patch(text: str) -> str:
    if "HAS_FLASH_ATTN" in text:
        return text  # already patched
    if IMPORT_OLD not in text:
        raise SystemExit(f"flash_attn import line not found; exllamav3 layout changed")
    text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    for d in DEFS:
        if d not in text:
            raise SystemExit(f"wrapper not found: {d}")
        text = text.replace(d + "\n", d + "\n" + GUARD + "\n", 1)
    return text


def main() -> int:
    path = _target_path()
    if not path or not os.path.isfile(path):
        print(f"flash_attn_2.py not found at {path!r}", file=sys.stderr)
        return 1
    with open(path, encoding="utf-8") as f:
        original = f.read()
    patched = patch(original)
    if patched == original:
        print(f"already patched: {path}")
        return 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
