"""Patch exllamav3 attn.py: view -> reshape in decode_flash_attn_nc.

flash-attn returns a non-contiguous tensor when global_head_dim differs from
head_dim (Gemma4 alternates local SWA and global attention layers). .view()
raises RuntimeError on non-contiguous tensors; .reshape() handles both.
"""
import importlib.util
import pathlib
import sys

OLD = "o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))"
NEW = "o = o.reshape((bsz, seqlen, self.num_q_heads * self.head_dim))"

spec = importlib.util.find_spec("exllamav3")
if not spec or not spec.origin:
    print("exllamav3 not found, skipping patch", file=sys.stderr)
    sys.exit(0)

p = pathlib.Path(spec.origin).parent / "modules" / "attn.py"
text = p.read_text(encoding="utf-8")

if NEW in text:
    print(f"already patched: {p}")
    sys.exit(0)

if OLD not in text:
    print(f"target line not found in {p}, skipping", file=sys.stderr)
    sys.exit(0)

p.write_text(text.replace(OLD, NEW), encoding="utf-8")
n = p.read_text().count(NEW)
print(f"patched {n} occurrence(s) in {p}")
