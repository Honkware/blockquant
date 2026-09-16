"""The KL eval has to give recurrent state slots back between rows.

A hybrid / linear-attention model (Qwen3.5 and friends) takes a recurrent state
slot per forward. exllamav3 allocates one when params carries no
"recurrent_states" -- and the eval builds a fresh params dict per row, so every
row took a slot and none came back. The pool is the cache's max_batch_size,
default 16; kl_rows defaults to 40. So row 17 raised "Cannot create new state:
no available slots", the eval was best-effort, and the job published with no KL
number at all.

Can't be exercised here (needs exllamav3 and a GPU), so this pins the shape.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


def _forward_rows():
    for node in ast.walk(ast.parse(SRC.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_rows":
            return node
    pytest.fail("_forward_rows is gone from quant.py")


def test_the_row_loop_returns_its_slots():
    fn = _forward_rows()
    calls = {n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"reset_states"} & calls, (
        "the row loop never returns recurrent state slots; with kl_rows over "
        "max_batch_size the eval dies partway and the card loses its KL column")


def test_the_reset_is_inside_the_row_loop():
    # Resetting once after the loop would be the same bug with extra steps.
    fn = _forward_rows()
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
    assert loops, "no row loop in _forward_rows"
    assert any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
        and c.func.attr == "reset_states"
        for loop in loops for c in ast.walk(loop)
    ), "slots are returned outside the per-row loop"


def test_kl_rows_can_exceed_the_default_slot_pool():
    """Documents why this matters: the default row count is over the default pool."""
    src = SRC.read_text()
    assert 'cfg.get("kl_rows", 40)' in src, "kl_rows default moved; recheck against max_batch_size (16)"
