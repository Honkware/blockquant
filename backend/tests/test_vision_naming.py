"""What the published name says about the vision tower.

The name is the only thing a downloader reads before the download, so a -V6 on
a tower that was copied fp16, or a missing one on a tower that was quantized,
is a lie about the weights. Both shipped: the SC path named off the request,
where 16 means "copy it" and None means "ask the arch", neither of which is a
tower bitrate.
"""
import ast
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# _name_parts_for imports cards by bare name, the way the pod lays it out
# flat beside quant.py. Without this the module only passes when another
# test file has already put that directory on the path -- which one did, so
# the dependency went unnoticed until this file was run on its own.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "blockquant"))
from blockquant import cards  # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


def _name_parts_for(*, sc, head_bits, vision_bits):
    """_name_parts_for lifted out of run(). Its closure vars become globals."""
    for node in ast.walk(ast.parse(SRC.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "_name_parts_for":
            ns = {"cards": cards, "sc": sc, "head_bits": head_bits,
                  "vision_bits": vision_bits}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)
            return ns["_name_parts_for"]
    pytest.fail("_name_parts_for is gone from quant.py")


@pytest.mark.parametrize("bits,want", [
    (None, None), (0, None), (16, None),   # not a quantized tower
    (4, 4), (6, 6), (8, 8),                # is one
    ("6", 6),                              # config.json round-trips ints as ints, but be safe
])
def test_only_a_quantized_tower_is_named(bits, want):
    assert cards.quantized_vision_bits(bits) == want


def test_a_copied_tower_gets_no_suffix():
    assert cards.exl3_repo_slug("M", "4.0", vision_bits=16) == "M-exl3-4.0bpw"
    assert cards.exl3_repo_slug("M", "4.0", sc=True, head_bits=6, vision_bits=16) \
        == "M-exl3-SC-4.0bpw-H6"


def test_a_quantized_tower_is_named():
    assert cards.exl3_repo_slug("M", "4.0", sc=True, head_bits=6, vision_bits=6) \
        == "M-exl3-SC-4.0bpw-H6-V6"


def test_sc_names_the_tower_the_converter_actually_wrote():
    # auto on a validated arch: the request said nothing, the converter wrote 6.
    f = _name_parts_for(sc=True, head_bits=None, vision_bits=None)
    assert f({"_head_bits": 6, "_vision_bits": 6}) == \
        {"sc": True, "head_bits": 6, "vision_bits": 6}
    # asked for 16; quantization_config records no vision_bits for a copy.
    f = _name_parts_for(sc=True, head_bits=6, vision_bits=16)
    assert f({"_head_bits": 6, "_vision_bits": None})["vision_bits"] is None


def test_sc_names_the_head_bits_the_converter_picked():
    f = _name_parts_for(sc=True, head_bits=None, vision_bits=None)
    assert f({"_head_bits": 6, "_vision_bits": None})["head_bits"] == 6


def test_a_plain_quant_states_only_what_was_pinned():
    # arch default: the plain name means "whatever the arch does", so say nothing
    f = _name_parts_for(sc=False, head_bits=None, vision_bits=None)
    assert f({"_head_bits": 6, "_vision_bits": 6}) == {}
    # pinned by the requester: that is a deliberate divergence, so name it
    f = _name_parts_for(sc=False, head_bits=None, vision_bits=4)
    assert f({"_head_bits": 6, "_vision_bits": 4}) == {"vision_bits": 4}
    # pinned to fp16: still not a quantized tower
    f = _name_parts_for(sc=False, head_bits=None, vision_bits=16)
    assert f({"_head_bits": 6, "_vision_bits": None}) == {}


def test_the_mode_column_calls_a_self_calibrated_quant_sc():
    mk = lambda v, sc, rid: {"variant": v, "head_bits": 6, "vision_bits": 6, "sc": sc,
                             "repo_id": rid, "size_gb": 0.5, "url": "u", "kl_div": 0.01}
    t = cards.build_quants_table(
        [mk("4.0", False, "o/m-exl3-4.0bpw"), mk("3.0", True, "o/m-exl3-SC-3.0bpw-H6-V6")],
        "3.0", current_repo_id="o/m-exl3-SC-3.0bpw-H6-V6")
    assert "SC&nbsp;H6&nbsp;V6" in t
    assert "plain&nbsp;V6" in t


def test_a_column_that_says_one_thing_on_every_row_is_not_drawn():
    mk = lambda v: {"variant": v, "head_bits": 6, "vision_bits": 6, "sc": False,
                    "repo_id": f"o/m-exl3-{v}bpw", "size_gb": 0.5, "url": "u"}
    assert "Mode" not in cards.build_quants_table([mk("4.0"), mk("5.0")], "4.0")


def test_the_card_rows_carry_the_sc_flag_the_column_reads():
    # build_quants_table read row["sc"] correctly; _finalize_cards never put it
    # there, so every SC card would have called itself plain.
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "quant_rows"):
            keys = {k.value for k in node.value.elt.keys if isinstance(k, ast.Constant)}
            assert "sc" in keys, f"quant_rows builds {sorted(keys)} -- no sc"
            return
    pytest.fail("no quant_rows in quant.py")


TRACE = "qbench · self-sampled trace · 40 rows"
WIKI2 = "qbench · wiki2 · 40×2048"


def _row(v, sc, kl, method, rid=None):
    return {"variant": v, "head_bits": 6, "vision_bits": 6, "sc": sc,
            "repo_id": rid or f"o/m-exl3-{v}bpw", "size_gb": 1.0, "url": "u",
            "kl_div": kl, "kl_method": method}


def test_a_tiny_kl_is_not_rounded_away():
    """The measured SC result is 4.35e-05 and .4f renders it as 0.0000.

    On-distribution KL sits two to three orders of magnitude below wiki2, so
    the format that suited one corpus erases the other -- and it erases the
    better number, which is the one the table exists to show.
    """
    t = cards.build_quants_table(
        [_row("3.0", False, 0.0001644, TRACE),
         _row("3.0", True, 0.00004346, TRACE, "o/m-exl3-SC-3.0bpw-H6-V6")],
        "3.0", n_params_b=0.8)
    assert "0.0000" not in t
    assert "4.35e-05" in t and "1.64e-04" in t


def test_wiki2_scale_numbers_stay_readable():
    t = cards.build_quants_table([_row("4.0", False, 0.0164, WIKI2)], "4.0", n_params_b=0.8)
    assert "0.0164" in t and "e-" not in t


def test_one_column_uses_one_format():
    # Mixing 0.0164 and 4.35e-05 in a column invites reading them as the same
    # kind of number.
    t = cards.build_quants_table(
        [_row("3.0", True, 0.00004346, TRACE), _row("4.0", False, 0.0164, TRACE)],
        "3.0", n_params_b=0.8)
    body = [l for l in t.splitlines() if l.startswith("|") and "BPW" not in l and ":--" not in l]
    assert all("e-" in l for l in body), body


def test_the_column_names_the_corpus_when_the_rows_agree():
    t = cards.build_quants_table([_row("3.0", True, 0.00004346, TRACE)], "3.0", n_params_b=0.8)
    assert "self-sampled" in t
    t2 = cards.build_quants_table([_row("4.0", False, 0.0164, WIKI2)], "4.0", n_params_b=0.8)
    assert "wiki2" in t2


def test_a_mixed_table_claims_neither_corpus():
    # The same quant measures 0.0000435 on its own output and 0.1144 on
    # wikitext-2. A table holding both must not label itself either way.
    t = cards.build_quants_table(
        [_row("3.0", True, 0.00004346, TRACE), _row("4.0", False, 0.0164, WIKI2)],
        "3.0", n_params_b=0.8)
    header = t.splitlines()[0]
    assert "self-sampled" not in header and "wiki2" not in header
