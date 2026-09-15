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
