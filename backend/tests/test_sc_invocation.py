"""How sc_measure is invoked, against what its own documentation asks for.

A self-calibrated 3.0bpw came out at 0.1200 median KL where a plain quant of
the same bitrate, head bits and vision bits measured 0.0782 -- worse, from a
pipeline that ran without a single error. The cause was the invocation, not
the pipeline:

  - iid noise, where the script's header says the iid default runs a median
    1.95x KL overestimate with per-type structure up to 5.8x on v_proj, and
    that "the bias is almost entirely the missing LDLQ error shaping". A
    recipe fitted to that protects the wrong tensors.
  - global --rfn, where sc_rfn_probe had already measured per-tensor anchors.
    rfn.json was generated, passed to sc_optimize, and never given to the
    measurement that -rr exists for.

Both are one flag. Neither would ever fail loudly.
"""
import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"
MEASURE = Path(__file__).resolve().parent.parent / "src/blockquant/selfcal/sc_measure.py"


def _stage_argv(name: str) -> list:
    for node in ast.walk(ast.parse(SRC.read_text())):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "stage"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == name):
            return [a.value if isinstance(a, ast.Constant) else "<expr>"
                    for a in node.args[1].elts]
    pytest.fail(f"no stage({name!r}) call in quant.py")


def test_the_measurement_uses_shaped_noise():
    argv = _stage_argv("sc_measure")
    assert "--shaped" in argv or "-sh" in argv, (
        "sc_measure is measuring with iid noise, which its own header documents as "
        "a median 1.95x KL overestimate, up to 5.8x per-type")


def test_shaped_is_what_the_script_recommends():
    # Guards the reasoning, not just the flag: if upstream ever stops
    # recommending it, this says so instead of silently disagreeing.
    help_text = MEASURE.read_text()
    m = re.search(r'"-sh",\s*"--shaped".*?help\s*=\s*"([^"]+)"', help_text, re.S)
    assert m, "the --shaped flag is gone from sc_measure"
    assert "recommend" in m.group(1).lower(), m.group(1)


def test_the_measurement_is_anchored_to_the_probe():
    argv = _stage_argv("sc_measure")
    assert "-rr" in argv or "--rfn_ref" in argv, (
        "sc_rfn_probe's per-tensor anchors never reach sc_measure, so every tensor "
        "is probed at the same global --rfn")


def test_the_probe_output_is_what_gets_passed():
    # -rr takes sc_rfn_probe's JSON. Handing it anything else is the same class
    # of error as feeding sc_measure the trace JSON instead of the packed
    # safetensors, which is documented in _run_sc_stages.
    src = SRC.read_text()
    i = src.index('stage("sc_measure"')
    j = src.index("done=", i)
    assert re.search(r'"-rr",\s*str\(rfn\)', src[i:j]), src[i:j]


def test_optimize_still_gets_its_own_anchors():
    argv = _stage_argv("sc_optimize")
    assert "-rr" in argv, "sc_optimize lost its rfn anchors"


def test_the_trace_is_the_packed_safetensors_not_the_json():
    # Already true, and worth keeping true: -tr is documented as "Packed
    # self-sampled trace (safetensors from sc_trace.py)", and the JSON would
    # fall back to the bundled corpus without saying so.
    src = SRC.read_text()
    i = src.index('stage("sc_measure"')
    j = src.index("done=", i)
    assert re.search(r'"-tr",\s*str\(cal\)', src[i:j]), src[i:j]


def test_a_shaped_measurement_cannot_be_served_from_an_iid_cache():
    """The cache key has to move when the measurement's meaning does."""
    body = SRC.read_text()
    i = body.index("def _sc_cache_key")
    j = body.index("def ", i + 10)
    key = body[i:j]
    assert "noise" in key, "cache key does not record the noise model"
    assert '"schema": 2' in key, "cache schema was not bumped past the iid entries"
