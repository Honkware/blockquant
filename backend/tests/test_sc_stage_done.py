"""A streaming stage is finished when its file has contents, not when it exists.

sc_measure --streaming writes its header and an empty "results" within seconds
of starting and keeps that shape for the rest of the stage -- observed on the
first real SC job: 257 bytes, results [], five seconds in, with half an hour
still to run. Judged on size the stage looks complete, so a resumed job skips
the measurement and a crash halfway reads as success. Both hand sc_optimize an
empty measurement, and the recipe it builds from that is what the quant gets
converted against. No error anywhere.
"""
import ast
import json
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


WANT = ("_rfn_count", "_measured_all")


@pytest.fixture
def mod():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in WANT]
    assert len(body) == len(WANT), f"missing from quant.py: {WANT}"
    ns = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


@pytest.fixture
def done(mod, tmp_path):
    """done(measure_obj, n_rfn) -> the predicate's verdict."""
    def run(measure_obj, n_rfn=151):
        m = tmp_path / "measure.json"
        r = tmp_path / "rfn.json"
        m.write_text(measure_obj if isinstance(measure_obj, str) else json.dumps(measure_obj))
        r.write_text(json.dumps({"results": [{"key": f"t{i}"} for i in range(n_rfn)]}))
        return mod["_measured_all"](m, r)
    return run


HEADER = {"model": "/m", "rows": 10, "length": 1024, "mode": "iid", "results": []}


def _measured(n):
    return {**HEADER, "results": [{"key": f"t{i}", "rfn": 0.07} for i in range(n)]}


def test_the_header_sc_measure_writes_first_is_not_a_finished_stage(done):
    assert not done(HEADER)


def test_a_partial_measurement_is_not_a_finished_stage(done):
    # The one non-empty made look finished. sc_measure resumes from a partial
    # file by design, so this is a normal crash state, not an exotic one --
    # and sc_optimize will build a recipe from a third of the model.
    assert not done(_measured(50))


def test_a_complete_measurement_is_done(done):
    # 151 of 151: the counts the first real SC run produced.
    assert done(_measured(151))


def test_more_results_than_expected_is_still_done(done):
    assert done(_measured(151), n_rfn=140)


@pytest.mark.parametrize("content", ["", "{", '{"results": null}', "not json"])
def test_an_unreadable_or_truncated_file_is_not_done(done, content):
    # A stage killed mid-write leaves exactly this.
    assert not done(content)


def test_an_empty_probe_means_re_run_not_trust(done):
    # With no expected count there is nothing to check against, so the stage
    # has to run again rather than be assumed complete. An `or` chain read the
    # empty list as missing and returned the wrapper dict's length, 1, which
    # any non-empty measurement clears.
    assert not done(_measured(151), n_rfn=0)


def test_a_missing_probe_means_re_run_not_trust(mod, tmp_path):
    m = tmp_path / "measure.json"
    m.write_text(json.dumps(_measured(151)))
    assert not mod["_measured_all"](m, tmp_path / "nope.json")


def test_sc_measure_is_the_stage_that_gets_the_content_check():
    # The wiring matters as much as the predicate: passing it to the wrong
    # stage leaves sc_measure judged on size again.
    src = SRC.read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "stage"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "sc_measure"):
            assert any(k.arg == "done" for k in node.keywords), \
                "sc_measure is judged on file size again"
            return
    pytest.fail("no sc_measure stage() call in quant.py")


def test_sc_measure_is_not_forced_onto_the_streaming_path():
    """Streaming is the CPU-bound fallback, not the default.

    It walks one module at a time and keeps cached states in system RAM: a
    0.8B measured that way took 12-16 minutes at ~13 cores with the GPU at 7%,
    on a card many times larger than the model. --load-mode auto checks free
    VRAM first and still falls back to streaming, both on the estimate and if
    the load OOMs, so forcing it only ever gave up speed.
    """
    src = SRC.read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "stage"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "sc_measure"):
            argv = [a.value for a in node.args[1].elts if isinstance(a, ast.Constant)]
            assert "--streaming" not in argv, "sc_measure pinned to the CPU-bound path"
            assert "auto" in argv, f"sc_measure should pick its load mode: {argv}"
            return
    pytest.fail("no sc_measure stage() call in quant.py")
