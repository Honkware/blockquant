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


@pytest.fixture
def has_results():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_has_results"]
    assert body, "_has_results is gone from quant.py"
    ns = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_has_results"]


HEADER = {"model": "/m", "rows": 10, "length": 1024, "mode": "iid", "results": []}


def test_the_header_sc_measure_writes_first_is_not_a_finished_stage(has_results, tmp_path):
    f = tmp_path / "measure.json"
    f.write_text(json.dumps(HEADER))
    assert f.stat().st_size > 0      # what the old check looked at
    assert not has_results(f)        # what it actually means


def test_a_measurement_with_results_is_done(has_results, tmp_path):
    f = tmp_path / "measure.json"
    f.write_text(json.dumps({**HEADER, "results": [{"key": "blk.0.attn_q", "rfn": 0.07}]}))
    assert has_results(f)


@pytest.mark.parametrize("content", ["", "{", '{"results": null}', "not json"])
def test_an_unreadable_or_truncated_file_is_not_done(has_results, tmp_path, content):
    # A stage killed mid-write leaves exactly this.
    f = tmp_path / "measure.json"
    f.write_text(content)
    assert not has_results(f)


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
