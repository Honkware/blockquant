"""Which corpus the KL number is measured on.

turboderp ships qbench_prompts.py precisely because external corpora are the
wrong yardstick: "evaluating quants on external corpora measures divergence on
text the model may never produce itself ... raw web text is so far out of
distribution that the noise floor inflates and KLD ordering degrades."

Ordering is the whole point of the number -- it is what says which of two
quants is better. We had vendored that file, used the other half of the
toolkit, and hardcoded wiki2; and sc_trace writes a qbench-compatible trace on
every self-calibrated run that we used only as sc_measure's -tr and then threw
away with the pod.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


@pytest.fixture
def subset():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_trace_subset"]
    assert body, "_trace_subset is gone from quant.py"
    ns = {"json": json, "Path": Path, "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_trace_subset"]


def _trace(tmp_path, n):
    rows = [{"conversation": i, "turn": 0,
             "input_ids": [1, 2, 3], "response_ids": [4, 5]} for i in range(n)]
    f = tmp_path / "trace.json"
    f.write_text(json.dumps({"model": "m", "rows": rows}))
    return f



def test_a_large_trace_is_cut_to_the_requested_rows(subset, tmp_path):
    out = subset(_trace(tmp_path, 300), 40)
    assert len(json.loads(out.read_text())["rows"]) == 40


def test_it_strides_rather_than_taking_a_prefix(subset, tmp_path):
    # Rows come out in conversation order and the seed set is grouped by
    # domain, so the first 40 would all be the same few domains.
    out = subset(_trace(tmp_path, 300), 40)
    picked = [r["conversation"] for r in json.loads(out.read_text())["rows"]]
    assert picked != list(range(40))
    assert max(picked) > 200


def test_a_small_trace_is_used_whole(subset, tmp_path):
    out = subset(_trace(tmp_path, 12), 40)
    assert len(json.loads(out.read_text())["rows"]) == 12


def test_it_keeps_the_qbench_row_shape(subset, tmp_path):
    # test_trace scores only the response positions, so both fields must survive.
    row = json.loads(subset(_trace(tmp_path, 50), 10).read_text())["rows"][0]
    assert "input_ids" in row and "response_ids" in row


@pytest.mark.parametrize("content", ["", "{}", '{"rows": []}', "not json"])
def test_an_unusable_trace_falls_back_rather_than_failing(subset, tmp_path, content):
    f = tmp_path / "trace.json"
    f.write_text(content)
    assert subset(f, 40) is None


def test_a_missing_trace_falls_back(subset, tmp_path):
    assert subset(tmp_path / "nope.json", 40) is None


def test_the_eval_prefers_a_trace_and_keeps_wiki2_as_the_fallback():
    src = SRC.read_text()
    i = src.index("def _kl_div_eval")
    j = src.index("def ", i + 10)
    body = src[i:j]
    assert "test_trace" in body, "the eval cannot use a self-sampled trace"
    assert '"source": "wiki2"' in body, "wiki2 fallback is gone for models with no trace"
    # The method string has to say which, or two cards are silently incomparable.
    assert "self-sampled trace" in body


def test_any_variant_with_a_trace_on_disk_is_scored_on_it():
    """Not only the self-calibrated ones.

    A plain quant of a model whose trace is present -- restored from the cache,
    or written by an SC variant earlier in the same job -- has to be scored on
    the same corpus. Otherwise both land in one card table looking comparable
    while being measured against different distributions, which is the exact
    mistake that made SC look worse than plain.
    """
    calls = [n for n in ast.walk(ast.parse(SRC.read_text()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_kl_div_eval"]
    assert calls, "no _kl_div_eval call in quant.py"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        if "trace_path" not in kw:
            continue
        expr = ast.unparse(kw["trace_path"])
        assert "is_file" in expr, f"the trace is not chosen by whether it exists: {expr}"
        assert "sc" not in expr.split("."), f"still gated on the job being SC: {expr}"
