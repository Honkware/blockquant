"""Drafting on the trace stage, and refusing to guess about it.

sc_trace is pure generation and the longest thing in an SC job, so it is the
one stage where speculative decoding pays. exllamav3 exposes -mtp through
model_init and sc_trace passes model_init options straight through; we had
never used it.

Declaring mtp_* in config is not evidence: fine-tunes inherit those keys and
ship none of the tensors, which is the whole reason _disable_missing_mtp
exists. So the weights decide.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


@pytest.fixture
def has_mtp():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_has_mtp"]
    assert body, "_has_mtp is gone from quant.py"
    ns = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_has_mtp"]


def _index(tmp_path, keys):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "m.safetensors" for k in keys}}))
    return tmp_path


def test_weights_present_means_yes(has_mtp, tmp_path):
    # The real shape: Swift-Qwen3.8-27b ships 15 of these.
    assert has_mtp(_index(tmp_path, [
        "mtp.fc.weight", "mtp.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight"]))


def test_no_weights_means_no(has_mtp, tmp_path):
    # A model that inherited the config keys and shipped nothing: drafting
    # would fail on a pod, which is what _disable_missing_mtp cleans up after.
    assert not has_mtp(_index(tmp_path, [
        "model.layers.0.self_attn.q_proj.weight", "lm_head.weight"]))


def test_a_directory_with_nothing_readable_is_no(has_mtp, tmp_path):
    assert not has_mtp(tmp_path)
    assert not has_mtp(tmp_path / "nope")


def test_a_broken_index_is_no_rather_than_a_crash(has_mtp, tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{not json")
    assert not has_mtp(tmp_path)


def test_drafting_is_attempted_only_where_the_weights_are():
    src = SRC.read_text()
    i = src.index('trace_argv = ["-m", str(donor_dir)')
    j = src.index("sc_rfn_probe", i)
    block = src[i:j]
    assert "_has_mtp(donor_dir)" in block, (
        "drafting is not gated on the donor actually having MTP weights")
    assert '"-mtp"' in block


def test_a_drafting_failure_falls_back_instead_of_losing_the_pod():
    # Drafting is the most model-specific thing in the chain, and sc_trace is
    # the stage that has already cost 25 minutes by the time it would fail.
    src = SRC.read_text()
    i = src.index('trace_argv = ["-m", str(donor_dir)')
    j = src.index("sc_rfn_probe", i)
    block = src[i:j]
    assert "except RuntimeError" in block, "an MTP failure would kill the job"
    assert block.count('stage("sc_trace"') >= 2, "there is no retry without -mtp"


def test_tensor_parallel_is_still_only_for_multi_gpu():
    src = SRC.read_text()
    i = src.index('trace_argv = ["-m", str(donor_dir)')
    j = src.index("sc_rfn_probe", i)
    assert "gpu_count > 1" in src[i:j]
