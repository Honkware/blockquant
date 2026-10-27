"""What sc_measure is told it may spend on pending states, and what survives a crash.

The first 27B self-calibration ran sc_trace for 133 minutes, finished the probe,
and then lost its pod 44 modules into sc_measure: the heartbeat showed the
pending count climbing, 201 to 566 at about 13 per module in a single pass, and
then the container stopped answering, sshd included. sc_measure sizes that
fan-out from MemAvailable in /proc/meminfo, which inside a container is the
host's memory. And because the cache only saved once all four stages were done,
the 133 minutes went with the pod.
"""
import ast
import os
import re
import select
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"
GB = 1024 ** 3


def _lift(name, ns):
    for node in ast.parse(SRC.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)
            return ns[name]
    pytest.fail(f"{name} is gone from quant.py")


@pytest.fixture
def budget():
    return _lift("_container_ram_budget_gb", {"Path": Path})


def _v2(root, limit, current, inactive=0):
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.max").write_text(str(limit))
    (root / "memory.current").write_text(str(current))
    (root / "memory.stat").write_text(f"anon 1\ninactive_file {inactive}\nactive_file 2\n")
    return root


def _v1(root, limit, usage, inactive=0):
    m = root / "memory"
    m.mkdir(parents=True, exist_ok=True)
    (m / "memory.limit_in_bytes").write_text(str(limit))
    (m / "memory.usage_in_bytes").write_text(str(usage))
    (m / "memory.stat").write_text(f"cache 1\ntotal_inactive_file {inactive}\n")
    return root


def test_v2_budget_comes_from_the_container_limit(budget, tmp_path):
    root = _v2(tmp_path, 120 * GB, 20 * GB)
    assert budget(0.4, str(root)) == pytest.approx(40.0)


def test_v1_budget_comes_from_the_container_limit(budget, tmp_path):
    root = _v1(tmp_path, 120 * GB, 20 * GB)
    assert budget(0.4, str(root)) == pytest.approx(40.0)


def test_page_cache_is_not_counted_as_used(budget, tmp_path):
    # Reading 52 GB of safetensors fills page cache charged to the cgroup, but
    # it gives way under pressure. Counting it would starve the measurement into
    # dozens of passes for no reason.
    root = _v2(tmp_path, 120 * GB, 80 * GB, inactive=60 * GB)
    assert budget(0.4, str(root)) == pytest.approx(40.0)


def test_no_limit_means_no_override(budget, tmp_path):
    # Then MemAvailable really is ours, and sc_measure's default is correct.
    assert budget(0.4, str(_v2(tmp_path / "a", "max", 1 * GB))) is None
    assert budget(0.4, str(_v1(tmp_path / "b", 2 ** 63 - 4096, 1 * GB))) is None


def test_no_cgroup_at_all_means_no_override(budget, tmp_path):
    assert budget(0.4, str(tmp_path / "nothing-here")) is None


def test_it_never_hands_back_less_than_a_floor(budget, tmp_path):
    # A budget of nearly zero would plan a pass per module and crawl.
    root = _v2(tmp_path, 10 * GB, 9 * GB)
    assert budget(0.4, str(root)) == 4.0


def test_an_exhausted_container_returns_none_rather_than_a_negative(budget, tmp_path):
    root = _v2(tmp_path, 10 * GB, 11 * GB)
    assert budget(0.4, str(root)) is None


def test_the_cache_saves_after_each_expensive_stage():
    """Structural: on_built fires from inside stage(), for the three cacheable
    stages, and the post-run save that waited for all four is gone."""
    src = SRC.read_text()
    fn = src[src.index("def _run_sc_stages"):src.index("def _selfcal_dir")]
    assert "on_built(name)" in fn
    assert '("sc_trace", "sc_rfn_probe", "sc_measure")' in fn
    caller = src[src.index("sc_timings: dict = {}"):src.index('print(f"[quantize] {variant} bpw')]
    assert "on_built=_save" in caller
    assert "_built = any(" not in caller, "the all-stages-or-nothing save is back"


@pytest.fixture
def heartbeat():
    return _lift("_heartbeat", {"os": os, "re": re, "time": time, "deque": deque,
                                "select": select})


def test_a_status_line_is_forwarded_before_the_next_beat(heartbeat, capsys):
    """The fan-out plan and its RAM budget print once. Waiting for the next
    timed beat meant later output overwrote them, so the line that would have
    explained the 27B failure never left the pod."""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "print(' -- Fan-out: 9001 experiments in 1 pass(es); budget 512.0 GiB');"
         "print('noise'); print('more noise')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    heartbeat(proc, "sc_measure", every=3600)      # never due on the clock
    proc.wait()
    out = capsys.readouterr().out
    assert "Fan-out: 9001 experiments" in out
    assert "noise" not in out, "ordinary lines should still wait for a beat"


def test_a_warning_line_is_forwarded_too(heartbeat, capsys):
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", "print(' !! Collected only 400,000 of 512,000')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    heartbeat(proc, "sc_trace", every=3600)
    proc.wait()
    assert "Collected only 400,000" in capsys.readouterr().out
