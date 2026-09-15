"""_heartbeat: a long SC stage has to keep talking.

capture_output meant a stage printed one line and went silent until it exited.
The controller's stall clock advances only when the progress text changes, so a
stage that outlives stall_timeout reads as a hung pod and the controller kills
a working one. These pin the two things that makes non-trivial: turboderp's
scripts draw tqdm bars with carriage returns rather than newlines, and emitting
every fragment would flood the log instead.

Loaded by AST because remote/quant.py imports the exllamav3 stack at import.
"""
import ast
import os
import re
import subprocess
import sys
import time
import types
from collections import deque
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


@pytest.fixture
def heartbeat():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_heartbeat"]
    assert body, "_heartbeat is gone from quant.py"
    ns = {"os": os, "re": re, "time": time, "deque": deque}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_heartbeat"]


def _run(script: str):
    return subprocess.Popen([sys.executable, "-u", "-c", script],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_a_tqdm_bar_is_split_on_carriage_returns(heartbeat, capsys):
    # One "line" of 5 bar updates, no newline until the end -- a readline loop
    # sees nothing here until the process exits.
    proc = _run("import sys\n"
                "for i in range(5): sys.stdout.write(f'\\rstep {i}/5')\n"
                "sys.stdout.write('\\ndone\\n')")
    tail = heartbeat(proc, "sc_trace", every=0)
    proc.wait()
    assert "step 4/5" in tail
    assert tail[-1] == "done"


def test_it_reports_while_the_stage_is_still_running(heartbeat, capsys):
    # The child emits row 0 at once, then lives another 0.6s. Seeing row 0 in
    # the output means it was reported before the process exited -- which is
    # the whole property, since capture_output produced row 0 as well, just
    # not until the stage was already over.
    proc = _run("import sys,time\n"
                "sys.stdout.write('\\rrow 0')\n"
                "time.sleep(0.6)\n"
                "sys.stdout.write('\\rrow 1\\n')\n")
    heartbeat(proc, "sc_trace", every=0)
    proc.wait()
    out = capsys.readouterr().out
    # Prefixed so it survives the controller's _PROGRESS_MARKERS filter.
    assert "[sc] sc_trace: row 0" in out, out


def test_it_does_not_print_every_fragment(heartbeat, capsys):
    proc = _run("import sys\n"
                "for i in range(200): sys.stdout.write(f'\\rrow {i}')\n")
    heartbeat(proc, "sc_trace", every=3600)  # never due
    proc.wait()
    assert "[sc] sc_trace:" not in capsys.readouterr().out


def test_the_last_lines_survive_for_the_error_message(heartbeat):
    proc = _run("import sys\nsys.stdout.write('boom: no such file\\n')")
    tail = heartbeat(proc, "sc_measure", every=0)
    proc.wait()
    assert "boom: no such file" in tail
