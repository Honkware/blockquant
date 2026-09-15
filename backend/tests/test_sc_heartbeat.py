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
import select
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
    ns = {"os": os, "re": re, "time": time, "deque": deque, "select": select}
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
    assert "[sc] sc_trace:" in out and "row 0" in out, out


def test_it_does_not_print_every_fragment(heartbeat, capsys):
    proc = _run("import sys\n"
                "for i in range(200): sys.stdout.write(f'\\rrow {i}')\n")
    heartbeat(proc, "sc_trace", every=3600)  # never due
    proc.wait()
    assert "[sc] sc_trace:" not in capsys.readouterr().out


def test_a_silent_stage_still_reports(heartbeat, capsys):
    # THE case this exists for. A child that writes nothing for a while is
    # exactly what a hung pod looks like, and reading straight off the pipe
    # blocks until it writes -- so the first version of this reported nothing
    # precisely when reporting mattered. One line in three minutes, observed on
    # a live job while sc_trace loaded its donor.
    proc = _run("import time\ntime.sleep(2.4)\n")
    heartbeat(proc, "sc_trace", every=0.6)
    proc.wait()
    beats = [l for l in capsys.readouterr().out.splitlines() if "[sc] sc_trace:" in l]
    assert len(beats) >= 2, beats


def test_every_beat_differs_even_when_the_stage_says_nothing_new(heartbeat, capsys):
    # get_progress is `grep | tail`, so repeating one line leaves the
    # controller's progress text identical and its stall clock frozen -- the
    # same failure by another route. Elapsed seconds is what breaks the tie.
    # Beats are a whole second apart here because the elapsed stamp has
    # second resolution; the real interval is 30s, where it always differs.
    proc = _run("import sys,time\nsys.stdout.write('\\rrow 0')\ntime.sleep(3.4)\n")
    heartbeat(proc, "sc_trace", every=1.1)
    proc.wait()
    beats = [l for l in capsys.readouterr().out.splitlines() if "[sc] sc_trace:" in l]
    assert len(beats) >= 3, beats
    assert len(set(beats)) == len(beats), beats
    assert all("row 0" in b for b in beats), beats


def test_the_stages_run_unbuffered(heartbeat):
    """A quiet stage has to be launched with -u or the heartbeat sees nothing.

    turboderp's scripts print without flush=True, and their stdout here is a
    pipe, so Python block-buffers. sc_measure printed "Reference pass" and then
    nothing for 13 minutes of saturated CPU -- not because it was silent, but
    because 8K had not accumulated. sc_trace printed enough to keep flushing
    and hid the problem.
    """
    src = SRC.read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and getattr(getattr(node.func, "value", None), "id", "") == "subprocess"
                and getattr(node.func, "attr", "") == "Popen"):
            argv = node.args[0]
            flags = [e.value for e in argv.elts if isinstance(e, ast.Constant)]
            assert "-u" in flags, f"sc stages launched buffered: {flags}"
            return
    pytest.fail("no subprocess.Popen in quant.py")


def test_the_last_lines_survive_for_the_error_message(heartbeat):
    proc = _run("import sys\nsys.stdout.write('boom: no such file\\n')")
    tail = heartbeat(proc, "sc_measure", every=0)
    proc.wait()
    assert "boom: no such file" in tail
