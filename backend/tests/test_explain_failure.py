"""What a failed job says on its Discord card: one line, and only what is known.

A converter killed from outside prints nothing, so the runner writes the exit
code and the cgroup's OOM-kill count as the last log line.

AST-loaded like test_drain_failure: the launcher needs the venv at import.
"""
import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts/run_runpod_job.py"

# The end of a real log, trimmed.
KILLED_AT_HEAD = [
    "[progress] quantize 4.0 43% (layer 28/64) eta 24m",
    " -- Loading unquantized module: model.language_model.layers.63",
    " -- Quantized: model.language_model.layers.63      bpw:  4.01  rfn: 0.017521  [42.44 s]  rss: 15.73 GB",
    " -- Loading unquantized module: model.language_model.norm",
    " -- Loading unquantized module: lm_head",
    " -- Captured: lm_head",
    " -- Hessians: block",
]


@pytest.fixture(scope="module")
def explain():
    want = ("_FRAME", "_last_exception", "_where", "_exit_info", "_explain")
    body = [n for n in ast.parse(SRC.read_text()).body
            if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in want)]
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["_explain"]


def test_an_oom_kill_names_ram_and_the_module(explain):
    got = explain(KILLED_AT_HEAD + ["[exit] code=137 oom_kills=1 mem_limit=49392123904"], "no_result")
    assert got == "Out of system RAM at lm_head (49 GB pod)"


def test_a_kill_without_the_counter_says_only_what_is_known(explain):
    got = explain(KILLED_AT_HEAD + ["[exit] code=137 oom_kills=? mem_limit=?"], "no_result")
    assert got == "Killed by signal 9 at lm_head"


def test_cuda_oom_names_the_card(explain):
    got = explain(KILLED_AT_HEAD, "failed", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.7 GiB", 24)
    assert got == "Out of GPU memory at lm_head (24 GB card)"


def test_a_layer_reads_one_based_with_the_total(explain):
    lines = ["[progress] quantize 4.0 64% (layer 41/64) eta 9m",
             " -- Loading unquantized module: model.layers.40",
             "[exit] code=139 oom_kills=0 mem_limit=max"]
    assert explain(lines, "no_result") == "Segfault at layer 41 of 64"


def test_the_filtered_poll_tail_still_places_it(explain):
    assert explain(["[progress] quantize 4.0 43% (layer 28/64) eta 24m"], "stalled") \
        == "No output for too long, stopped at layer 28 of 64"


def test_disk_full_and_other_errors(explain):
    assert explain(["[download] 43% (1/2 GB)", "OSError: [Errno 28] No space left on device"],
                   "failed") == "Disk full while downloading"
    got = explain(["[upload] 4.0 -> o/r ...",
                   "huggingface_hub.errors.HfHubHTTPError: (Request ID: Root=1-abc)",
                   "403 Forbidden: storage quota"], "failed")
    assert got == "HfHubHTTPError: 403 Forbidden: storage quota (while uploading)"


def test_no_raw_log_text_reaches_the_card(explain):
    assert "Hessians" not in explain(KILLED_AT_HEAD, "no_result")


def test_the_runner_reports_a_kill(tmp_path):
    src = (ROOT / "src/blockquant/providers/runpod/provider.py").read_text()
    runner = tmp_path / "run.sh"
    runner.write_text(src.split('RUNNER = r"""')[1].split('"""')[0])
    out = subprocess.run(["sh", str(runner), "python3", "-c",
                          "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
                         capture_output=True, text=True).stdout
    assert re.search(r"^\[exit\] code=137 oom_kills=\S+ mem_limit=\S+$", out, re.M)


def test_quant_py_runs_under_the_runner():
    src = (ROOT / "src/blockquant/providers/runpod/provider.py").read_text()
    assert "nohup setsid sh {REMOTE_RUNNER} {self._remote_py} -X faulthandler {script_path}" in src
