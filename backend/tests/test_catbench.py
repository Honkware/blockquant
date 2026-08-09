"""CatBench: what we pull out of a model reply, and what the sandbox lets run.

remote/catbench.py is stdlib-only at module scope, so it imports directly.
size_check lives in scripts/run_catbench_job.py, which pulls the provider stack
at import time, so it is AST-loaded the way test_drain_failure does it.
"""
import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockquant.remote import catbench as cb  # noqa: E402

JOB = "scripts/run_catbench_job.py"
QUANT_JOB = "scripts/run_runpod_job.py"


def _load(src, want, ns):
    """Exec just the named top-level defs/assigns of a script into ns."""
    body = [n for n in ast.parse(open(src).read()).body
            if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in want)]
    exec(compile(ast.Module(body=body, type_ignores=[]), src, "exec"), ns)
    return ns


class _P:
    """Stand-in for RunPodProvider: only the size lookup is ever called."""
    gb = None

    @staticmethod
    def _base_download_gb(model_id, token=""):
        return _P.gb


@pytest.fixture(scope="module")
def job():
    """run_catbench_job.py's pure helpers, with the /quant imports stubbed."""
    import re
    ns = {"RunPodProvider": _P, "re": re}
    _load(QUANT_JOB, ("_FRAME", "_last_exception"), ns)
    _load(JOB, ("size_check", "DEFAULT_MAX_GB", "failure_reason"), ns)
    return ns


@pytest.fixture(scope="module")
def size_check(job):
    return job["size_check"], _P


# ── Prompts are the benchmark; they must not drift ──────────────────────────

def test_prompts_are_verbatim():
    assert cb.PROMPT_SVG == "Create a detailed SVG image of a cute kitten."
    assert cb.PROMPT_PY == "Write a Python script that draws a cute kitten using matplotlib."


# ── Pulling the answer out of a chatty reply ────────────────────────────────

def test_extract_svg_ignores_the_prose_around_it():
    reply = "Sure! Here you go:\n```xml\n<svg viewBox='0 0 10 10'><circle r='3'/></svg>\n```\nCute, right?"
    assert cb.extract_svg(reply) == "<svg viewBox='0 0 10 10'><circle r='3'/></svg>"


def test_extract_svg_returns_none_when_there_is_none():
    assert cb.extract_svg("I cannot draw.") is None
    assert cb.extract_svg("") is None


def test_extract_python_prefers_the_biggest_fenced_block():
    reply = (
        "First install it:\n```bash\npip install matplotlib\n```\n"
        "Then:\n```python\nimport matplotlib.pyplot as plt\nplt.plot([1,2])\nplt.show()\n```\n"
    )
    got = cb.extract_python(reply)
    assert got.startswith("import matplotlib")
    assert "pip install" not in got


def test_extract_python_accepts_an_unfenced_script():
    assert cb.extract_python("import matplotlib.pyplot as plt\nplt.show()").startswith("import")


def test_extract_python_rejects_a_refusal():
    assert cb.extract_python("Sorry, I would rather not.") is None


# ── The sandbox. This is the part that runs model-written code. ─────────────

KITTEN = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(3, 3))
ax.add_patch(plt.Circle((0.5, 0.5), 0.3, color="orange"))
ax.set_axis_off()
plt.show()
"""


def test_sandbox_renders_a_figure():
    png, err = cb.run_untrusted_python(KITTEN)
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert err is None


def test_sandbox_saves_a_figure_the_script_never_showed():
    png, _ = cb.run_untrusted_python(
        "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1, 2, 3])\n"
    )
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_sandbox_blocks_the_network():
    png, err = cb.run_untrusted_python(
        "import socket\nsocket.create_connection(('1.1.1.1', 80), 3)\n"
    )
    assert png is None
    assert "network disabled" in err


def test_sandbox_kills_a_script_that_never_ends(monkeypatch):
    monkeypatch.setattr(cb, "SBX_CPU_S", 3)
    monkeypatch.setattr(cb, "SBX_TIMEOUT_S", 15)
    png, err = cb.run_untrusted_python("while True:\n    pass\n")
    assert png is None
    assert err


def test_sandbox_hands_the_child_no_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_do_not_leak")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp_do_not_leak")
    png, err = cb.run_untrusted_python(
        "import os\n"
        "leak = [k for k in os.environ if 'TOKEN' in k or 'RUNPOD' in k]\n"
        "assert not leak, leak\n"
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\nax.plot([1])\n"
    )
    assert png, err


def test_sandbox_survives_a_script_that_crashes_after_drawing():
    png, err = cb.run_untrusted_python(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\nax.plot([1, 2])\n"
        "raise ValueError('boom')\n"
    )
    assert png
    assert "ValueError" in err


# ── The size cap, before any pod exists ─────────────────────────────────────

def test_size_check_passes_a_small_model(size_check):
    check, provider = size_check
    provider.gb = 16.0
    got = check("org/small", "", 64.0)
    assert got == {"ok": True, "gb": 16.0, "error": None}


def test_size_check_rejects_over_one_h100(size_check):
    check, provider = size_check
    provider.gb = 140.0
    got = check("org/huge", "", 64.0)
    assert got["ok"] is False
    assert "140 GB" in got["error"]
    assert "64 GB" in got["error"]


def test_size_check_rejects_an_unmeasurable_model(size_check):
    check, provider = size_check
    provider.gb = None
    got = check("org/gated", "", 64.0)
    assert got["ok"] is False
    assert "could not read the size" in got["error"]


# ── What a failed run actually says ─────────────────────────────────────────

class _Log:
    def __init__(self, text):
        self.text = text

    def get_progress(self, instance_id, lines=15, raw=False):
        return self.text


# The Ornith-9B run: the pod crashed loading the weights, and all the
# controller said was "run failed".
CRASH_TAIL = (
    "[progress] loading weights\n"
    "Traceback (most recent call last):\n"
    '  File "/root/catbench.py", line 403, in main\n'
    "    model, tok = load_model()\n"
    "ValueError: Unknown quantization type, got exl3 - supported types are: ['awq']\n"
    "[joberror] ValueError: Unknown quantization type, got exl3\n"
)


def test_a_crash_reports_the_exception_not_the_word_failed(job):
    got = job["failure_reason"](_Log(CRASH_TAIL), "pod-1", "failed")
    assert "Unknown quantization type" in got
    assert got != "failed"


def test_a_crash_with_no_traceback_falls_back_to_the_pods_own_error(job):
    got = job["failure_reason"](_Log("[progress] loading weights\n[joberror] the pod ran dry\n"),
                                "pod-1", "failed")
    assert got == "the pod ran dry"


def test_a_watchdog_outcome_names_the_watchdog(job):
    got = job["failure_reason"](_Log("[download] 41% complete\n"), "pod-1", "stalled")
    assert "stalled" in got
    assert "41%" in got


def test_an_unreadable_log_still_says_something(job):
    class _Dead:
        def get_progress(self, *a, **k):
            raise OSError("ssh gone")

    assert job["failure_reason"](_Dead(), "pod-1", "failed")
