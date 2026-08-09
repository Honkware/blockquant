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
    import json
    import re
    from pathlib import Path
    ns = {"RunPodProvider": _P, "json": json, "Path": Path, "re": re,
          "_IMAGE_BY_NAME": {"exl3_043": "ghcr.io/x:043", "master": "ghcr.io/x:master",
                             "stable": ""},
          "_resolve_arch": lambda m, t: ("ArchForCausalLM", None, True)}
    _load(QUANT_JOB, ("_FRAME", "_last_exception"), ns)
    _load(JOB, ("size_check", "DEFAULT_MAX_GB", "format_check", "failure_reason"), ns)
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


# ── Which loader the pod reaches for ────────────────────────────────────────

def _config(tmp_path, monkeypatch, body):
    (tmp_path / "config.json").write_text(body, encoding="utf-8")
    monkeypatch.setattr(cb, "MODEL_DIR", tmp_path)


def test_an_exl3_repo_is_detected_from_its_config(tmp_path, monkeypatch):
    # Verbatim from AnuAmba/Ornith-9B-6bpw-exl3, the run that crashed.
    _config(tmp_path, monkeypatch,
            '{"architectures": ["Qwen3_5ForConditionalGeneration"], '
            '"quantization_config": {"quant_method": "exl3", "bits": 6.0}}')
    assert cb.quant_method() == "exl3"


def test_plain_weights_report_no_quant_method(tmp_path, monkeypatch):
    _config(tmp_path, monkeypatch, '{"architectures": ["Qwen3ForCausalLM"]}')
    assert cb.quant_method() == ""


def test_a_missing_config_is_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "MODEL_DIR", tmp_path)
    assert cb.quant_method() == ""


def test_a_format_with_no_loader_says_so_before_transformers_does(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "MODEL_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="awq"):
        cb.load_model("awq")


def test_reply_is_cut_at_the_turn_boundary():
    assert cb._trim("<svg/></svg><|im_end|>\n<|im_start|>user") == "<svg/></svg>"
    assert cb._trim("  plain  ") == "plain"


def test_a_base_model_gets_the_raw_prompt():
    class _NoTemplate:
        chat_template = None

    assert cb._chat_wrap(_NoTemplate(), "hi") == ("hi", False)
    assert cb._chat_wrap(None, "hi") == ("hi", False)


def test_an_instruct_model_gets_its_template():
    class _Tok:
        chat_template = "x"

        @staticmethod
        def apply_chat_template(msgs, **kw):
            return f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>"

    text, special = cb._chat_wrap(_Tok(), "hi")
    assert special is True
    assert "hi" in text


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


# ── The format cap, also before any pod exists ──────────────────────────────

def _fmt(job, fmt, entry=None, arch="Qwen3ForCausalLM"):
    job["repo_format"] = lambda m, t: fmt
    job["_resolve_arch"] = lambda m, t: (arch, entry, True)
    return job["format_check"]("org/m", "")


def test_plain_weights_pass_and_pick_no_image(job):
    assert _fmt(job, "") == {"ok": True, "format": "", "image": "", "error": None}


def test_exl3_passes_on_a_stable_arch(job):
    got = _fmt(job, "exl3", {"tier": "stable", "image": "stable"})
    assert got["ok"] is True
    assert got["image"] == ""


def test_exl3_on_a_special_arch_forces_that_image(job):
    got = _fmt(job, "exl3", {"tier": "special", "image": "exl3_043"},
               arch="Qwen3_5ForConditionalGeneration")
    assert got["ok"] is True
    assert got["image"] == "ghcr.io/x:043"


def test_exl3_of_an_arch_exllamav3_never_heard_of_is_rejected(job):
    got = _fmt(job, "exl3", None, arch="MadeUpForCausalLM")
    assert got["ok"] is False
    assert "MadeUpForCausalLM" in got["error"]


@pytest.mark.parametrize("fmt", ["gguf", "awq", "gptq", "compressed-tensors"])
def test_a_format_with_no_loader_is_rejected_by_name(job, fmt):
    got = _fmt(job, fmt)
    assert got["ok"] is False
    assert fmt.upper() in got["error"]


# ── What a failed run actually says ─────────────────────────────────────────

class _Log:
    def __init__(self, text):
        self.text = text

    def get_progress(self, instance_id, lines=15, raw=False):
        return self.text


# The Ornith-9B run: transformers refusing an EXL3 quant, reported as "failed".
EXL3_TAIL = (
    "[progress] loading weights (exl3)\n"
    "Traceback (most recent call last):\n"
    '  File "/root/catbench.py", line 403, in main\n'
    "    runner = load_model(fmt)\n"
    "ValueError: Unknown quantization type, got exl3 - supported types are: ['awq']\n"
    "[joberror] ValueError: Unknown quantization type, got exl3\n"
)


def test_a_crash_reports_the_exception_not_the_word_failed(job):
    got = job["failure_reason"](_Log(EXL3_TAIL), "pod-1", "failed")
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
