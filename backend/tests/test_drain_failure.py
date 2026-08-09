"""_last_exception, the thing that decides what a failed job says in Discord.

Loaded by AST rather than import: run_runpod_job.py pulls in the provider stack
at module scope, which needs the venv, and this function needs none of it.
"""
import ast
import re

import pytest

SRC = "scripts/run_runpod_job.py"


@pytest.fixture(scope="module")
def last_exception():
    tree = ast.parse(open(SRC).read())
    want = ("_FRAME", "_last_exception")
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in want)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in want)]
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), SRC, "exec"), ns)
    return ns["_last_exception"]


# The real thing, from the Qwopus3.6-27B run: three variants quantized, every
# upload failed, and the reported line stopped inside the request id.
HF_TAIL = [
    "[quantize] 6.0 complete",
    "[upload] 6.0 -> blockblockblock/Qwopus3.6-27B-Fusion-BF16-exl3-6.0bpw ...",
    "Traceback (most recent call last):",
    '  File "/opt/blockquant/quant.py", line 1102, in _publish',
    "    api.upload_folder(...)",
    "huggingface_hub.errors.HfHubHTTPError: (Request ID: Root=1-6a73d746-684494410ef5e519554d521f;15fe3927-5e8a-41b2)",
    "403 Forbidden: You have exceeded the storage quota for your account.",
]


def test_hf_error_keeps_the_message_and_drops_the_request_id(last_exception):
    got = last_exception(HF_TAIL)
    assert "403 Forbidden" in got
    assert "storage quota" in got
    assert "Request ID" not in got


def test_a_plain_exception_still_works(last_exception):
    got = last_exception(["  File \"x.py\", line 3",
                          "torch.cuda.OutOfMemoryError: CUDA out of memory."])
    assert got.startswith("torch.cuda.OutOfMemoryError")


def test_continuation_stops_at_the_next_log_line(last_exception):
    got = last_exception(["ValueError: bad", "why it was bad", "[upload] 5.0 -> repo ..."])
    assert got == "ValueError: bad why it was bad"


def test_no_exception_reports_nothing_rather_than_a_progress_line(last_exception):
    assert last_exception(["[download] 12%", "[disk] ok"]) == ""


def test_output_is_bounded(last_exception):
    got = last_exception(["RuntimeError: x"] + ["y" * 200] * 20)
    assert len(got) <= 600
