"""Every long phase the pod prints has to reach the controller.

get_progress greps the remote log down to _PROGRESS_MARKERS, and poll_remote
runs its stall clock off that filtered string: if a phase prints nothing
matching, the progress text is frozen for its whole duration, which is exactly
what a hung pod looks like. Self-calibration was four such stages -- long
enough on a big model to trip stall_timeout and terminate a working pod, and
in the meantime the job embed sat still.

So this is not a style check on log prefixes. A prefix is either a progress
marker or explicitly declared too chatty/too brief to be one.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from blockquant.providers.runpod.provider import RunPodProvider  # noqa: E402

QUANT = ROOT / "src/blockquant/remote/quant.py"

# Prefixes that deliberately do not drive progress. Startup lines that print
# once before the poll loop exists, or per-item chatter guarded by a phase
# marker that does report.
QUIET = {
    "gpu", "disk", "config", "tokenizer", "preprocess",  # setup, before polling
    "skip", "sample", "collection",                      # brief, inside a reported phase
}


def _printed_prefixes() -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(QUANT.read_text())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print"):
            continue
        if not node.args:
            continue
        a = node.args[0]
        first = a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else (
            a.values[0].value if isinstance(a, ast.JoinedStr) and a.values
            and isinstance(a.values[0], ast.Constant) else "")
        if (m := re.match(r"\[([a-z_]+)\]", str(first))):
            out.add(m.group(1))
    return out


def test_every_phase_the_pod_prints_is_a_marker_or_declared_quiet():
    markers = RunPodProvider._PROGRESS_MARKERS
    missing = sorted(p for p in _printed_prefixes()
                     if p not in QUIET and rf"\[{p}\]" not in markers)
    assert not missing, (
        f"quant.py prints {missing} and the controller filters it out. A phase with no "
        f"marker freezes the stall clock -- add it to _PROGRESS_MARKERS, or to QUIET here "
        f"if it is genuinely brief.")


def test_the_prefix_scan_actually_finds_things():
    # A regex that silently matched nothing would make the test above vacuous.
    found = _printed_prefixes()
    assert {"sc", "kl", "download", "upload"} <= found, found
