"""The launcher, the provider and the pod script have to agree.

Each of these is a keyword the previous layer spells out by hand, and a
mismatch is invisible until a pod is up: run_pipeline() gained --sc and
--donor-repo at the launcher and quant.py read them at the pod, but the
provider in between never grew the parameters, so every SC job rented a card,
bootstrapped, and died on a TypeError. The same shape ate a job once before
(_resolve_arch). Nothing here needs a GPU, so it can fail in CI instead.
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

QUANT = ROOT / "src/blockquant/remote/quant.py"
# Everywhere a provider is driven. pipeline.py is the API path and the launcher
# is the bot's; they drifted apart, and only one of them was ever run.
CALLERS = [ROOT / "scripts/run_runpod_job.py", ROOT / "src/blockquant/pipeline.py"]


def _call_kwargs(src: Path, func: str) -> set[str]:
    """Keywords passed at every call site of `func` in one file."""
    out: set[str] = set()
    found = False
    for node in ast.walk(ast.parse(src.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == func):
            found = True
            out |= {k.arg for k in node.keywords if k.arg}
    if not found:
        pytest.fail(f"no call to {func}() in {src.name}")
    return out


def _cfg_keys() -> set[str]:
    """Keys the provider puts in the pod's config JSON."""
    src = (ROOT / "src/blockquant/providers/runpod/provider.py").read_text()
    for node in ast.walk(ast.parse(src)):
        # `cfg: dict = {...}` is an AnnAssign, not an Assign.
        tgt = node.target if isinstance(node, ast.AnnAssign) else \
            (node.targets[0] if isinstance(node, ast.Assign) and node.targets else None)
        if (getattr(tgt, "id", "") == "cfg" and isinstance(getattr(node, "value", None), ast.Dict)):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    pytest.fail("no cfg dict in provider.py")


@pytest.mark.parametrize("caller", CALLERS, ids=lambda p: p.name)
def test_callers_pass_only_what_run_pipeline_takes(caller):
    from blockquant.providers.runpod.provider import RunPodProvider
    takes = set(inspect.signature(RunPodProvider.run_pipeline).parameters)
    extra = _call_kwargs(caller, "run_pipeline") - takes
    assert not extra, f"{caller.name} passes {sorted(extra)}, run_pipeline does not take it"


@pytest.mark.parametrize("key", ["sc", "donor_repo", "head_bits", "vision_bits",
                                 "subfolder", "gpu_count", "codebook"])
def test_what_the_pod_reads_is_what_the_provider_sends(key):
    # cfg.get("x") in quant.py is the pod's only view of the request.
    reads = {n.args[0].value
             for n in ast.walk(ast.parse(QUANT.read_text()))
             if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "get" and getattr(n.func.value, "id", "") == "cfg"
                 and n.args and isinstance(n.args[0], ast.Constant))}
    if key not in reads:
        pytest.skip(f"quant.py does not read {key}")
    assert key in _cfg_keys(), f"quant.py reads cfg[{key!r}], the provider never sends it"
