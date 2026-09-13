"""The launcher's arch gate has to be callable the way the launcher calls it.

_resolve_arch grew a subfolder argument in its body before it grew one in its
signature, so the gate raised TypeError at the call site -- on every job, before
any pod was rented. Nothing caught it because nothing here calls the launcher.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_runpod_job.py"


def _load(*names):
    """Exec just these top-level functions, the way test_pod_price_cap does."""
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names), f"missing one of {names}"
    mod = ast.Module(
        body=[ast.ImportFrom(module="pathlib", names=[ast.alias(name="Path")], level=0)] + body,
        type_ignores=[],
    )
    ns: dict = {"__file__": str(SRC)}
    exec(compile(ast.fix_missing_locations(mod), str(SRC), "exec"), ns)
    return ns


def test_resolve_arch_accepts_every_kwarg_the_launcher_passes():
    ns = _load("_resolve_arch", "_load_arch_support")
    params = inspect.signature(ns["_resolve_arch"]).parameters
    for kw in re.findall(r"_resolve_arch\([^)]*?(\w+)=", SRC.read_text()):
        assert kw in params, f"launcher passes {kw}= but _resolve_arch has no such parameter"


def test_resolve_arch_runs_without_unbound_names():
    """A repo that does not exist takes the except path, which still proves every
    name in the body is bound -- a NameError would surface as a crash, not a
    return value."""
    ns = _load("_resolve_arch", "_load_arch_support")
    assert ns["_resolve_arch"]("no/such-repo-xyz", "", subfolder="BF16") == ("", False, False)
