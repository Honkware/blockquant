"""--max-price is a ceiling on the POD, not on one card.

Loaded by AST rather than import, the way test_drain_failure does it:
run_runpod_job.py pulls in the provider stack at module scope and neither of
these two functions needs any of it.
"""
import ast

import pytest

SRC = "scripts/run_runpod_job.py"


@pytest.fixture(scope="module")
def cap():
    want = ("_recommend_max_price", "_pod_price_cap")
    body = [n for n in ast.parse(open(SRC).read()).body
            if isinstance(n, ast.FunctionDef) and n.name in want]
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations", asname=None)], level=0)
    mod = ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[]))
    ns = {}
    exec(compile(mod, SRC, "exec"), ns)
    return ns


def test_auto_cap_unchanged_for_a_single_gpu(cap):
    """One card is what every job asks for today; the tiers must still be the
    plain per-card numbers there."""
    for base_gb in (None, 17.0, 40.0, 72.0, 200.0):
        assert cap["_pod_price_cap"]("auto", base_gb, 1) == cap["_recommend_max_price"](base_gb)


def test_auto_cap_scales_with_the_card_count(cap):
    """A 4-GPU pod bills 4x, so a cap left at the one-GPU figure would reject
    every candidate and spend the whole launch sweep finding that out."""
    assert cap["_pod_price_cap"]("auto", 72.0, 4) == pytest.approx(4 * 1.80)


def test_a_pinned_cap_already_names_the_pod(cap):
    """A number the caller typed is the hourly burn they will accept, whole
    pod. Multiplying it by the count would silently raise their own ceiling."""
    assert cap["_pod_price_cap"]("2.50", 72.0, 8) == pytest.approx(2.50)
    assert cap["_pod_price_cap"](0, 72.0, 8) == 0.0
