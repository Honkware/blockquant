"""The end of a run: the final log drain and the result fetch.

Structural, for the same reason as test_drain_failure: this all lives inside
main(), next to the provider stack.
"""
import ast

SRC = "scripts/run_runpod_job.py"


def _main():
    tree = ast.parse(open(SRC).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return fn, ast.get_source_segment(open(SRC).read(), fn)


def test_the_final_drain_skips_anything_already_printed():
    # It reads the raw tail while polls read a filtered one, so matching the
    # two by overlap found nothing and the 27B donor log got 670 lines twice.
    _, src = _main()
    drain = src[src.index("raw=True"):src.index("[6/6] Fetching result")]
    assert "not in printed" in drain
    assert "_unseen_lines(last_lines" not in drain
    assert "printed.update(new)" in src


def test_a_gone_pod_is_not_retried_over_ssh():
    _, src = _main()
    fetch = src[src.index("[6/6] Fetching result"):]
    assert fetch.index("pod_is_gone(") < fetch.index("get_result()")
    assert "range(0 if pod_gone else 5)" in fetch


def test_a_gone_pod_still_checks_hf_before_failing():
    # A clean run that tore its own pod down looks exactly like a dead one
    # from here; the variant on HF is what tells them apart.
    _, src = _main()
    fetch = src[src.index("[6/6] Fetching result"):]
    assert fetch.index("_variant_uploaded(") < fetch.index("the pod disappeared")


def test_printed_is_bound_in_main_not_in_the_closure():
    # Bound inside _print_new it would be a fresh set every poll, and the
    # drain would see nothing in it.
    fn, _ = _main()
    inner = next(n for n in ast.walk(fn)
                 if isinstance(n, ast.FunctionDef) and n.name == "_print_new")
    in_inner = {id(n) for n in ast.walk(inner)}
    binds = [n for n in ast.walk(fn) if isinstance(n, ast.AnnAssign)
             and getattr(n.target, "id", "") == "printed"]
    assert binds and all(id(n) not in in_inner for n in binds)
