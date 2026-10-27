"""Nested helpers in main() must import `cards` themselves.

main() imports cards late, inside a try near the end. That single statement
makes `cards` a LOCAL of main for the whole function, so any nested function
reading it gets an unassigned free variable and raises NameError when called --
which happens at _publish, i.e. after the model is downloaded, calibrated,
measured and quantized. The first SC run died there and its pod was reaped with
the finished quant on it.

quant.py cannot import cards at module scope: in the repo it is
blockquant.cards, on the pod it is flat beside this file.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


def _main_fn():
    for node in ast.walk(ast.parse(SRC.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    pytest.fail("no main() in quant.py")


def _imports_cards(fn) -> bool:
    return any(isinstance(n, ast.Import) and any(a.name == "cards" for a in n.names)
               for n in ast.walk(fn))


def _reads_cards(fn) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "cards" and isinstance(n.ctx, ast.Load)
               for n in ast.walk(fn))


def test_main_shadows_cards_so_this_check_is_needed():
    # If main() ever stops importing cards locally the hazard is gone and this
    # whole module can go; until then, it is real.
    assert _imports_cards(_main_fn()), "main() no longer imports cards; drop this test"


def test_every_nested_helper_that_uses_cards_imports_it():
    main = _main_fn()
    bad = []
    for node in main.body:
        for inner in ast.walk(node):
            if (isinstance(inner, ast.FunctionDef)
                    and _reads_cards(inner) and not _imports_cards(inner)):
                bad.append(inner.name)
    assert not bad, (
        f"{bad} read `cards` without importing it; main()'s late import makes it a "
        f"local, so these NameError at call time -- after the quant is built")


def test_no_nested_helper_relies_on_a_name_main_imports():
    """The same trap for every name, not just `cards`.

    Any `import X` in main()'s body makes X a local of main for the whole
    function. A nested helper reading X therefore gets a free variable that is
    unassigned until that line runs -- and these helpers are called from the
    middle of main, long before its tail. The rule is simply: if you use a name
    main imports, import it yourself.
    """
    main = _main_fn()
    imported = {a.asname or a.name.split(".")[0]
                for n in ast.walk(main) if isinstance(n, ast.Import) for a in n.names}
    bad = []
    for node in main.body:
        for fn in ast.walk(node):
            if not isinstance(fn, ast.FunctionDef):
                continue
            own = {a.asname or a.name.split(".")[0]
                   for n in ast.walk(fn) if isinstance(n, ast.Import) for a in n.names}
            used = {n.id for n in ast.walk(fn)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            for name in sorted(used & imported - own):
                bad.append(f"{fn.name} reads {name}")
    assert not bad, (
        "nested helpers read names main() imports, so they resolve to unassigned "
        f"locals at call time: {bad}")


def test_the_scoping_really_does_raise():
    """Proves the hazard rather than asserting it: same shape, run for real."""
    src = (
        "def main():\n"
        "    def publish():\n"
        "        return cards.name()\n"
        "    out = publish()\n"
        "    import cards\n"
        "    return out\n"
    )
    ns = {}
    exec(compile(src, "<shape>", "exec"), ns)
    with pytest.raises(NameError, match="cards"):
        ns["main"]()
