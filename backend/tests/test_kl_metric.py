"""What the KL number on a card is measured on.

The metric used to run on exllamav3's conversion/standard_cal_data -- the very
corpus EXL3 calibrates against. That is a train-set evaluation: it flatters
every quant we publish and cannot be read against turboderp's curves. These
tests pin the corpus apart from the calibration set and pin the method string
that travels with the number, because the failure being fixed was two
incompatible measurement series sharing one unlabelled field.

remote/quant.py imports the exllamav3 stack at module scope, so the helpers are
AST-loaded the way test_preprocessor_shim.py does it.
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "blockquant" / "remote" / "quant.py"
FETCH = Path(__file__).resolve().parents[2] / "docker" / "fetch_kl_corpus.py"
CARDS = Path(__file__).resolve().parents[1] / "src" / "blockquant" / "cards.py"

WANT = ("_eval_text", "_chat_format", "KL_CORPUS")


@pytest.fixture
def kl(tmp_path, monkeypatch):
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANT)
            or (isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in WANT for t in n.targets))]
    ns = {"json": json, "Path": Path, "__name__": "kl"}
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations", asname=None)], level=0)
    mod = ast.fix_missing_locations(ast.Module(body=[future] + body, type_ignores=[]))
    exec(compile(mod, str(SRC), "exec"), ns)
    return ns


def test_the_baked_corpus_is_used_when_present(kl, tmp_path):
    corpus = tmp_path / "kl_eval_corpus.utf8"
    corpus.write_text("held out text, not the calibration set", encoding="utf-8")
    kl["KL_CORPUS"] = corpus
    text, source = kl["_eval_text"]()
    assert "held out" in text
    assert source == "openwebtext"


@pytest.fixture
def fake_exllamav3(tmp_path, monkeypatch):
    """Stand in for the installed package so the fallback branch can run here.

    It reads standard_cal_data relative to exllamav3.__file__, which is the
    whole point: the fallback IS the calibration corpus.
    """
    import types
    pkg = tmp_path / "exllamav3"
    cal = pkg / "conversion" / "standard_cal_data"
    cal.mkdir(parents=True)
    (cal / "c4.utf8").write_text("calibration text", encoding="utf-8")
    mod = types.ModuleType("exllamav3")
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "exllamav3", mod)
    return mod


def test_a_missing_corpus_falls_back_but_says_the_number_is_tainted(
        kl, tmp_path, capsys, fake_exllamav3):
    """An image built before the corpus was baked must not report a number that
    looks like the corrected one."""
    kl["KL_CORPUS"] = tmp_path / "does-not-exist.utf8"
    text, source = kl["_eval_text"]()
    assert text == "calibration text"
    assert "calibration" in source.lower()
    assert "understates" in capsys.readouterr().out


def test_an_empty_corpus_is_not_silently_accepted(kl, tmp_path, fake_exllamav3):
    corpus = tmp_path / "kl_eval_corpus.utf8"
    corpus.write_text("   \n\n  ", encoding="utf-8")
    kl["KL_CORPUS"] = corpus
    _text, source = kl["_eval_text"]()
    assert "calibration" in source.lower()


def test_chat_formatting_is_best_effort_never_fatal(kl, tmp_path):
    """A model with no usable template still gets measured; it just says so."""
    text, label = kl["_chat_format"](tmp_path, "some prose")
    assert text == "some prose"
    assert label.startswith("raw")


# ── The corpus must not be the one the quantizer trains on ──────────────────

def test_the_eval_corpus_is_disjoint_from_the_calibration_set():
    """exllamav3 calibrates on c4/code/multilingual/technical/tiny/wiki.

    Whatever we measure on must not be any of those, which is the entire point
    of this change. Read the constant rather than trusting a comment.
    """
    src = FETCH.read_text()
    ns: dict = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in (
                "DATASET", "CONFIG", "SPLIT"):
            ns[node.targets[0].id] = ast.literal_eval(node.value)
    assert "openwebtext" in ns["DATASET"].lower()
    cal = ("c4", "code", "multilingual", "technical", "tiny", "wiki")
    assert not any(ns["DATASET"].lower().endswith(c) for c in cal)


# ── The card has to state the method, not describe it from memory ───────────

@pytest.fixture
def cards():
    spec = importlib.util.spec_from_file_location("cards", CARDS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cards"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(method):
    return [{"variant": "5.0", "head_bits": 8, "cal_rows": 250, "size_gb": 17.6,
             "url": "https://x", "kl_div": 0.014, "kl_method": method}]


def test_the_table_carries_the_kl_column_and_no_footnote(cards):
    """The card stays a table. The method is recorded in bq_quality.json, not
    explained under it."""
    table = cards.build_quants_table(_rows("openwebtext \u00b7 8\u00d78192 \u00b7 formatted"), "5.0")
    assert "0.0140" in table
    assert "<sub>" not in table
    assert "openwebtext" not in table
    # The old footnote claimed these were wikitext rows. They never were, and
    # now nothing on the card claims anything about the corpus.
    assert "wikitext" not in table.lower()


def test_an_old_row_with_no_method_still_renders(cards):
    """Quants published before the field existed must not break a re-render."""
    assert "0.0140" in cards.build_quants_table(_rows(None), "5.0")


def test_a_run_with_no_kl_at_all_drops_the_column(cards):
    rows = _rows(None)
    rows[0].pop("kl_div")
    table = cards.build_quants_table(rows, "5.0")
    assert "KL" not in table
