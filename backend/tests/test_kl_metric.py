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
CARDS = Path(__file__).resolve().parents[1] / "src" / "blockquant" / "cards.py"

SELFCAL = Path(__file__).resolve().parents[1] / "src" / "blockquant" / "selfcal"


def _kl_source() -> str:
    return SRC.read_text()


def test_the_eval_asks_qbench_for_wiki2_at_its_own_geometry():
    """wiki2 at 10x2048 is what qbench's example project ships and the only
    corpus sc_measure implements. We measured openwebtext at 8x8192 before --
    a supported source, but nobody else's geometry, so the number could not be
    read against anyone's."""
    src = _kl_source()
    assert '"source": "wiki2"' in src
    assert "rows: int = 10" in src
    assert "seq_len: int = 2048" in src


def test_the_eval_split_is_not_the_calibration_set():
    """EXL3 calibrates on standard_cal_data/*.utf8, wiki.utf8 among them, so
    measuring on the same text would be measuring on the training set.

    The vendored dataset spec has to name the TEST split; the calibration wiki
    is a separate general Wikipedia dump. Read the spec rather than trusting
    the comment above it.
    """
    spec = (SELFCAL / "eval" / "qbench" / "data.py").read_text()
    block = spec[spec.index('"wiki2"'):spec.index('"wikitext2"')]
    assert '"wikitext-2-raw-v1"' in block
    assert '"split": "test"' in block


def test_the_kl_kernel_is_probed_out_of_process():
    """compute_kl_div segfaulted on builds before 1.5.0, and a segfault kills
    the pod rather than raising. The probe has to be a subprocess or it cannot
    help."""
    src = _kl_source()
    probe = src[src.index("def _kl_kernel_usable"):src.index("def _kl_div_eval")]
    assert "subprocess.run" in probe
    assert "returncode" in probe


def test_the_card_quotes_the_median_not_the_mean():
    """turboderp's own note: the mean is dominated by tokens the reference is
    undecided on. The median is what isolates quantization damage."""
    src = _kl_source()
    assert 'kl = stats["kld_median"]' in src
    assert 'rec["kl_stats"] = stats' in src


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
