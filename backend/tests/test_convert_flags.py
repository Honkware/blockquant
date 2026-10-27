"""-mb / -ngb / -hq and the trace-only SCT kind: what they are called, and that
the name never claims a setting that did nothing.

turboderp's own advice in the Discord was to pass -mb, -vb and -ngb, and for a
big model to calibrate on sc_trace with -hq 2 rather than measure it. All four
change the weights, so each is a name component -- but only when pinned, the
same rule the vision tower follows.
"""
import ast
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "blockquant"))
from blockquant import cards  # noqa: E402

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"
LAUNCHER = Path(__file__).resolve().parent.parent / "scripts/run_runpod_job.py"


def _lift(src, name, ns):
    for node in ast.walk(ast.parse(src.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(src), "exec"), ns)
            return ns[name]
    pytest.fail(f"{name} is gone from {src.name}")


@pytest.mark.parametrize("kw,slug", [
    ({}, "M-exl3-4.0bpw"),
    ({"sc_lite": True, "head_bits": 6, "hq": 2}, "M-exl3-SCT-4.0bpw-H6-HQ2"),
    ({"sc": True, "head_bits": 5, "vision_bits": 6, "mtp_bits": 3.0},
     "M-exl3-SC-4.0bpw-H5-V6-M3"),
    ({"mtp_bits": 2.5, "ngram_bits": 6}, "M-exl3-4.0bpw-M2.5-NG6"),
])
def test_names_round_trip(kw, slug):
    assert cards.exl3_repo_slug("M", "4.0", **kw) == slug
    back = cards.parse_exl3_slug(slug, "M")
    assert back is not None, f"{slug} built but does not parse"
    rebuilt = {k: v for k, v in back.items() if k not in ("slug", "variant") and v}
    assert cards.exl3_repo_slug("M", back["variant"], **rebuilt) == slug


def test_sct_is_not_sc():
    # Same bitrate, different weights: one must not overwrite the other.
    a = cards.exl3_repo_slug("M", "3.0", sc=True, head_bits=6)
    b = cards.exl3_repo_slug("M", "3.0", sc_lite=True, head_bits=6)
    assert a != b
    assert cards.parse_exl3_slug(b, "M")["sc"] is False
    assert cards.parse_exl3_slug(b, "M")["sc_lite"] is True


def test_sct_must_name_its_head_bits():
    with pytest.raises(ValueError):
        cards.exl3_repo_slug("M", "3.0", sc_lite=True)


def _name_parts_for(*, sc=False, sc_lite=False, pinned=None, has_ngram=True):
    pinned = pinned or {}
    return _lift(SRC, "_name_parts_for", {
        "cards": cards, "sc": sc, "sc_lite": sc_lite, "head_bits": 6,
        "vision_bits": None, "_pinned": pinned,
        "mtp_bits": pinned.get("mtp_bits"), "ngram_bits": pinned.get("ngram_bits"),
        "model_dir": Path("/nonexistent"),
        "_model_has_tensor_substring": lambda d, part: has_ngram,
    })


def test_a_pinned_mtp_head_is_named_from_what_was_written():
    f = _name_parts_for(pinned={"mtp_bits": 3.0})
    assert f({"_mtp_bits": 3.0}) == {"mtp_bits": 3.0}


def test_mtp_bits_on_a_model_with_no_head_names_nothing():
    # convert writes mtp_bits only when there was a head to quantize.
    f = _name_parts_for(pinned={"mtp_bits": 3.0})
    assert "mtp_bits" not in f({"_mtp_bits": None})


def test_ngram_bits_without_ngram_tables_names_nothing():
    f = _name_parts_for(pinned={"ngram_bits": 6}, has_ngram=False)
    assert "ngram_bits" not in f({})
    f = _name_parts_for(pinned={"ngram_bits": 6}, has_ngram=True)
    assert f({})["ngram_bits"] == 6


def test_sct_names_itself_and_its_hq():
    f = _name_parts_for(sc=True, sc_lite=True, pinned={"hq": 2})
    got = f({"_head_bits": 6, "_vision_bits": None})
    assert got.get("sc_lite") is True and "sc" not in got
    assert got["hq"] == 2


def test_an_sc_card_says_what_it_was_calibrated_on():
    # Every SC card claimed exllamav3's bundled mix, off the model config.
    bundled = "exllamav3 bundled mix"
    assert cards._cal_source(bundled, sc=False, sc_lite=False, donor_repo=None, hq=None) == bundled
    sc = cards._cal_source(bundled, sc=True, sc_lite=False, donor_repo="o/d-exl3-6.0bpw", hq=None)
    assert "bundled" not in sc and "o/d-exl3-6.0bpw" in sc and "sc_measure" in sc
    sct = cards._cal_source(bundled, sc=False, sc_lite=True, donor_repo=None, hq=2)
    assert "no sensitivity measurement" in sct and "-hq 2" in sct


def test_the_card_names_pinned_settings():
    card = cards.render_exl3_card(
        base_repo="o/M", repo_id="x/M-exl3-4.0bpw-M3-HQ2", variant="4.0", head_bits=6,
        cal_rows=250, size_gb=1.0, model_config={}, quant_rows=[], collection_url="",
        quantized_by="x", mtp_bits=3.0, hq=2)
    assert "| MTP head | `3` bits |" in card
    assert "`-hq 2`" in card
    assert "{{" not in card


def _convert_block():
    src = SRC.read_text()
    return src[src.index('"--parallel_mode",'):src.index("sys.argv = argv")]


def test_convert_gets_the_new_flags():
    blk = _convert_block()
    for flag in ('"-mb"', '"-ngb"', '"-hq"'):
        assert flag in blk, f"{flag} never reaches convert"


def test_sct_converts_on_the_trace_without_a_recipe():
    blk = _convert_block()
    i = blk.index("elif sc_cal is not None:")
    branch = blk[i:blk.index("if gpu_count > 1", i)]
    assert '"-cd"' in branch and '"-rcp"' not in branch


def test_sct_stops_after_the_trace():
    src = SRC.read_text()
    fn = src[src.index("def _run_sc_stages"):src.index("def _selfcal_dir")]
    assert fn.index("if trace_only:") < fn.index('stage("sc_rfn_probe"')


class _Info:
    def __init__(self, mod):
        self.siblings = [types.SimpleNamespace(rfilename="config.json")]
        self.last_modified = mod


def _variant_uploaded(repos):
    import huggingface_hub
    seen = []

    class Api:
        def __init__(self, token=None):
            pass

        def whoami(self):
            return {"name": "Honkware"}

        def model_info(self, repo_id):
            seen.append(repo_id)
            if repo_id not in repos:
                raise FileNotFoundError(repo_id)
            return _Info(repos[repo_id])

    orig = huggingface_hub.HfApi
    huggingface_hub.HfApi = Api
    fn = _lift(LAUNCHER, "_variant_uploaded", {})
    return fn, seen, lambda: setattr(huggingface_hub, "HfApi", orig)


def test_a_dead_sc_pod_is_not_judged_on_its_own_donor():
    # The plain 4.0 is the donor an SC 4.0 builds an hour earlier.
    now = datetime.now(timezone.utc)
    fn, seen, undo = _variant_uploaded({"Honkware/M-exl3-4.0bpw": now})
    try:
        assert not fn("o/M", ["4.0"], "", "t",
                      name_parts={"sc": True, "head_bits": 6}, since=0)
        assert seen == ["Honkware/M-exl3-SC-4.0bpw-H6"]
    finally:
        undo()


def test_a_repo_older_than_this_launch_does_not_count():
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fn, _, undo = _variant_uploaded({"Honkware/M-exl3-4.0bpw": old})
    try:
        assert not fn("o/M", ["4.0"], "", "t", since=datetime(2026, 6, 1).timestamp())
        assert fn("o/M", ["4.0"], "", "t", since=old.timestamp() - 1)
    finally:
        undo()


def test_the_launcher_names_jobs_the_way_the_pod_does():
    fn = _lift(LAUNCHER, "_name_parts_from_args", {})
    a = types.SimpleNamespace(sc=True, sc_lite=True, hq=None, head_bits=6,
                              vision_bits=16, mtp_bits=None, ngram_bits=None)
    assert cards.exl3_repo_slug("M", "3.0", **fn(a)) == "M-exl3-SCT-3.0bpw-H6-HQ2"
    a = types.SimpleNamespace(sc=False, sc_lite=False, hq=None, head_bits=None,
                              vision_bits=6, mtp_bits=3.0, ngram_bits=None)
    assert cards.exl3_repo_slug("M", "4.0", **fn(a)) == "M-exl3-4.0bpw-V6-M3"


def _kl_rows(method="qbench · self-sampled trace · 40 rows"):
    return [dict(variant=v, kl_div=k, kl_method=method, repo_id=f"o/m-{v}-{sc}", sc=sc)
            for v, k, sc in [("2.0", 0.08, False), ("4.0", 0.011, False), ("3.0", 0.018, True)]]


def test_kl_chart_is_drawn_for_a_family_scored_on_one_corpus():
    svg = cards.build_kl_svg(_kl_rows(), "o/m-4.0-False")
    assert svg.startswith("<svg") and "self-sampled" in svg
    assert svg.count("<circle") >= 4      # three points, a ring, the legend


def test_kl_chart_refuses_to_mix_corpora_or_draw_one_point():
    rows = _kl_rows()
    rows[0]["kl_method"] = "qbench · wiki2 · 40×2048"
    assert cards.build_kl_svg(rows) is None
    assert cards.build_kl_svg(_kl_rows()[:1]) is None


def test_the_card_links_the_chart_only_when_there_is_one():
    kw = dict(base_repo="o/M", repo_id="o/m-4.0-False", variant="4.0", head_bits=6,
              cal_rows=250, size_gb=1.0, model_config={}, collection_url="",
              quantized_by="x")
    assert "(kl.svg)" in cards.render_exl3_card(quant_rows=_kl_rows(), **kw)
    assert "kl.svg" not in cards.render_exl3_card(quant_rows=_kl_rows()[:1], **kw)


def test_every_card_upload_carries_its_chart():
    # A card that links kl.svg without the file shows a broken image.
    for src in (SRC, Path(__file__).resolve().parent.parent / "scripts/publish_quant.py"):
        text = src.read_text()
        assert text.count('path_in_repo="kl.svg"') == text.count("build_kl_svg("), src.name


def test_the_card_links_the_drafter_published_beside_it():
    card = cards.render_exl3_card(
        base_repo="o/M", repo_id="x/M-exl3-4.0bpw", variant="4.0", head_bits=6, cal_rows=250,
        size_gb=1.0, model_config={}, quant_rows=[], collection_url="", quantized_by="x",
        drafter_repo="x/M-DFlash2-exl3-4.0bpw")
    assert "| Drafter | [`x/M-DFlash2-exl3-4.0bpw`](https://huggingface.co/x/M-DFlash2-exl3-4.0bpw) |" in card


def test_a_drafter_failure_cannot_fail_the_job():
    # Best-effort by construction: every step inside one try that returns None.
    src = SRC.read_text()
    fn = src[src.index("def _quant_drafter"):src.index("def _selfcal_dir")]
    body = fn[fn.index("    try:"):]
    assert "except Exception as e:" in body and "return None" in body
    assert 'print(f"[drafter] done -> ' in fn, "the bot finds the drafter by this line"
