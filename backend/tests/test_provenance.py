"""What a published quant says about itself, and which commit is recorded for it.

A registry pins weights as <repo>@<40-hex commit>. Every upload recorded
hf_revision = "main", which moves with each card re-render and each later
push, and nothing recorded which revision of the base model was quantized.
"""
import ast
import json
import sys
import time
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


def _lift(name, ns):
    for node in ast.parse(SRC.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC), "exec"), ns)
            return ns[name]
    pytest.fail(f"{name} is gone from quant.py")


SECRET = "hf_" + "x" * 34


def _prov(**over):
    f = _lift("_provenance", {"time": time})
    kw = dict(repo_id="o/M-exl3-4.0bpw", variant="4.0",
              rec={"_head_bits": 6, "_codebook": "mul1", "kl_div": 0.008, "kl_method": "qbench",
                   "hf_token": SECRET, "runpod_api_key": "rpa_secret"},
              name_parts={}, base_repo="b/M", base_revision="a" * 40, subfolder="",
              codebook="mul1", cal_rows=250, cal_cols=2048, donor_repo="",
              exllamav3={"version": "1.6.5", "commit": "c" * 40}, job_id="job-1")
    kw.update(over)
    return f(**kw)


def test_it_names_the_base_commit_and_the_exllamav3_that_built_it():
    p = _prov()
    assert p["base_model"] == {"repo_id": "b/M", "revision": "a" * 40, "subfolder": None}
    assert p["exllamav3"]["commit"] == "c" * 40
    assert p["quality"]["kl_median"] == 0.008
    assert p["job_id"] == "job-1"


def test_it_never_carries_a_secret_whatever_the_record_holds():
    text = json.dumps(_prov())
    assert SECRET not in text and "rpa_secret" not in text and "token" not in text


def test_it_says_how_it_was_calibrated():
    assert _prov()["quant"]["calibration"] == "exllamav3 bundled"
    sc = _prov(name_parts={"sc": True, "head_bits": 6}, donor_repo="o/M-exl3-6.0bpw")
    assert sc["quant"]["calibration"] == "self-sampled, measured"
    assert sc["quant"]["donor_repo"] == "o/M-exl3-6.0bpw"
    lite = _prov(name_parts={"sc_lite": True, "hq": 2})
    assert lite["quant"]["calibration"] == "self-sampled, trace only" and lite["quant"]["hq"] == 2


def test_the_exllamav3_commit_comes_from_pips_record(monkeypatch):
    import importlib.metadata as md
    dist = types.SimpleNamespace(version="1.6.5", read_text=lambda f: json.dumps(
        {"url": "https://github.com/turboderp-org/exllamav3", "vcs_info": {"vcs": "git", "commit_id": "d" * 40}}))
    monkeypatch.setattr(md, "distribution", lambda name: dist)
    f = _lift("_exllamav3_source", {"json": json, "Path": Path})
    assert f() == {"version": "1.6.5", "commit": "d" * 40}


def test_a_missing_exllamav3_does_not_fail_the_upload(monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "distribution", lambda name: (_ for _ in ()).throw(md.PackageNotFoundError()))
    assert _lift("_exllamav3_source", {"json": json, "Path": Path})() == {"version": None, "commit": None}


def test_the_commit_recorded_is_the_attempt_that_succeeded(monkeypatch, capsys):
    calls = []

    class Api:
        def upload_folder(self, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise ConnectionError("reset")
            return types.SimpleNamespace(oid="e" * 40)

    import threading
    ns = {"time": types.SimpleNamespace(sleep=lambda s: None), "_UPLOAD_TRIES": 3,
          "_UPLOAD_TERMINAL": {401, 403}, "threading": threading}
    f = _lift("_upload_folder_hb", ns)
    assert f(Api(), "/x", "o/M", "4.0").oid == "e" * 40
    assert len(calls) == 2


def test_the_pinned_head_is_read_after_the_card(capsys):
    api = types.SimpleNamespace(model_info=lambda r: types.SimpleNamespace(sha="f" * 40))
    outs = [{"variant": "4.0", "hf_repo_id": "o/M-exl3-4.0bpw"}, {"variant": "3.0"}]
    _lift("_pin_heads", {})(api, outs)
    assert outs[0]["hf_pinned"] == "f" * 40 and "hf_pinned" not in outs[1]
    assert f"[upload] 4.0 pinned o/M-exl3-4.0bpw@{'f' * 40}" in capsys.readouterr().out


def test_main_downloads_at_the_resolved_revision_and_records_no_branch():
    src = SRC.read_text()
    dl = src[src.index("def _do_download"):src.index("_dl_thread = threading.Thread")]
    assert "revision=base_revision" in dl
    assert 'rec["hf_revision"] = "main"' not in src
    assert "_pin_heads(api, outputs)" in src


def test_a_wheel_install_takes_its_commit_from_the_image(monkeypatch, tmp_path):
    # pip records no commit for a prebuilt wheel; the image writes the ref.
    import importlib.metadata as md
    dist = types.SimpleNamespace(version="1.6.5+cu128.torch2.8.0",
                                 read_text=lambda f: json.dumps({"url": "file:///tmp/x.whl"}))
    monkeypatch.setattr(md, "distribution", lambda name: dist)
    ref = tmp_path / "exllamav3_ref"
    ref.write_text("c" * 40 + "\n")
    real = Path
    fake = lambda p: ref if str(p) == "/opt/blockquant/exllamav3_ref" else real(p)
    f = _lift("_exllamav3_source", {"json": json, "Path": fake})
    assert f() == {"version": "1.6.5+cu128.torch2.8.0", "commit": "c" * 40}


def test_the_health_gate_reads_a_wheel_version():
    # "1.6.5+cu128.torch2.8.0".split(".") made the patch "5+cu128", which the
    # gate dropped, so 1.6.5 read as (1, 6) and failed a (1, 6, 5) minimum.
    src = (SRC.parent.parent / "providers/runpod/provider.py").read_text()
    assert "v.split('+')[0].split('.')" in src
    v = "1.6.5+cu128.torch2.8.0"
    p = tuple(int(x) for x in v.split("+")[0].split(".")[:3] if x.isdigit())
    assert p == (1, 6, 5)
