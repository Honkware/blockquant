"""The self-calibration cache, and above all when it must refuse.

trace/cal/rfn/measure depend on the model and the donor, not the bitrate, so a
later SC run of the same model can skip the two long stages -- 25.5 and 16.3
minutes, measured. The failure that matters is not a miss, it is a hit under
the wrong key: reusing another model's measurements produces a recipe that is
wrong in a way nothing downstream can detect. So the match is exact and
anything unrecognised regenerates.
"""
import ast
import json
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"
WANT = ("_sc_cache_repo", "_sc_cache_key", "_sc_cache_restore", "_sc_cache_save")


@pytest.fixture
def mod(monkeypatch):
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANT)
            or (isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "_SC_CACHE_FILES")]
    assert len(body) == len(WANT) + 1, "cache helpers missing from quant.py"
    ns = {"json": json, "Path": Path, "shutil": __import__("shutil"),
          "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


class _Api:
    token = "t"

    def __init__(self):
        self.uploaded = []
        self.created = []

    def create_repo(self, repo_id, **kw):
        self.created.append((repo_id, kw.get("repo_type")))

    def upload_file(self, **kw):
        self.uploaded.append(kw.get("path_in_repo"))


def _store(tmp_path, key, files=None):
    """A fake hub: manifest plus artifacts, served by a stub hf_hub_download."""
    d = tmp_path / "remote"
    d.mkdir(exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(key))
    for name in (files if files is not None else ["trace.json", "cal.safetensors",
                                                  "rfn.json", "measure.json"]):
        (d / name).write_text(f"contents of {name}")

    def download(repo_id, filename, **kw):
        f = d / filename
        if not f.exists():
            raise FileNotFoundError(filename)
        return str(f)
    return download


def _patch_download(mod, monkeypatch, fn):
    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = fn
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


KEY = {"model_id": "Qwen/Qwen3.5-0.8B", "model_rev": "", "donor_repo": "o/d-exl3-4.0bpw",
       "cal_rows": 250, "cal_cols": 2048, "exllamav3": "1.5.0", "schema": 1}


def test_the_repo_name_is_predictable_and_not_tied_to_a_bitrate(mod):
    # A pod has to be able to ask for it before spending anything, and deleting
    # one quant must not take the cache with it.
    name = mod["_sc_cache_repo"]("Honkware", "Qwen3.5-0.8B")
    assert name == "Honkware/Qwen3.5-0.8B-exl3-selfcal"
    assert "bpw" not in name


def test_the_key_covers_what_the_artifacts_depend_on(mod):
    k = mod["_sc_cache_key"]("m", "rev", "o/d", 250, 2048)
    for field in ("model_id", "model_rev", "donor_repo", "cal_rows", "cal_cols", "exllamav3"):
        assert field in k, field
    # Not the bitrate: that is the whole point of sharing them.
    assert not any("bpw" in str(v) or k_ == "bits" for k_, v in k.items())


def test_a_matching_cache_is_restored(mod, monkeypatch, tmp_path):
    _patch_download(mod, monkeypatch, _store(tmp_path, KEY))
    work = tmp_path / "work"
    assert mod["_sc_cache_restore"](_Api(), "o/r", dict(KEY), work) is True
    assert (work / "measure.json").read_text() == "contents of measure.json"


@pytest.mark.parametrize("field,value", [
    ("model_id", "someone/else"),
    ("donor_repo", "o/other-exl3-6.0bpw"),
    ("cal_rows", 100),
    ("cal_cols", 1024),
    ("exllamav3", "1.4.9"),
])
def test_a_cache_under_a_different_key_is_refused(mod, monkeypatch, tmp_path, field, value):
    # THE case. A hit here optimizes the quant against the wrong measurements
    # and nothing downstream can tell.
    stored = {**KEY, field: value}
    _patch_download(mod, monkeypatch, _store(tmp_path, stored))
    work = tmp_path / "work"
    assert mod["_sc_cache_restore"](_Api(), "o/r", dict(KEY), work) is False
    assert not (work / "measure.json").exists()


def test_no_manifest_is_a_miss_not_a_crash(mod, monkeypatch, tmp_path):
    def download(**kw):
        raise FileNotFoundError("manifest.json")
    _patch_download(mod, monkeypatch, download)
    assert mod["_sc_cache_restore"](_Api(), "o/r", dict(KEY), tmp_path / "w") is False


def test_a_partial_cache_reports_incomplete(mod, monkeypatch, tmp_path):
    # Whatever is missing just gets rebuilt, but it must not claim a full hit.
    _patch_download(mod, monkeypatch, _store(tmp_path, KEY, files=["trace.json", "rfn.json"]))
    work = tmp_path / "work"
    assert mod["_sc_cache_restore"](_Api(), "o/r", dict(KEY), work) is False
    assert (work / "trace.json").exists()


def test_saving_puts_the_manifest_last(mod, tmp_path):
    # It is what a later run trusts, so it must not land before the files it
    # vouches for -- otherwise a crash mid-upload leaves a manifest promising
    # artifacts that are not there.
    work = tmp_path / "w"
    work.mkdir()
    for name in ("trace.json", "cal.safetensors", "rfn.json", "measure.json"):
        (work / name).write_text("x")
    api = _Api()
    mod["_sc_cache_save"](api, "o/r", dict(KEY), work)
    assert api.uploaded[-1] == "manifest.json"
    assert api.created == [("o/r", "dataset")]


def test_saving_skips_empty_artifacts(mod, tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "trace.json").write_text("x")
    (work / "measure.json").write_text("")     # a stage that died mid-write
    api = _Api()
    mod["_sc_cache_save"](api, "o/r", dict(KEY), work)
    assert "trace.json" in api.uploaded
    assert "measure.json" not in api.uploaded


def test_a_save_failure_is_not_fatal(mod, tmp_path):
    # The quant is the deliverable; the cache is an optimisation.
    class _Broken(_Api):
        def upload_file(self, **kw):
            raise RuntimeError("hub down")
    work = tmp_path / "w"
    work.mkdir()
    (work / "trace.json").write_text("x")
    mod["_sc_cache_save"](_Broken(), "o/r", dict(KEY), work)   # must not raise


def _main_src() -> str:
    return SRC.read_text()


def test_the_cache_is_consulted_before_the_donor_is_fetched():
    """Order is the whole saving here.

    A complete cache skips sc_trace and sc_rfn_probe, and those are the only
    two stages that read the donor -- so fetching it first means pulling ~20 GB
    on a 27B for a file nothing opens, on exactly the repeat runs the cache
    exists to make cheap.
    """
    src = _main_src()
    restore = src.index("_sc_cache_restore(api, sc_cache_repo")
    fetch = src.index("snapshot_download(repo_id=donor_repo")
    assert restore < fetch, "the donor is fetched before the cache is checked"


def test_the_donor_fetch_is_conditional_on_a_cache_miss():
    src = _main_src()
    i = src.index("if sc and not sc_cached:")
    j = src.index("snapshot_download(repo_id=donor_repo", i)
    # Nothing but the guard's own body between them.
    assert "def " not in src[i:j], "the donor fetch drifted out of the miss branch"


def test_a_cached_run_still_has_a_donor_path_for_the_skipped_stages():
    # _run_sc_stages builds its argv eagerly, so donor_dir must stay a path
    # even when nothing downloads to it -- the stages that name it are skipped.
    src = _main_src()
    assert 'donor_dir = workspace / "donor" if sc else None' in src
