"""_publish, run for real against a written conversion.

This is the step that ate a pod: every SC stage ran, the conversion finished,
and then _publish raised NameError on `cards` and the controller reaped the
machine with the quant on it. Nothing offline covered it, so the only way to
find out was to rent a GPU for 45 minutes.

The helpers are lifted out of main() and executed with their closure supplied,
so this exercises the actual bytecode -- name resolution included -- rather
than asserting things about the source.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src/blockquant/remote/quant.py"
# The pod lays cards.py flat beside quant.py, which is why these import it by
# bare name. Mirror that here.
sys.path.insert(0, str(ROOT / "src/blockquant"))

WANT = ("_name_parts_for", "_publish")


def _lift(ns: dict):
    """Exec the nested helpers with `ns` as their globals."""
    found = {n.name: n for n in ast.walk(ast.parse(SRC.read_text()))
             if isinstance(n, ast.FunctionDef) and n.name in WANT}
    missing = set(WANT) - set(found)
    assert not missing, f"gone from quant.py: {missing}"
    for name in WANT:
        exec(compile(ast.Module(body=[found[name]], type_ignores=[]), str(SRC), "exec"), ns)
    return ns


class _Api:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, **kw):
        self.created.append(repo_id)

    def upload_file(self, **kw):
        self.uploaded.append(kw.get("repo_id"))


def _out_dir(tmp_path, **qc):
    d = tmp_path / "output-3.0bpw"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"quantization_config": qc}))
    return d


def _ns(api, *, sc, vision_bits=None, head_bits=6, hf_token="t"):
    ns = {
        "json": json, "Path": Path,
        "_dir_size_gb": lambda d: 0.5,
        "_written_quant_config": lambda d: json.loads(
            (Path(d) / "config.json").read_text()).get("quantization_config") or {},
        "_upload_folder_hb": lambda api, src, repo_id, variant: None,
        "shutil": type(sys)("shutil"),
        "head_bits": head_bits, "codebook": "mul1", "cal_rows": 250,
        "sc": sc, "vision_bits": vision_bits,
        "hf_token": hf_token, "owner": "Honkware", "model_name": "Qwen3.5-0.8B",
        "api": api, "repo_ids": [],
    }
    ns["shutil"].rmtree = lambda *a, **k: None
    return _lift(ns)


def test_an_sc_quant_publishes_under_its_self_calibrated_name(tmp_path):
    """The name the lost run was meant to produce."""
    api = _Api()
    ns = _ns(api, sc=True)
    out = _out_dir(tmp_path, bits=3.0, head_bits=6, vision_bits=6, codebook="mul1")
    rec = {"variant": "3.0"}
    ns["_publish"]("3.0", out, tmp_path / "work", rec)
    assert rec["hf_repo_id"] == "Honkware/Qwen3.5-0.8B-exl3-SC-3.0bpw-H6-V6"
    assert api.created == ["Honkware/Qwen3.5-0.8B-exl3-SC-3.0bpw-H6-V6"]
    assert rec["hf_url"].endswith("-SC-3.0bpw-H6-V6")


def test_an_sc_quant_with_a_copied_tower_carries_no_v(tmp_path):
    # No vision_bits in the written config = the tower was copied fp16.
    ns = _ns(_Api(), sc=True)
    out = _out_dir(tmp_path, bits=3.0, head_bits=6, codebook="mul1")
    rec = {"variant": "3.0"}
    ns["_publish"]("3.0", out, tmp_path / "work", rec)
    assert rec["hf_repo_id"] == "Honkware/Qwen3.5-0.8B-exl3-SC-3.0bpw-H6"


def test_a_plain_quant_states_nothing_it_did_not_choose(tmp_path):
    # Arch default tower: written as vision_bits 6, but unpinned, so unnamed.
    ns = _ns(_Api(), sc=False)
    out = _out_dir(tmp_path, bits=4.0, head_bits=6, vision_bits=6, codebook="mul1")
    rec = {"variant": "4.0"}
    ns["_publish"]("4.0", out, tmp_path / "work", rec)
    assert rec["hf_repo_id"] == "Honkware/Qwen3.5-0.8B-exl3-4.0bpw"


def test_a_plain_quant_names_a_tower_the_requester_pinned(tmp_path):
    ns = _ns(_Api(), sc=False, vision_bits=4)
    out = _out_dir(tmp_path, bits=4.0, head_bits=6, vision_bits=4, codebook="mul1")
    rec = {"variant": "4.0"}
    ns["_publish"]("4.0", out, tmp_path / "work", rec)
    assert rec["hf_repo_id"] == "Honkware/Qwen3.5-0.8B-exl3-4.0bpw-V4"


def test_without_a_token_it_records_the_readback_and_uploads_nothing(tmp_path):
    api = _Api()
    ns = _ns(api, sc=True, hf_token="")
    out = _out_dir(tmp_path, bits=3.0, head_bits=6, vision_bits=6, codebook="mul1")
    rec = {"variant": "3.0"}
    ns["_publish"]("3.0", out, tmp_path / "work", rec)
    assert api.created == []
    assert rec["_head_bits"] == 6 and rec["_vision_bits"] == 6
    assert rec["_name_parts"]["sc"] is True
