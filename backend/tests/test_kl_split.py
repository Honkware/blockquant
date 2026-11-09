"""KL on a second, short pod, so the KL floor stops holding the whole job.

The KL eval loads the fp16, so a plain job only went to cards that could hold
it. NorthCUA-Mini, a 70 GB MoE, paid an RTX PRO 6000 at $2.59/h for 6925 s for
an eval that took minutes; a 4090 converts it for well under half that. The
launcher now converts on the cheapest card with KL off and rents a big card
for the eval alone, when that saves enough to bother.
"""
import ast
import importlib.util
import json
import sys
import traceback
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
QUANT = ROOT / "src/blockquant/remote/quant.py"
TABLE = json.loads((ROOT / "gpu_table.json").read_text())

# RunPod's listing, trimmed to the cards these jobs choose between.
GPUS = [{"id": f"NVIDIA {n}", "memoryInGb": m} for n, m in (
    ("GeForce RTX 3090", 24), ("GeForce RTX 4090", 24), ("GeForce RTX 5090", 32),
    ("A40", 48), ("A100-SXM4-80GB", 80), ("H100 NVL", 94),
    ("RTX PRO 6000 Blackwell Server Edition", 96))]


@pytest.fixture(scope="module")
def rj():
    sys.argv = ["run_runpod_job.py"]
    spec = importlib.util.spec_from_file_location("rj", ROOT / "scripts/run_runpod_job.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _saving(rj, gb, moe):
    return rj._kl_split_saving(GPUS, gb, rj._kl_min_vram(gb, 24), 24, moe)


def test_the_big_moe_converts_on_the_cheapest_card(rj):
    assert rj._kl_min_vram(70.2, 24) == 81
    assert rj._rank_cards(GPUS, 81)[0] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert rj._rank_cards(GPUS, 24)[0] == "NVIDIA GeForce RTX 4090"
    assert _saving(rj, 70.2, moe=True) > 1.5


@pytest.mark.parametrize("gb", [55.6, 66.0])
def test_a_dense_27b_or_33b_keeps_one_pod(rj, gb):
    # A second boot and download eats most of what the cheaper card saves.
    assert _saving(rj, gb, moe=False) < TABLE["kl_split_min_saving"]


def test_nothing_to_split_when_the_cheapest_card_holds_the_fp16(rj):
    assert rj._kl_min_vram(16.4, 24) == 24
    assert _saving(rj, 16.4, moe=False) == 0.0


def test_moe_is_read_off_the_config(rj, monkeypatch, tmp_path):
    import huggingface_hub

    def config(cfg):
        def fake(*a, **k):
            if cfg is None:
                raise OSError("gated")
            p = tmp_path / "config.json"
            p.write_text(json.dumps(cfg))
            return str(p)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)

    config({"num_experts": 256})
    assert rj._is_moe("m", "") is True and rj._default_codebook("m", "") == "mcg"
    config({"text_config": {"num_experts": 256}})
    assert rj._is_moe("m", "") is True
    config({"hidden_size": 5120})
    assert rj._is_moe("m", "") is False and rj._default_codebook("m", "") == "mul1"
    # Unreadable: no split, and the codebook keeps its old answer.
    config(None)
    assert rj._is_moe("m", "") is None and rj._default_codebook("m", "") == "mcg"


def test_the_kl_pod_gets_its_keys_from_the_environment(rj, monkeypatch, capsys):
    seen = {}

    class Popen:
        def __init__(self, cmd, env):
            seen.update(cmd=cmd, env=env)

        def wait(self):
            return 1

        def terminate(self):
            pass

    monkeypatch.setattr(rj.subprocess, "Popen", Popen)
    args = types.SimpleNamespace(model="b/M", hf_org="o", subfolder="", job_id="j1", image="img",
                                 head_bits=None, cal_rows=None, codebook="mcg",
                                 hf_token="hf_secret", runpod_api_key="rpa_secret")
    result = {"outputs": [{"variant": "4.0", "hf_repo_id": "o/M-exl3-4.0bpw"}],
              "drafter_repo": "o/D-exl3-4.0bpw"}
    assert rj._run_kl_pod(args, result, 81) == 1
    cmd = seen["cmd"]
    assert cmd[cmd.index("--kl-only") + 1] == "o/M-exl3-4.0bpw"
    assert cmd[cmd.index("--variants") + 1] == "4.0"
    assert cmd[cmd.index("--min-vram") + 1] == "81"
    assert cmd[cmd.index("--drafter-repo") + 1] == "o/D-exl3-4.0bpw"
    assert "--head-bits" not in cmd
    assert not any("secret" in c for c in cmd)
    assert seen["env"]["HF_TOKEN"] == "hf_secret"
    assert seen["env"]["RUNPOD_API_KEY"] == "rpa_secret"
    # A failed KL pod costs the number, never the job.
    assert "published without a KL number" in capsys.readouterr().out


def test_the_launcher_turns_kl_off_only_for_a_split_job():
    src = (ROOT / "scripts/run_runpod_job.py").read_text()
    assert "kl_eval=False if split_kl else None" in src
    main = src[src.index("def main():"):src.index("def _run_kl_pod(")]
    # After the finally, so the conversion pod is already terminated.
    assert main.rindex("_run_kl_pod(args, result, split_kl)") > main.rindex("finally:")
    # A stock-out that lands the conversion on a big card keeps KL there.
    assert main.index("holds the fp16; measuring KL on this pod") < main.index("provider.run_pipeline(")


# --- the pod side -----------------------------------------------------------

def _lift(name, ns):
    for node in ast.parse(QUANT.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(QUANT), "exec"), ns)
            return ns[name]
    pytest.fail(f"{name} is gone from quant.py")


PROV = {"schema": 1, "job_id": "j1", "quant": {"repo_id": "o/M-exl3-4.0bpw", "bits": 4.0,
        "head_bits": 6, "mtp_bits": None, "ngram_bits": None, "hq": None},
        "quality": None, "created_at": "2026-11-06T12:14:00Z"}


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """_kl_only with Hugging Face and the eval faked, and what it did."""
    calls = []

    class Op:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path, self.data = path_in_repo, path_or_fileobj

    class Api:
        def __init__(self, token=None):
            pass

        def create_commit(self, repo_id, repo_type, operations, commit_message):
            calls.append(("commit", repo_id, {o.path: o.data for o in operations}))

    def snapshot_download(repo, local_dir, token=None, allow_patterns=None):
        d = Path(local_dir)
        d.mkdir(parents=True)
        (d / "blockquant.json").write_text(json.dumps(PROV, indent=2))
        (d / "model.safetensors").write_bytes(b"x")

    hub = types.SimpleNamespace(CommitOperationAdd=Op, HfApi=Api,
                                snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    state = {"stats": {"kld_median": 0.0315, "kld": 0.0716}}

    def kl_eval(qdir, model_dir, rows):
        calls.append(("eval", Path(qdir).name, rows))
        if isinstance(state["stats"], Exception):
            raise state["stats"]
        return state["stats"], "qbench · wiki2 · 40×2048"

    ns = {"json": json, "Path": Path, "traceback": traceback,
          "_written_quant_config": lambda d: {"head_bits": 6, "codebook": "mcg",
                                               "calibration": {"rows": 250}},
          "_dir_size_gb": lambda d: 18.7,
          "_kl_div_eval": kl_eval,
          "_finalize_cards": lambda outputs, *a, drafter_repo=None, **k: calls.append(
              ("cards", [o.get("kl_div") for o in outputs], drafter_repo)),
          "_pin_heads": lambda api, outputs: calls.append(("pin", [o["hf_repo_id"] for o in outputs]))}
    fn = _lift("_kl_only", ns)

    def run():
        return fn(["o/M-exl3-4.0bpw"], ["4.0"], model_dir=tmp_path / "model", scratch=tmp_path,
                  model_id="b/M", model_name="M", owner="o", hf_token="t", head_bits=None,
                  cal_rows=None, codebook="mcg", kl_rows=40, drafter_repo="o/D")
    return run, calls, state, tmp_path


def test_the_kl_pod_writes_what_the_conversion_pod_would_have(pod):
    run, calls, _, tmp = pod
    outputs = run()
    commit = next(c for c in calls if c[0] == "commit")
    files = commit[2]
    assert set(files) == {"bq_quality.json", "blockquant.json"}, "one commit, both files"
    quality = json.loads(files["bq_quality.json"])
    assert quality["kl_div"] == 0.0315 and quality["metric"] == "KL(fp16||quant) median"
    prov = json.loads(files["blockquant.json"])
    assert prov["quality"] == {"kl_median": 0.0315, "method": "qbench · wiki2 · 40×2048",
                               "file": "bq_quality.json"}
    assert {k: v for k, v in prov.items() if k != "quality"} == {
        k: v for k, v in PROV.items() if k != "quality"}
    assert outputs[0]["kl_div"] == 0.0315 and outputs[0]["_head_bits"] == 6
    assert not (tmp / "kl-4.0bpw").exists()


def test_the_pin_comes_after_the_number_and_the_card(pod):
    run, calls, _, _ = pod
    run()
    order = [c[0] for c in calls]
    assert order == ["eval", "commit", "cards", "pin"]
    assert calls[2][1:] == ([0.0315], "o/D")


@pytest.mark.parametrize("stats", [None, RuntimeError("OOM")])
def test_a_failed_eval_leaves_the_quant_alone(pod, stats):
    run, calls, state, _ = pod
    state["stats"] = stats
    outputs = run()
    assert [c[0] for c in calls] == ["eval", "pin"]
    assert outputs[0].get("kl_div") is None


def test_a_kl_only_pod_never_reaches_the_converter():
    src = QUANT.read_text()
    main = src[src.index("def main() -> int:"):]
    branch = main.index("if kl_only_repos:\n            drafter_repo")
    # After the base is patched the way a conversion would see it ...
    assert main.index("_ensure_fast_tokenizer(model_dir)") < branch
    # ... and gone before the converter is imported.
    assert branch < main.index("from exllamav3.conversion.convert_model import")
    assert "return _finish(cfg, outputs, drafter_repo, t0)" in main[branch:branch + 900]


def test_the_kl_pod_scores_against_the_base_the_quant_was_built_from():
    main = QUANT.read_text()
    main = main[main.index("def main() -> int:"):]
    assert main.index('base_revision = (_bq.get("base_model") or {}).get("revision")') \
        < main.index("revision=base_revision,")
