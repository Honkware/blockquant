"""Which card an SC job lands on, and which it must never land on.

The price cap and the candidate ordering have to agree. _recommend_max_price
floors SC at $1.30 because a slow card costs more in hours than a fast one
costs per hour -- but the ordering did not know about SC, so a small model
still took the cheapest card and the raised cap bought nothing. The first real
SC job ran on a $0.50 3090 for that reason.

Raising what SC can reach also reaches cards that were previously out of
budget, which is how the Blackwell list turned out to be narrower than its own
comment: torch 2.6/cu124 has no sm_120 kernels, so a 5090 dies mid-run with
"no kernel image is available for execution on the device".
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def rj():
    sys.argv = ["run_runpod_job.py"]
    spec = importlib.util.spec_from_file_location("rj", ROOT / "scripts/run_runpod_job.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _excluded(m, gid: str) -> bool:
    return any(tok in gid for tok in m._BLACKWELL_EXCLUDE)


# Real RunPod GPU ids. The Ada ones are the trap: an "RTX 5000"/"RTX 6000"
# substring rule would drop perfectly good cards we rely on.
@pytest.mark.parametrize("gid", [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA GeForce RTX 5080",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA RTX PRO 4500 Blackwell Server Edition",
    "NVIDIA B200",
])
def test_blackwell_is_refused(rj, gid):
    assert _excluded(rj, gid), f"{gid} has no cu124 kernels and would die mid-run"


@pytest.mark.parametrize("gid", [
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L40S",
])
def test_working_cards_are_kept(rj, gid):
    assert not _excluded(rj, gid), f"{gid} runs fine and must stay a candidate"


def test_sc_raises_the_cap_on_a_small_model(rj):
    # 0.8B. Without the SC floor this is the $0.80 tier, which no A100 fits.
    assert rj._recommend_max_price(3.6) == 0.80
    assert rj._recommend_max_price(3.6, sc=True) == 1.30


def test_sc_sorts_capable_first_so_the_raised_cap_means_something(rj, monkeypatch):
    cards = [("NVIDIA GeForce RTX 3090", 24, 0.46),
             ("NVIDIA A100-SXM4-40GB", 40, 1.29),
             ("NVIDIA L40S", 48, 0.86)]
    monkeypatch.setattr(rj, "_auto_gpu_ids", rj._auto_gpu_ids)
    fake = type(sys)("runpod")
    fake.get_gpus = lambda: [{"id": g, "memoryInGb": m} for g, m, _ in cards]
    fake.api_key = ""
    monkeypatch.setitem(sys.modules, "runpod", fake)
    prices = {g: p for g, _, p in cards}
    monkeypatch.setitem(
        sys.modules, "blockquant.providers.runpod.pricing",
        type(sys)("p"))
    sys.modules["blockquant.providers.runpod.pricing"].static_price = lambda g: prices.get(g, 99)

    cheap = rj._auto_gpu_ids("k", 24, 3.6, sc=False)
    capable = rj._auto_gpu_ids("k", 24, 3.6, sc=True)
    assert cheap[0] == "NVIDIA GeForce RTX 3090"
    assert capable[0] == "NVIDIA A100-SXM4-40GB"
