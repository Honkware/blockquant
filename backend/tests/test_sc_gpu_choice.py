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


# Real RunPod prices at the time this was written. The SC cap exists to clear
# the top two, which both hold a 50 GB model resident -- the difference between
# sc_measure running on the GPU and falling back to its CPU-bound streaming
# path, measured at ~13 cores for 16 minutes on a model 30x smaller.
HIGH_VRAM = {
    "NVIDIA A100-SXM4-40GB": 1.29,
    "NVIDIA A100 80GB PCIe": 1.64,
    "NVIDIA A100-SXM4-80GB": 1.89,
    "NVIDIA H100 80GB HBM3": 1.99,
    "NVIDIA H200 NVL": 2.00,
}


def test_a_big_sc_job_can_reach_an_h100_or_h200(rj):
    cap = rj._recommend_max_price(51.75, sc=True)
    reachable = {g for g, p in HIGH_VRAM.items() if p <= cap}
    assert "NVIDIA H100 80GB HBM3" in reachable
    assert "NVIDIA H200 NVL" in reachable


def test_the_old_cap_could_not(rj):
    # The plain tier for this size is $1.80 and both cards sit at $1.99-2.00,
    # so a 20-cent cap was all that stood between a 52 GB model and a 143 GB
    # card. Documents why the SC branch is not just max(1.30, tier).
    plain = rj._recommend_max_price(51.75)
    assert plain < 1.99
    assert rj._recommend_max_price(51.75, sc=True) > 2.00


def test_a_small_sc_job_stays_on_the_cheap_tier(rj):
    # Nothing to gain: the model fits resident on anything, and scarce H200
    # stock should not go to a 0.8B.
    for gb in (1.63, 15.3, 20):
        assert rj._recommend_max_price(gb, sc=True) == 1.30


def test_sc_never_gets_less_card_than_the_same_model_plain(rj):
    for gb in (0, 1.63, 15.3, 30, 51.75, 72, 150):
        assert rj._recommend_max_price(gb, sc=True) >= rj._recommend_max_price(gb)


def test_an_unknown_size_does_not_reach_for_the_top(rj):
    # preflight can fail to read it; that must not silently rent an H200.
    assert rj._recommend_max_price(None, sc=True) <= 1.50


def test_a_big_sc_job_will_not_settle_for_a_card_it_does_not_fit_in(rj):
    """Falling back is a worse trade than waiting, for this one stage.

    sc_measure runs a different algorithm depending on whether the fp16
    weights fit: resident on the GPU, or one module at a time with the states
    in system RAM. Landing on a 40 GB card with a 52 GB model is not "slower",
    it is the CPU-bound path -- so the smaller card leaves the sweep entirely
    rather than acting as a fallback.
    """
    assert rj._sc_min_vram(51.75, 24) > 40
    assert rj._sc_min_vram(51.75, 24) <= 80


def test_the_floor_leaves_small_jobs_alone(rj):
    # They fit anywhere; raising it would only shrink the candidate pool.
    for gb in (1.63, 15.3):
        assert rj._sc_min_vram(gb, 24) == 24


def test_the_floor_only_ever_rises(rj):
    for gb in (0, 1.63, 51.75, 140):
        assert rj._sc_min_vram(gb, 24) >= 24
    assert rj._sc_min_vram(51.75, 96) == 96      # caller asked for more


def test_an_unknown_size_does_not_invent_a_floor(rj):
    assert rj._sc_min_vram(None, 24) == 24
    assert rj._sc_min_vram(0, 24) == 24


def test_it_leaves_headroom_above_the_weights(rj):
    # The measurement rows need activations on top of the weights, so the
    # floor has to sit above the raw model size or the load OOMs and drops
    # back to streaming anyway.
    assert rj._sc_min_vram(51.75, 24) > 51.75
