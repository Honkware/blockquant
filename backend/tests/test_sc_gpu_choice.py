"""Which card an SC job lands on, and which it must never land on.

The price cap and the candidate ordering have to agree. _recommend_max_price
floors SC at $1.30 because a slow card costs more in hours than a fast one
costs per hour -- but the ordering did not know about SC, so a small model
still took the cheapest card and the raised cap bought nothing. The first real
SC job ran on a $0.50 3090 for that reason.

Blackwell was excluded twice for having no sm_120 kernels under torch 2.6. The
image has been torch 2.8/cu129 with 12.0 in TORCH_CUDA_ARCH_LIST since before
the second time, and an RTX PRO 6000 Blackwell converted a 27B two days before
it for less than half what an A100 cost. Only sm_100 (B200/B300) is missing.

The order was by price, descending for big work. It is by price over measured
conversion speed now, because the dearest allowed card was an A100, which
converts at about a 3090's rate.
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
@pytest.mark.parametrize("gid", ["NVIDIA B200", "NVIDIA B300 SXM6 AC"])
def test_datacenter_blackwell_is_refused(rj, gid):
    # The wheel has sm_100 kernels, but nothing has run on one. Untested and
    # $5+/h, so they stay off the list until a job proves them.
    assert _excluded(rj, gid)


@pytest.mark.parametrize("gid", [
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 4500 Blackwell",
])
def test_sm120_blackwell_is_a_candidate(rj, gid):
    # The wheel ships 12.0. The Workstation Edition converted a Qwen3.8-27B
    # in a 1905 s pod for $1.16 on 2026-09-13.
    assert not _excluded(rj, gid)


def test_the_image_still_ships_what_that_assumes():
    # turboderp's release wheels build 7.5-12.0+PTX. A source build or another
    # wheel needs this rechecked before sm_120 cards stay on the list.
    df = (ROOT.parent / "docker/Dockerfile.runpod").read_text()
    url = df.split("EXLLAMAV3_WHEEL=")[1].split("\n")[0]
    assert url.startswith("https://github.com/turboderp-org/exllamav3/releases/download/")
    assert "cu128" in url and "cp312" in url


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


def _ranked(rj, monkeypatch, cards, min_vram=24, sc=False):
    fake = type(sys)("runpod")
    fake.get_gpus = lambda: [{"id": g, "memoryInGb": m} for g, m, _ in cards]
    fake.api_key = ""
    monkeypatch.setitem(sys.modules, "runpod", fake)
    prices = {g: p for g, _, p in cards}
    mod = type(sys)("p")
    mod.static_price = lambda g: prices.get(g, 99)
    monkeypatch.setitem(sys.modules, "blockquant.providers.runpod.pricing", mod)
    return rj._auto_gpu_ids("k", min_vram, 3.6, sc=sc)


def test_the_cheapest_card_per_job_goes_first(rj, monkeypatch):
    # Secure prices, 2026-10-27. Price-descending put the A100 first; per job
    # the Blackwell is a third of its cost, and the 3090 beats it on a model
    # that fits.
    cards = [("NVIDIA GeForce RTX 3090", 24, 0.46),
             ("NVIDIA A100-SXM4-80GB", 80, 1.89),
             ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 96, 2.49)]
    order = _ranked(rj, monkeypatch, cards)
    assert order[0] == "NVIDIA GeForce RTX 3090"
    assert order.index("NVIDIA RTX PRO 6000 Blackwell Server Edition") < order.index("NVIDIA A100-SXM4-80GB")


def test_a_card_nobody_has_timed_is_not_preferred_on_a_guess(rj):
    assert rj._convert_speed("NVIDIA Some Future Card") == 1.0


def test_a_big_plain_job_keeps_its_kl_number(rj):
    # Cheapest-per-job would put a 27B on a 24-48 GB card that cannot hold the
    # fp16 for the KL eval, and the card would publish without one.
    assert rj._kl_min_vram(55.6, 24) > 48
    assert rj._kl_min_vram(55.6, 24) <= 80
    # An 8B with KL ran on a 3090; the floor must not push it off one.
    assert rj._kl_min_vram(16.4, 24) == 24
    # Split across cards, each needs its share.
    assert rj._kl_min_vram(55.6, 24, gpu_count=2) <= 48
    # Bigger than any one card: no floor to hold to, keep the old one.
    assert rj._kl_min_vram(400, 24) == 24


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


def test_a_big_model_reaches_the_secure_blackwell_but_not_an_h100(rj):
    # $2.49 secure is half an A100 per job; an H100 HBM3 at $3.49 stays out.
    for sc in (False, True):
        cap = rj._recommend_max_price(51.75, sc=sc)
        assert cap >= 2.49
        assert cap < 3.49


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
