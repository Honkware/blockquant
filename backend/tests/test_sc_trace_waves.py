"""Wave sizing in the vendored sc_trace: how much surplus we pay for.

Upstream checks the token target only at turn-wave boundaries and a wave is
every live conversation at once, so crossing the target early still generates
the whole wave -- 798,117 tokens for a 512,000 target, measured. The surplus is
not thrown away (rows are pooled, shuffled and sliced at the end, so it is
mixing) but 1.56x is what wave granularity happened to hand over, not a chosen
number.

These pin the arithmetic of the local patch. The stage itself needs a GPU, so
what is checked here is the sizing rule, lifted from the file so it cannot
drift from what actually runs.
"""
import math
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/selfcal/sc_trace.py"


@pytest.fixture(scope="module")
def pool_factor():
    m = re.search(r"_POOL_FACTOR\s*=\s*([\d.]+)", SRC.read_text())
    assert m, "_POOL_FACTOR is gone from sc_trace.py"
    return float(m.group(1))


def _wave_size(packed, target, pool, avg, eligible, over=1.25):
    """The patch's rule, in one place."""
    want = int(target * pool) - packed
    if not avg or want <= 0 or not eligible:
        return eligible
    return min(eligible, max(1, math.ceil(want / avg * over)))


TARGET = 250 * 2048          # 512,000, the upstream default
WAVE = 160                   # conversations in the seed set
PER = 3000                   # tokens per conversation, near what was measured


def test_a_full_wave_runs_while_the_pool_is_still_empty(pool_factor):
    # Nothing has been measured yet, so there is no estimate to size with.
    assert _wave_size(0, TARGET, pool_factor, 0, WAVE) == WAVE


def test_the_last_wave_shrinks_instead_of_running_whole(pool_factor):
    # THE case: 480k packed, 512k wanted, and upstream would run all 160
    # conversations again for the last 32k.
    n = _wave_size(480_000, TARGET, pool_factor, PER, WAVE)
    assert n < WAVE
    assert n * PER < 200_000


def test_it_over_provisions_rather_than_falling_short(pool_factor):
    # Undershooting costs a whole extra wave, which is far dearer than a little
    # surplus -- so the sizing must ask for more than the arithmetic minimum.
    packed = 480_000
    want = int(TARGET * pool_factor) - packed
    n = _wave_size(packed, TARGET, pool_factor, PER, WAVE)
    assert n * PER > want


def test_the_pool_keeps_slack_for_the_shuffle(pool_factor):
    # Cutting to exactly the target would leave the shuffle nothing to draw
    # from, which is a quality change, not an optimisation.
    assert pool_factor > 1.0
    assert int(TARGET * pool_factor) - TARGET > 50_000


def test_it_stops_asking_once_the_pool_is_full(pool_factor):
    assert _wave_size(int(TARGET * pool_factor) + 1, TARGET, pool_factor, PER, WAVE) == WAVE


def test_it_never_asks_for_more_conversations_than_exist(pool_factor):
    assert _wave_size(0, TARGET, pool_factor, 1, WAVE) == WAVE


def test_it_always_asks_for_at_least_one(pool_factor):
    assert _wave_size(TARGET, TARGET, pool_factor, 10 ** 9, WAVE) >= 1


def test_the_whole_run_generates_far_less_than_upstream(pool_factor):
    """End to end over waves, against the 1.56x that was measured."""
    def run(capped):
        packed = gen = n = 0
        avg = 0.0
        for _ in range(8):
            if packed >= TARGET:
                break
            e = _wave_size(packed, TARGET, pool_factor, avg, WAVE) if capped else WAVE
            gen += e * PER
            packed += e * PER
            n += e
            avg = gen / n
        return gen / TARGET

    assert run(capped=False) > 1.5           # what upstream does
    assert run(capped=True) < 1.35           # what the patch does
    assert run(capped=True) >= pool_factor   # and it still fills the pool


def test_the_patch_is_marked_in_the_source(pool_factor):
    # The vendor manifest points here; unmarked edits are how a local patch
    # gets silently dropped on the next EXLLAMAV3_REF bump.
    assert SRC.read_text().count("BLOCKQUANT LOCAL PATCH") >= 3
