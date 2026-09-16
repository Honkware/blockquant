"""The price cap has to bind against what a card actually bills.

The SC cap was raised to $2.10 to reach an H200 NVL "at $2.00". That figure
came from the static fallback table, where "NVIDIA H200 NVL" is not listed --
only "NVIDIA H200" is -- so it fell through to the unlisted default of $2.00,
cleared the cap, and would have billed its real $3.79/hr. A cap evaluated
against a stale number is not a cap.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from blockquant.providers.runpod import pricing  # noqa: E402


def test_the_card_the_sc_cap_reaches_for_is_priced():
    # Not "NVIDIA H200" -- that is a different id and the one that was listed.
    assert "NVIDIA H200 NVL" in pricing.STATIC_PRICES
    assert pricing.static_price("NVIDIA H200 NVL") > 3.0


def test_an_unlisted_card_is_treated_as_expensive():
    # Skipping a cheap new card wastes nothing. Taking an expensive one bills
    # by the hour for as long as the job runs.
    assert pricing.static_price("NVIDIA SOMETHING NEW 2027") >= 9.0


def test_the_cheap_tier_stays_listed():
    # The same fallback must never wrongly EXCLUDE the cards auto-select
    # actually targets.
    for gid in ("NVIDIA GeForce RTX 3090", "NVIDIA A40", "NVIDIA RTX A5000",
                "NVIDIA L40S", "NVIDIA GeForce RTX 4090"):
        assert pricing.static_price(gid) < 1.20, gid


def test_high_end_fallbacks_are_not_understated():
    # These are the ones a cap is supposed to exclude, so erring low defeats it.
    for gid, floor in (("NVIDIA H100 80GB HBM3", 3.0), ("NVIDIA H200 NVL", 3.0),
                       ("NVIDIA H200", 3.0), ("NVIDIA H100 NVL", 2.5)):
        assert pricing.static_price(gid) >= floor, gid


def test_a_211_cap_no_longer_admits_an_h200():
    # The concrete regression: SC on a 50GB model caps at $2.10.
    assert pricing.static_price("NVIDIA H200 NVL") > 2.10
    assert pricing.static_price("NVIDIA H100 80GB HBM3") > 2.10
    # ...while the 80GB A100 that holds such a model resident still fits.
    assert pricing.static_price("NVIDIA A100-SXM4-80GB") <= 2.10


def test_live_lookup_prefers_the_pool_it_is_asked_for():
    class _SDK:
        api_key = None

        @staticmethod
        def get_gpu(_):
            return {"securePrice": 3.79, "communityPrice": 1.19}

    assert pricing.lookup_live_price(_SDK, "k", "NVIDIA H200 NVL", "SECURE") == 3.79
    assert pricing.lookup_live_price(_SDK, "k", "NVIDIA H200 NVL", "COMMUNITY") == 1.19


def test_live_lookup_returns_none_rather_than_guessing():
    class _SDK:
        api_key = None

        @staticmethod
        def get_gpu(_):
            return None

    assert pricing.lookup_live_price(_SDK, "k", "x", "SECURE") is None
