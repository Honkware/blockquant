"""RunPod GPU catalogue lookups — price, and how many fit in one pod."""

import json
from pathlib import Path

# Fallback prices used only when the live lookup fails. Values are the SECURE
# rate (the higher of the two clouds) so the --max-price cap fails safe. Keep
# the cheap VRAM-appropriate tier listed so a lookup hiccup never wrongly
# EXCLUDES them.
#
# An unlisted card is treated as expensive, not as $2.00. The old default did
# the opposite of what this comment claimed: "NVIDIA H200 NVL" is not the same
# id as "NVIDIA H200", so it fell through to 2.00, cleared a $2.10 cap, and
# billed its real $3.79. A default below the high-end tier cannot fail safe --
# it is exactly the case it was meant to guard.
# The prices themselves live in backend/gpu_table.json with each card's
# conversion speed, so the bot's quote ranks cards the way the launcher does.
_TABLE = json.loads((Path(__file__).resolve().parents[4] / "gpu_table.json").read_text())
STATIC_PRICES = {g: c["price"] for g, c in _TABLE["cards"].items() if "price" in c}

# What an unlisted card costs, for capping purposes. High on purpose: skipping
# a cheap new card wastes nothing, taking an expensive one bills for hours.
UNKNOWN_PRICE = 9.99


def lookup_live_price(runpod_sdk, api_key: str, gpu_type: str, cloud_type: str) -> float | None:
    runpod_sdk.api_key = api_key
    gpu = runpod_sdk.get_gpu(gpu_type)
    if not gpu:
        return None
    if cloud_type == "SECURE":
        price = gpu.get("securePrice") or gpu.get("communityPrice")
    else:
        price = gpu.get("communityPrice") or gpu.get("securePrice")
    if not price:
        lowest_price = gpu.get("lowestPrice")
        if isinstance(lowest_price, dict):
            price = lowest_price.get("uninterruptablePrice")
    if isinstance(price, (int, float)) and price > 0:
        return float(price)
    return None


def static_price(gpu_type: str) -> float:
    return STATIC_PRICES.get(gpu_type, UNKNOWN_PRICE)


def lookup_max_gpu_count(runpod_sdk, api_key: str, gpu_type: str) -> int | None:
    """Most of this card RunPod will put in a single pod, or None when the
    lookup fails. Plenty of listed cards only ever come one to a pod, and asking
    for more of them is taken by the API and failed, which costs a whole slot in
    the launch sweep per pass. None means "unknown, let the create decide"."""
    runpod_sdk.api_key = api_key
    gpu = runpod_sdk.get_gpu(gpu_type)
    n = (gpu or {}).get("maxGpuCount")
    return int(n) if isinstance(n, int) and n > 0 else None
