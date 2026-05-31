"""RunPod GPU price lookup helpers."""

# Fallback prices used only when the live lookup fails. Values are the SECURE
# rate (the higher of the two clouds) so the --max-price cap fails safe. Any
# card not listed defaults to 2.00, which keeps the cap from accidentally
# grabbing an unknown high-end card on a price-lookup hiccup. Keep the cheap
# VRAM-appropriate tier listed so that same hiccup never wrongly EXCLUDES them.
STATIC_PRICES = {
    # Cheap consumer / workstation tier (what auto-select targets).
    "NVIDIA RTX 4000 Ada Generation": 0.26,
    "NVIDIA RTX 4000 SFF Ada Generation": 0.26,
    "NVIDIA RTX 2000 Ada Generation": 0.28,
    "NVIDIA GeForce RTX 3090": 0.46,
    "NVIDIA GeForce RTX 3090 Ti": 0.46,
    "NVIDIA L4": 0.44,
    "NVIDIA GeForce RTX 4090": 0.69,
    "NVIDIA RTX 5000 Ada Generation": 0.83,
    "NVIDIA GeForce RTX 5090": 0.94,
    "NVIDIA RTX A4000": 0.32,
    "NVIDIA RTX A4500": 0.44,
    "NVIDIA RTX A5000": 0.46,
    "NVIDIA RTX A6000": 0.79,
    "NVIDIA RTX 6000 Ada Generation": 0.77,
    "NVIDIA A40": 0.47,
    "NVIDIA L40": 0.99,
    "NVIDIA L40S": 0.86,
    # Pricey tier (kept accurate so the cap reliably skips them).
    "NVIDIA A100 80GB PCIe": 1.64,
    "NVIDIA A100-SXM4-80GB": 1.89,
    "NVIDIA A100-SXM4-40GB": 1.29,
    "NVIDIA H100 PCIe": 2.39,
    "NVIDIA H100 80GB HBM3": 1.99,
    "NVIDIA H100 NVL": 2.79,
    "NVIDIA H200": 3.99,
}


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
    return STATIC_PRICES.get(gpu_type, 2.00)
