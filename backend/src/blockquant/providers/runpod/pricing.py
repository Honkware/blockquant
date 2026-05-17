"""RunPod GPU price lookup helpers."""

STATIC_PRICES = {
    "NVIDIA RTX A4000": 0.32,
    "NVIDIA RTX A4500": 0.44,
    "NVIDIA RTX A5000": 0.46,
    "NVIDIA RTX A6000": 0.79,
    "NVIDIA A40": 0.47,
    "NVIDIA A100 80GB PCIe": 1.64,
    "NVIDIA A100-SXM4-80GB": 1.89,
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
