"""flash_attn import stub for the prebuilt blockquant Docker image.

ExLlamaV3 0.0.30+ does `from flash_attn import flash_attn_func, ...` at
module load time inside `exllamav3/modules/attn.py`. Real flash-attn has
no prebuilt wheel for our torch+CUDA12.4 combo and source builds reliably
fail in CI, so we ship a stub that satisfies the import without claiming
to actually accelerate attention.

The quantization code path (the only thing this image is meant for) never
invokes any flash_attn function — only inference does. If something does
call one of these at runtime, we raise loudly so the failure mode is
clear instead of silently producing wrong outputs.

The module-level __getattr__ catches every attribute the importer might
ask for (current API: flash_attn_func, flash_attn_with_kvcache,
flash_attn_varlen_func — but futureproof against new names).
"""


def _stub(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn is stubbed in the blockquant image. The quant path "
        "never calls into flash-attn; if you've reached this from an "
        "inference workflow, install real flash-attn explicitly."
    )


def __getattr__(name):  # PEP 562 — Python 3.7+ module __getattr__.
    return _stub
