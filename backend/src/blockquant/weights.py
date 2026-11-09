"""Which files in a model repo are the model.

Repos carry more than the weights: GGUFs, or an older shard set left beside
the current one. Summing or downloading the whole repo prices, sizes and pulls
all of it, and exllamav3 loads every *.safetensors in the folder, so a second
shard set is a second copy of every tensor.

The index names the shards that make up the model, so that is the source of
truth when there is one. Standalone on purpose: it is shipped next to quant.py
on the pod and imported by path from scripts/preflight.py.
"""
from __future__ import annotations

import json
from pathlib import Path

INDEX = "model.safetensors.index.json"

# Weights in a format the converter never reads.
OTHER_WEIGHTS = (".gguf", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack",
                 ".onnx", ".npz", ".pkl", ".mlmodel", ".tflite")


def pick(files: dict, index: dict | None = None, prefix: str = "") -> tuple[list, int]:
    """(paths to download, weight bytes) out of {repo path: size}.

    Only direct children of `prefix` count: a GGUF/ or original/ folder beside
    the model is not part of it, and exllamav3 only reads the top of model_dir.
    """
    here = {p: s for p, s in files.items()
            if p.startswith(prefix) and "/" not in p[len(prefix):]}
    st = {p for p in here if p.endswith(".safetensors")}
    if index and index.get("weight_map"):
        shards = {prefix + f for f in index["weight_map"].values()} & st
        # An index naming files the repo lacks is broken in a way the converter
        # will report better than we can; fall back to everything.
        if shards:
            st = shards
    rest = [p for p in here if not p.endswith(".safetensors") and not p.endswith(OTHER_WEIGHTS)]
    return sorted(st) + sorted(rest), sum(here[p] or 0 for p in st)


def repo_weights(model_id: str, token: str | None = None, subfolder: str = "",
                 revision: str | None = None):
    """(model_info, paths, weight bytes) for a repo on the hub."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=token or None).model_info(model_id, revision=revision, files_metadata=True)
    files = {s.rfilename: s.size or 0 for s in (info.siblings or [])}
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    index = None
    if prefix + INDEX in files:
        try:
            p = hf_hub_download(model_id, prefix + INDEX, revision=info.sha, token=token or None)
            index = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            index = None
    paths, nbytes = pick(files, index, prefix)
    return info, paths, nbytes
