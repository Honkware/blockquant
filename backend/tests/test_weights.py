"""Only the files that make up the model get sized, priced and downloaded.

OBLITERATUS/Qwen3.8-27B-OBLITERATED ships its 28 indexed shards beside an older
18-shard set and 128 GB of GGUFs. The launcher read that as a 240 GB model, gave
up on the KL eval because nothing holds a 240 GB fp16, put the job on a 4090
and downloaded all of it, both shard sets into one folder.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("weights", ROOT / "src/blockquant/weights.py")
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)

GB = 10**9


def _obliterated(prefix=""):
    files = {f"{prefix}model-{i:05d}-of-00028.safetensors": 2 * GB for i in range(1, 29)}
    files.update({f"{prefix}model-{i:05d}-of-00018.safetensors": 3 * GB for i in range(1, 19)})
    files[f"{prefix}model-extra-00001-of-00001.safetensors"] = GB
    files.update({f"{prefix}Qwen3.8-27B-OBLITERATED-{q}.gguf": 16 * GB for q in ("Q4_K_M", "Q8_0")})
    files.update({f"{prefix}{n}": 1000 for n in ("config.json", "tokenizer.json",
                                                   "chat_template.jinja", weights.INDEX)})
    shards = [f"model-{i:05d}-of-00028.safetensors" for i in range(1, 29)]
    shards.append("model-extra-00001-of-00001.safetensors")
    index = {"weight_map": {f"t{i}": s for i, s in enumerate(shards)}}
    return files, index


def test_the_index_decides_which_shards_are_the_model():
    files, index = _obliterated()
    paths, nbytes = weights.pick(files, index)
    assert nbytes == 57 * GB
    assert not any("of-00018" in p or p.endswith(".gguf") for p in paths)
    assert {"config.json", "tokenizer.json", "chat_template.jinja", weights.INDEX} <= set(paths)
    assert sum(p.endswith(".safetensors") for p in paths) == 29


def test_no_index_takes_every_safetensors_but_no_other_format():
    files = {"model.safetensors": 8 * GB, "model-Q4.gguf": 4 * GB, "pytorch_model.bin": 8 * GB,
             "config.json": 1000}
    paths, nbytes = weights.pick(files)
    assert paths == ["model.safetensors", "config.json"]
    assert nbytes == 8 * GB


def test_folders_beside_the_model_are_not_part_of_it():
    files = {"model.safetensors": 8 * GB, "config.json": 1000,
             "fp8/model.safetensors": 4 * GB, "original/consolidated.00.pth": 8 * GB}
    paths, nbytes = weights.pick(files)
    assert paths == ["model.safetensors", "config.json"] and nbytes == 8 * GB


def test_a_subfolder_model_is_picked_the_same_way():
    files, index = _obliterated("BF16/")
    files["README.md"] = 1000
    paths, nbytes = weights.pick(files, index, "BF16/")
    assert nbytes == 57 * GB
    assert all(p.startswith("BF16/") for p in paths)


def test_an_index_naming_missing_files_falls_back_to_what_is_there():
    files = {"model.safetensors": 8 * GB, "config.json": 1000}
    paths, nbytes = weights.pick(files, {"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    assert nbytes == 8 * GB


def test_the_pod_downloads_only_the_picked_files():
    src = (ROOT / "src/blockquant/remote/quant.py").read_text()
    main = src[src.index("def main() -> int:"):]
    assert "allow_patterns=_files," in main
    # Pinned before the listing, so the kl-only pod lists the base it scores.
    assert main.index('base_revision = (_bq.get("base_model") or {}).get("revision")') \
        < main.index("repo_weights(model_id, hf_token, subfolder or \"\", base_revision)")


def test_weights_py_reaches_the_pod_and_the_image():
    prov = (ROOT / "src/blockquant/providers/runpod/provider.py").read_text()
    assert 'f"{script_dir}/weights.py"' in prov
    docker = (ROOT.parent / "docker/Dockerfile.runpod").read_text()
    assert "/opt/blockquant/weights.py" in docker


def test_the_launcher_and_the_quote_size_from_the_same_files(monkeypatch):
    sys.path.insert(0, str(ROOT / "src"))
    import blockquant.weights
    from blockquant.providers.runpod.provider import RunPodProvider
    files, index = _obliterated()
    monkeypatch.setattr(blockquant.weights, "repo_weights",
                        lambda *a, **k: (None, *weights.pick(files, index)))
    assert RunPodProvider._base_download_gb("o/m") == 57.0
