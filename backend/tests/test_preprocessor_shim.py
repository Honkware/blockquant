"""The preprocessor config a VL quant has to ship, and why it must be real.

Loaded by AST because remote/quant.py imports the exllamav3 stack at module
scope and none of it is needed here.
"""
import ast
import json

import pytest

SRC = "src/blockquant/remote/quant.py"

WANT = ("_CLIP_MEAN_STD", "_HALF_MEAN_STD", "_VL_PREP_FAMILIES", "_VL_PREP_FALLBACK",
        "_vision_preprocessor_config", "_image_processor_block")


@pytest.fixture
def prep():
    tree = ast.parse(open(SRC).read())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANT)
            or (isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in WANT for t in n.targets))]
    ns = {"json": json, "Path": __import__("pathlib").Path}
    exec(compile(ast.Module(body=body, type_ignores=[]), SRC, "exec"), ns)
    return ns["_vision_preprocessor_config"]


def _model(tmp_path, config, processor=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if processor is not None:
        (tmp_path / "processor_config.json").write_text(json.dumps(processor))
    return tmp_path


def _written(tmp_path):
    return json.loads((tmp_path / "preprocessor_config.json").read_text())


QWEN3_5 = {
    "model_type": "qwen3_5",
    "vision_config": {"model_type": "qwen3_5", "patch_size": 16,
                      "spatial_merge_size": 2, "temporal_patch_size": 2},
}


def test_a_text_only_model_gets_nothing(prep, tmp_path):
    """No vision_config means exllamav3 never opens the file. Don't ship one."""
    prep(_model(tmp_path, {"architectures": ["Qwen3ForCausalLM"]}))
    assert not (tmp_path / "preprocessor_config.json").exists()


def test_a_model_that_ships_its_own_keeps_it(prep, tmp_path):
    d = _model(tmp_path, QWEN3_5)
    (d / "preprocessor_config.json").write_text('{"size": {"shortest_edge": 336}}')
    prep(d)
    assert _written(d)["size"]["shortest_edge"] == 336


def test_the_models_own_processor_config_wins_over_our_derivation(prep, tmp_path):
    """A Qwen3-VL repo carries the real numbers in processor_config.json even
    when the split-out file is missing. Hoist those rather than guess."""
    real = {"size": {"shortest_edge": 65536, "longest_edge": 16777216},
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "image_processor_type": "Qwen2VLImageProcessorFast"}
    prep(_model(tmp_path, QWEN3_5, {"image_processor": real}))
    assert _written(tmp_path) == real


def test_a_half_written_processor_config_is_not_trusted(prep, tmp_path):
    prep(_model(tmp_path, QWEN3_5, {"image_processor": {"patch_size": 16}}))
    assert _written(tmp_path)["size"]["shortest_edge"] == 65536  # derived, not copied


def test_qwen3_derivation_matches_what_qwen_ships(prep, tmp_path):
    prep(_model(tmp_path, QWEN3_5))
    assert _written(tmp_path) == {
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }


def test_qwen2_5_derivation_matches_what_qwen_ships(prep, tmp_path):
    prep(_model(tmp_path, {
        "model_type": "qwen2_5_vl",
        "vision_config": {"model_type": "qwen2_5_vl", "patch_size": 14,
                          "spatial_merge_size": 2, "temporal_patch_size": 2},
    }))
    cfg = _written(tmp_path)
    assert cfg["size"] == {"shortest_edge": 3136, "longest_edge": 12845056}
    assert cfg["image_mean"][0] == 0.48145466


def test_glm4v_gets_the_processor_type_exllamav3_asserts_on(prep, tmp_path):
    prep(_model(tmp_path, {
        "model_type": "glm4v",
        "vision_config": {"model_type": "glm4v", "patch_size": 14,
                          "spatial_merge_size": 2, "temporal_patch_size": 2},
    }))
    cfg = _written(tmp_path)
    assert cfg["image_processor_type"] == "Glm4vImageProcessor"
    assert cfg["size"] == {"shortest_edge": 12544, "longest_edge": 9633792}


def test_an_unknown_family_still_gets_a_loadable_budget(prep, tmp_path):
    prep(_model(tmp_path, {
        "model_type": "somethingnew",
        "vision_config": {"patch_size": 14, "spatial_merge_size": 2},
    }))
    assert _written(tmp_path)["size"]["longest_edge"] == 12845056


def test_what_we_write_survives_smart_resize(prep, tmp_path):
    """The bug the 56px stub caused, as arithmetic.

    exllamav3 reads size->shortest/longest_edge as min_pixels/max_pixels --
    total pixels, not edge lengths -- and qwen2_smart_resize floors the image
    to a multiple of patch*merge under that ceiling. At 56 total pixels a
    1024x1024 image resolves to 0x0 and PIL raises "height and width must be
    > 0". A real ceiling leaves at least one merged token in each direction.
    """
    import math
    prep(_model(tmp_path, QWEN3_5))
    cfg = _written(tmp_path)
    factor = cfg["patch_size"] * cfg["merge_size"]
    h = w = 1024
    beta = math.sqrt(h * w / cfg["size"]["longest_edge"])
    h_bar = math.floor(h / max(beta, 1) / factor) * factor
    assert h_bar > 0 and h_bar % factor == 0
    assert cfg["size"]["shortest_edge"] >= factor ** 2
