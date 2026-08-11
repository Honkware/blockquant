"""The preprocessor stub, and why it must not reach a published repo.

Loaded by AST because remote/quant.py imports the exllamav3 stack at module
scope and none of it is needed here.
"""
import ast
import json

import pytest

SRC = "src/blockquant/remote/quant.py"


@pytest.fixture
def shim():
    tree = ast.parse(open(SRC).read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_qwen2vl_preprocessor_shim")
    ns = {"json": json, "Path": __import__("pathlib").Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC, "exec"), ns)
    return ns["_qwen2vl_preprocessor_shim"]


def test_it_reports_writing_one_so_the_caller_can_strip_it(shim, tmp_path):
    assert shim(tmp_path) is True
    assert (tmp_path / "preprocessor_config.json").exists()


def test_a_model_that_ships_its_own_keeps_it(shim, tmp_path):
    (tmp_path / "preprocessor_config.json").write_text('{"size": {"shortest_edge": 336}}')
    assert shim(tmp_path) is False
    # And the real one is untouched.
    assert json.loads((tmp_path / "preprocessor_config.json").read_text())["size"]["shortest_edge"] == 336


def test_the_placeholder_is_not_usable_by_a_vision_loader(shim, tmp_path):
    """Why this file must never ship.

    56px at patch 14 is a 4x4 patch grid, and merge_size 2 squares to 4, so the
    merged grid rounds to zero tokens. That is the "height and width must be
    > 0" a downloader of the Qwopus quants hit. The stub exists only to stop
    convert.py choking while it reads a processor config.
    """
    shim(tmp_path)
    cfg = json.loads((tmp_path / "preprocessor_config.json").read_text())
    side = cfg["size"]["shortest_edge"]
    assert side < 100, "a placeholder, not a real preprocessing size"
    assert (side // cfg["patch_size"]) ** 2 // (cfg["merge_size"] ** 2) <= 4
