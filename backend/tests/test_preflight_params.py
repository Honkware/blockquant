"""The parameter count on the request card, off the weight size and the dtype.

Newer configs say `dtype` rather than `torch_dtype`, and a multimodal repo says
it under text_config, so a 27.8B with vision showed up as just "Dense".
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PREFLIGHT = Path(__file__).resolve().parents[2] / "scripts" / "preflight.py"


@pytest.fixture(scope="module")
def params_b():
    sys.argv = ["preflight.py"]
    spec = importlib.util.spec_from_file_location("preflight", PREFLIGHT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m._params_b


def test_dtype_under_text_config(params_b):
    assert params_b({"text_config": {"dtype": "bfloat16"}}, 55.56) == 27.8


def test_the_old_spelling_still_works(params_b):
    assert params_b({"torch_dtype": "float32"}, 4.0) == 1.0


def test_an_unknown_dtype_says_nothing(params_b):
    assert params_b({"torch_dtype": "float8_e4m3fn"}, 30.0) is None
    assert params_b({}, 10.0) is None
