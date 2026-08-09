"""_upload_folder_hb retry behaviour.

A 27B conversion is hours of GPU and the pod is torn down straight after the
upload, so a transient error there destroys finished work. These pin the two
halves of that: retry what can come good, give up at once on what cannot.

Loaded by AST because remote/quant.py imports the exllamav3 stack at module
scope and none of it is needed here.
"""
import ast
import types

import pytest

SRC = "src/blockquant/remote/quant.py"
WANT = ("_UPLOAD_TRIES", "_UPLOAD_TERMINAL", "_upload_folder_hb")


@pytest.fixture
def mod():
    tree = ast.parse(open(SRC).read())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANT)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in WANT)]
    ns = {"time": types.SimpleNamespace(sleep=lambda s: None), "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=body, type_ignores=[]), SRC, "exec"), ns)
    return ns


class _Api:
    """upload_folder that fails a given number of times, then succeeds."""

    def __init__(self, fails, exc):
        self.fails, self.exc, self.calls = fails, exc, 0

    def upload_folder(self, **_):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc


def _http(code):
    e = Exception(f"{code} from hub")
    e.response = types.SimpleNamespace(status_code=code)
    return e


def test_a_transient_error_is_retried_and_the_upload_survives(mod):
    api = _Api(2, _http(429))
    mod["_upload_folder_hb"](api, "/out", "org/repo", "4.5")
    assert api.calls == 3


def test_a_server_error_is_retried(mod):
    api = _Api(1, _http(503))
    mod["_upload_folder_hb"](api, "/out", "org/repo", "4.5")
    assert api.calls == 2


def test_a_dropped_connection_is_retried(mod):
    api = _Api(1, ConnectionError("reset by peer"))
    mod["_upload_folder_hb"](api, "/out", "org/repo", "4.5")
    assert api.calls == 2


@pytest.mark.parametrize("code", [400, 401, 403, 404, 413])
def test_errors_that_cannot_come_good_fail_on_the_first_try(mod, code):
    api = _Api(99, _http(code))
    with pytest.raises(Exception):
        mod["_upload_folder_hb"](api, "/out", "org/repo", "4.5")
    assert api.calls == 1


def test_it_gives_up_eventually_and_raises_the_real_error(mod):
    api = _Api(99, _http(429))
    with pytest.raises(Exception, match="429"):
        mod["_upload_folder_hb"](api, "/out", "org/repo", "4.5")
    assert api.calls == mod["_UPLOAD_TRIES"]


def test_a_clean_upload_does_not_retry(mod):
    api = _Api(0, _http(429))
    mod["_upload_folder_hb"](api, "/out", "org/repo", "4.5")
    assert api.calls == 1
