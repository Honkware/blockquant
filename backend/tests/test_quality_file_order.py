"""bq_quality.json has to be on disk before the folder upload.

The KL number is written into out_dir and then shipped by _publish's folder
upload. That only works because the write happens first: reverse them and the
file stays on a pod that is about to be terminated, which is what the stale
comment here used to claim was already happening. A dead upload branch guarded
on rec["hf_repo_id"] -- not set until _publish -- was covering for it.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src/blockquant/remote/quant.py"


def _variant_loop():
    """The `for variant in variants:` loop inside main()."""
    for node in ast.walk(ast.parse(SRC.read_text())):
        if (isinstance(node, ast.For) and getattr(node.target, "id", "") == "variant"
                and getattr(node.iter, "id", "") == "variants"):
            return node
    pytest.fail("no `for variant in variants` loop in quant.py")


def _line_of(loop, pred):
    return min((n.lineno for n in ast.walk(loop) if pred(n)), default=None)


def test_the_quality_file_is_written_before_publish_uploads():
    loop = _variant_loop()
    write = _line_of(loop, lambda n: (
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_text"))
    publish = max((n.lineno for n in ast.walk(loop)
                   if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_publish"),
                  default=None)
    assert write and publish, (write, publish)
    assert write < publish, (
        "bq_quality.json is written after the folder upload, so the KL number "
        "never reaches the repo")


def test_no_upload_is_guarded_on_a_repo_id_publish_has_not_set_yet():
    loop = _variant_loop()
    for n in ast.walk(loop):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "upload_file"):
            pytest.fail("upload_file inside the variant loop: rec['hf_repo_id'] "
                        "is not set until _publish, so this cannot run")
