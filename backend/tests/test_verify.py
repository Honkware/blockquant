from pathlib import Path

import pytest

from blockquant.models import QuantConfig, QuantFormat, QuantOutput, VerificationStatus
from blockquant.stages import verify


def test_exl3_verify_sets_passed_status(tmp_path):
    out_dir = tmp_path / "4.0bpw"
    out_dir.mkdir()
    (out_dir / "model.safetensors").write_text("stub", encoding="utf-8")
    output = QuantOutput(
        variant="4.0",
        format=QuantFormat.EXL3,
        output_path=str(out_dir),
    )

    verify.run(QuantConfig(model_id="tester/model"), tmp_path, [output])

    assert output.verified is True
    assert output.verification.status == VerificationStatus.PASSED
    assert output.verification.method == "filesystem"


def test_verify_marks_missing_output_before_raising(tmp_path):
    missing = tmp_path / "missing"
    output = QuantOutput(
        variant="4.0",
        format=QuantFormat.EXL3,
        output_path=str(missing),
    )

    with pytest.raises(FileNotFoundError):
        verify.run(QuantConfig(model_id="tester/model"), tmp_path, [output])

    assert output.verified is False
    assert output.verification.status == VerificationStatus.FAILED
    assert "Output missing" in output.verification.message


def test_gguf_verify_is_explicit_when_llama_cpp_is_missing(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("stub", encoding="utf-8")

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    result = verify._verify_gguf(Path(gguf))

    assert result.status == VerificationStatus.SKIPPED
    assert result.method == "llama_cpp"
    assert "not installed" in result.message
