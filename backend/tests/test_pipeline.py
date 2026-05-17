"""Integration tests for the BlockQuant pipeline."""
import os
import tempfile
from pathlib import Path

import pytest

from blockquant.models import QuantConfig, QuantFormat
from blockquant.monitoring import (
    record_job_start,
    record_job_complete,
    get_daily_stats,
    get_leaderboard,
    get_recent_jobs,
)
from blockquant.pipeline import run_pipeline


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_quant_config_defaults():
    config = QuantConfig(model_id="test/model")
    assert config.format == QuantFormat.EXL3
    assert config.variants == ["4.0"]
    assert config.provider == "local"
    assert config.verify_quality is True


def test_monitoring_roundtrip(workspace):
    record_job_start(workspace, "job-1", "model/a", "exl3", ["4.0"], "local")
    record_job_complete(workspace, "job-1", True, 120.0, "local")

    stats = get_daily_stats(workspace)
    assert stats["jobs_started"] == 1
    assert stats["jobs_completed"] == 1
    assert stats["successes"] == 1
    assert stats["success_rate"] == 1.0

    leaderboard = get_leaderboard(workspace, limit=10)
    assert len(leaderboard) == 1
    assert leaderboard[0]["model_id"] == "model/a"

    recent = get_recent_jobs(workspace, limit=5)
    assert len(recent) == 1
    assert recent[0]["success"] is True


def _cached_pipeline_workspace() -> Path | None:
    raw = os.environ.get("BLOCKQUANT_TEST_WORKSPACE")
    if not raw:
        return None
    return Path(raw).expanduser()


def _cached_pipeline_model_id() -> str:
    return os.environ.get("BLOCKQUANT_TEST_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")


def _cached_model_config_exists() -> bool:
    workspace = _cached_pipeline_workspace()
    if workspace is None:
        return False
    model_dir = workspace / _cached_pipeline_model_id().replace("/", "--") / "model"
    return (model_dir / "config.json").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not _cached_model_config_exists(),
    reason="Set BLOCKQUANT_TEST_WORKSPACE to a workspace containing the cached model",
)
def test_pipeline_runs_locally():
    """Opt-in smoke test for a local cached-model pipeline run."""
    workspace = _cached_pipeline_workspace()
    assert workspace is not None

    config = QuantConfig(
        model_id=_cached_pipeline_model_id(),
        format=QuantFormat.EXL3,
        variants=["4.0"],
        workspace_dir=workspace,
        verify_quality=False,
    )

    result = run_pipeline(config)

    assert result.status in ("complete", "failed")
    assert len(result.stages) >= 4
    assert result.total_wall_time >= 0

    stats = get_daily_stats(workspace)
    assert stats["jobs_started"] >= 1
