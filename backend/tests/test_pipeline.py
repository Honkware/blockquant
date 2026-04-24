"""Integration tests for the BlockQuant pipeline."""
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


def test_monitoring_cost_tracking(workspace):
    record_job_start(workspace, "job-lambda", "model/b", "exl3", ["4.0"], "lambda")
    record_job_complete(workspace, "job-lambda", True, 3600.0, "lambda")

    stats = get_daily_stats(workspace)
    assert stats["total_cost_usd"] > 0  # Lambda costs $1.10/hr


@pytest.mark.slow
@pytest.mark.skipif(
    not Path(r"C:\Users\juden\AppData\Local\Temp\blockquant-work\microsoft--Phi-3-mini-4k-instruct\model\config.json").exists(),
    reason="Model not cached — skipping slow pipeline test"
)
def test_pipeline_runs_locally(workspace):
    """Test that the pipeline can run end-to-end with a cached model."""
    config = QuantConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
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
