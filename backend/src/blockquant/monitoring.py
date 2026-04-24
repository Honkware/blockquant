"""Stats persistence and cost tracking for BlockQuant.

Stores append-only JSON lines in workspace_dir/blockquant-stats.jsonl.
Each line is a job record with start time, completion time, success/failure,
provider, model, variant, and estimated cost.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from blockquant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PROVIDER_RATES = {
    "local": 0.0,
    "lambda": 1.10,  # A10 hourly
}


def _get_stats_path(workspace_dir: Path) -> Path:
    return workspace_dir / "blockquant-stats.jsonl"


def record_job_start(
    workspace_dir: Path,
    job_id: str,
    model_id: str,
    format: str,
    variants: list[str],
    provider: str,
) -> None:
    """Append a job-start record to the stats file."""
    path = _get_stats_path(workspace_dir)
    record = {
        "event": "start",
        "job_id": job_id,
        "model_id": model_id,
        "format": format,
        "variants": variants,
        "provider": provider,
        "timestamp": time.time(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def record_job_complete(
    workspace_dir: Path,
    job_id: str,
    success: bool,
    wall_time_seconds: float,
    provider: str = "local",
) -> None:
    """Append a job-complete record to the stats file."""
    path = _get_stats_path(workspace_dir)
    hours = wall_time_seconds / 3600
    rate = DEFAULT_PROVIDER_RATES.get(provider, 0.0)
    cost_usd = hours * rate

    record = {
        "event": "complete",
        "job_id": job_id,
        "success": success,
        "wall_time_seconds": wall_time_seconds,
        "provider": provider,
        "cost_usd": round(cost_usd, 4),
        "timestamp": time.time(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _read_records(workspace_dir: Path) -> list[dict]:
    path = _get_stats_path(workspace_dir)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def get_daily_stats(workspace_dir: Path) -> dict:
    """Return stats for today."""
    records = _read_records(workspace_dir)
    now = datetime.now()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = start_of_day.timestamp()

    today_jobs = [r for r in records if r.get("timestamp", 0) >= start_ts]
    completed = [r for r in today_jobs if r.get("event") == "complete"]
    successes = [r for r in completed if r.get("success")]

    total_cost = sum(r.get("cost_usd", 0) for r in completed)
    avg_time = (
        sum(r.get("wall_time_seconds", 0) for r in successes) / len(successes)
        if successes else 0
    )

    return {
        "date": now.strftime("%Y-%m-%d"),
        "jobs_started": len([r for r in today_jobs if r.get("event") == "start"]),
        "jobs_completed": len(completed),
        "successes": len(successes),
        "failures": len(completed) - len(successes),
        "success_rate": len(successes) / len(completed) if completed else 0.0,
        "total_cost_usd": round(total_cost, 4),
        "avg_wall_time_seconds": round(avg_time, 1),
    }


def get_leaderboard(workspace_dir: Path, limit: int = 10) -> list[dict]:
    """Return most-quantized models."""
    records = _read_records(workspace_dir)
    model_counts: dict[str, dict] = {}

    for r in records:
        if r.get("event") != "start":
            continue
        model_id = r.get("model_id", "unknown")
        variant = ",".join(r.get("variants", []))
        if model_id not in model_counts:
            model_counts[model_id] = {"model_id": model_id, "count": 0, "variants": set()}
        model_counts[model_id]["count"] += 1
        model_counts[model_id]["variants"].add(variant)

    sorted_models = sorted(model_counts.values(), key=lambda x: x["count"], reverse=True)
    return [
        {
            "model_id": m["model_id"],
            "quant_count": m["count"],
            "top_variant": ", ".join(sorted(m["variants"]))[:50],
        }
        for m in sorted_models[:limit]
    ]


def get_recent_jobs(workspace_dir: Path, limit: int = 5) -> list[dict]:
    """Return recent completed jobs."""
    records = _read_records(workspace_dir)
    # Pair start/complete records by job_id
    jobs: dict[str, dict] = {}
    for r in records:
        jid = r.get("job_id", "")
        if r.get("event") == "start":
            jobs[jid] = {**r, "success": None, "wall_time_seconds": 0, "cost_usd": 0}
        elif r.get("event") == "complete" and jid in jobs:
            jobs[jid]["success"] = r.get("success")
            jobs[jid]["wall_time_seconds"] = r.get("wall_time_seconds", 0)
            jobs[jid]["cost_usd"] = r.get("cost_usd", 0)

    # Sort by timestamp desc
    sorted_jobs = sorted(jobs.values(), key=lambda x: x.get("timestamp", 0), reverse=True)
    return sorted_jobs[:limit]


def check_cost_alert(workspace_dir: Path, max_cost_usd: float = 5.0) -> list[str]:
    """Return warning messages for jobs that exceeded the cost threshold."""
    records = _read_records(workspace_dir)
    alerts: list[str] = []
    for r in records:
        if r.get("event") == "complete":
            cost = r.get("cost_usd", 0)
            if cost > max_cost_usd:
                alerts.append(
                    f"Job {r.get('job_id', '?')} on {r.get('provider', '?')} "
                    f"cost ${cost:.2f} (threshold: ${max_cost_usd:.2f})"
                )
    return alerts
