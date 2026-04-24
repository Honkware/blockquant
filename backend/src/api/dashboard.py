"""Public dashboard endpoints for BlockQuant."""
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from blockquant.monitoring import get_daily_stats, get_leaderboard, get_recent_jobs

router = APIRouter(prefix="/dashboard")


def _get_workspace() -> Path:
    import tempfile
    return Path(tempfile.gettempdir()) / "blockquant-work"


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Simple HTML dashboard with job stats."""
    stats = get_daily_stats(_get_workspace())
    leaderboard = get_leaderboard(_get_workspace(), limit=10)

    rows = ""
    for i, m in enumerate(leaderboard, 1):
        rows += f"""
        <tr>
            <td>{i}</td>
            <td>{m['model_id']}</td>
            <td>{m['quant_count']}</td>
            <td>{m['top_variant']}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>BlockQuant Dashboard</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #333; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 16px; margin: 24px 0; }}
        .stat {{ background: #f5f5f5; border-radius: 8px; padding: 16px; text-align: center; }}
        .stat-value {{ font-size: 1.8rem; font-weight: bold; color: #2563eb; }}
        .stat-label {{ font-size: 0.85rem; color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #ddd; text-align: left; }}
        th {{ background: #fafafa; font-weight: 600; }}
    </style>
</head>
<body>
    <h1>BlockQuant Dashboard</h1>
    <p>Stats for <strong>{stats['date']}</strong></p>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{stats['jobs_completed']}</div>
            <div class="stat-label">Jobs Today</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats['success_rate']:.0%}</div>
            <div class="stat-label">Success Rate</div>
        </div>
        <div class="stat">
            <div class="stat-value">${stats['total_cost_usd']:.2f}</div>
            <div class="stat-label">Cost Today</div>
        </div>
        <div class="stat">
            <div class="stat-value">{stats['avg_wall_time_seconds']:.0f}s</div>
            <div class="stat-label">Avg Time</div>
        </div>
    </div>

    <h2>Leaderboard</h2>
    <table>
        <thead>
            <tr><th>#</th><th>Model</th><th>Quants</th><th>Variants</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <p style="margin-top: 40px; color: #888; font-size: 0.85rem;">
        BlockQuant v2.0 — Auto-generated dashboard
    </p>
</body>
</html>"""


@router.get("/api/stats")
async def api_stats():
    """JSON stats for external consumption."""
    return get_daily_stats(_get_workspace())


@router.get("/api/leaderboard")
async def api_leaderboard(limit: int = Query(10, ge=1, le=100)):
    """Top quantized models."""
    return get_leaderboard(_get_workspace(), limit=limit)


@router.get("/api/recent")
async def api_recent(limit: int = Query(5, ge=1, le=50)):
    """Recent completed jobs."""
    jobs = get_recent_jobs(_get_workspace(), limit=limit)
    # Convert sets to lists for JSON serialization
    for j in jobs:
        if isinstance(j.get("variants"), set):
            j["variants"] = list(j["variants"])
    return jobs
