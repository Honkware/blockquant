#!/usr/bin/env python3
"""BLOCKQUANT // LIVE OPS — browser dashboard for the RunPod quant log.

Tails backend/logs/runpod-qwen35b-4.5.log and renders an operator-console-style
dashboard at http://localhost:8088. Streams updates over SSE every 2s.

Two layers of cleverness:

  1. LogParser walks the log with named regex rules and emits typed events
     (BOOT, NET, DOWNLOAD, QUANT_TENSOR, QUANT_LAYER, UPLOAD, ERROR, etc.).
     New patterns can be added by appending to RULES — no other changes needed.

  2. Derived stats: per-layer wall time, tensor rate, instantaneous ETA refinement,
     stage timestamps. Dashboard surfaces these in dedicated panels.

Usage:
    ./venv/Scripts/python backend/scripts/log_dashboard.py
    # then open http://localhost:8088
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO_ROOT / "backend" / "logs" / "runpod-qwen35b-4.5.log"

# Mutable at startup — populated by main() from --log / --logs-dir.
JOB_LOGS: dict[str, Path] = {}
LOGS_DIR: Path | None = None
LOG_PATTERN: str = "runpod-*.log"

SESSION_ID = "-".join(f"{secrets.randbelow(10000):04d}" for _ in range(5))


def _derive_job_id(log_path: Path) -> str:
    """Job ID from log filename. `runpod-qwen35b-4.0.log` -> `qwen35b-4.0`.
    Falls back to the bare stem if the prefix isn't present."""
    stem = log_path.stem  # strips .log
    return stem[7:] if stem.startswith("runpod-") else stem


def _discover_logs() -> dict[str, Path]:
    """Build the live job → log map. Returns explicitly-passed --log entries
    plus any new ones in --logs-dir matching --log-pattern. Re-evaluated each
    request so newly-created log files appear without a dashboard restart."""
    out: dict[str, Path] = dict(JOB_LOGS)
    if LOGS_DIR and LOGS_DIR.is_dir():
        for p in sorted(LOGS_DIR.glob(LOG_PATTERN)):
            jid = _derive_job_id(p)
            out.setdefault(jid, p)
    return out


# ---------------------------------------------------------------------------
# LogParser — named regex rules that emit typed events.
#
# Each rule: (event_kind, severity, regex, extractor)
#   - event_kind: short tag for the dashboard ("POD", "QUANT", etc.)
#   - severity: "info" | "ok" | "warn" | "bad"
#   - regex: compiled
#   - extractor: callable that turns a regex Match into a dict of fields,
#                or None to use named groups directly.
# ---------------------------------------------------------------------------

@dataclass
class Event:
    kind: str
    severity: str
    message: str
    line_no: int
    fields: dict = field(default_factory=dict)


def _r(p: str) -> re.Pattern:
    return re.compile(p)


RULES: list[tuple[str, str, re.Pattern, Callable[[re.Match, str], dict] | None]] = [
    # Job header (printed by run_runpod_job.py before any pod work)
    ("JOB",      "info", _r(r"\[job\]\s+model=(?P<model>\S+)\s+variants=(?P<variants>\S+)"
                             r"\s+format=(?P<format>\S+)(?:\s+head_bits=(?P<head_bits>\S+))?"
                             r"(?:\s+hf_org=(?P<hf_org>\S+))?"), None),

    # Remote-script download line — also our fallback source of model_id
    # for jobs launched before the [job] header was added.
    ("DOWNLOAD_REPO", "info",
        _r(r"\[download\]\s+(?P<repo>[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)"), None),

    # Pod lifecycle
    ("GPU_TRY",  "info", _r(r"\[1/6\]\s*Trying\s+(?P<gpu>[^(]+?)\s+\(~\$(?P<rate>[0-9.]+)/hr"),  None),
    ("STOCK",    "warn", _r(r"out of stock(?:\s*\((?P<err>[^)]+)\))?"), None),
    ("POD_NEW",  "ok",   _r(r"Pod ID:\s*(?P<pod_id>\S+)\s*\(GPU:\s*(?P<gpu>[^)]+)\)"), None),
    ("SSH",      "ok",   _r(r"SSH ready at\s+(?P<host>[^:]+):(?P<port>\d+)"), None),

    # Bootstrap stages
    ("INTERP",   "info", _r(r"remote python:\s*(?P<py>\S+)\s+torch:\s*(?P<torch>\S+)"), None),
    ("APT",      "info", _r(r"Installing system packages"), None),
    ("HF_DEPS",  "info", _r(r"Installing HF deps"), None),
    ("EXL_REQS", "info", _r(r"Installing exllamav3 requirements\.txt"), None),
    ("FA_SKIP",  "info", _r(r"Skipping flash-attn"), None),
    ("UPLOAD_EXL", "info", _r(r"Uploading local exllamav3"), None),
    ("FORMATRON_SHIM", "warn", _r(r"formatron import broken; stubbing"), None),
    ("HEALTH",   "ok",   _r(r"\[ok\] exllamav3 imported"), None),
    ("BOOT_OK",  "ok",   _r(r"Bootstrap complete"), None),

    # Remote pipeline
    ("DOWNLOAD", "info", _r(r"\[download\]\s*(?P<msg>.+)"), None),
    ("UPLOAD",   "info", _r(r"\[upload\]\s*(?P<msg>.+)"), None),
    # Strip JSON / quote / paren punctuation that often follows a URL when
    # it's embedded in stdout JSON (`"hf_url": "https://..."}` etc.).
    ("HF_URL",   "ok",   _r(r"(?P<url>https?://huggingface\.co/[A-Za-z0-9._/\-]+)"), None),

    # Quantization (the noisy ones — tracked but not surfaced individually)
    ("QUANT_TENSOR", "info",
        _r(r" -- Quantized:\s*model\.\S*?layers\.(?P<layer>\d+)(?:\.\S*?experts\.(?P<expert>\d+))?"
           r".*?bpw:\s*(?P<bpw>[0-9.]+)"), None),
    ("QUANT_HEAD",   "info",
        _r(r" -- Quantized:\s*lm_head\s+bpw:\s*(?P<bpw>[0-9.]+)"), None),
    ("ETA",     "info", _r(r" -- Estimated remaining time:\s*(?P<eta>.+)"), None),
    ("CREATE_DIR", "info", _r(r" -- Creating directory\s*(?P<dir>\S+)"), None),

    # Network / SSH transient hiccups — handled by the retry layer in
    # RunPodProvider. Severity is "info" because they're normal-operation
    # events; the parser also drops them from the visible Ledger stream.
    ("NET_RESET", "info",
        _r(r"connection (?:reset|aborted|forcibly closed)|10054|EOFError|Server connection dropped"), None),
    ("RETRY",     "warn", _r(r"transient error, retry\s+(?P<n>\d+)/\d+"), None),

    # Hard failures
    ("BOOT_FAIL", "bad",  _r(r"ERROR:\s*bootstrap failed"), None),
    ("HF_AUTH",   "bad",  _r(r"Invalid (?:user token|username or password)"), None),
    ("DRIVER",    "bad",  _r(r"NVIDIA driver on your system is too old"), None),
    ("MODULE_MISS", "bad", _r(r"ModuleNotFoundError:\s*No module named '(?P<mod>[^']+)'"), None),
    ("OOM",       "bad",  _r(r"CUDA out of memory|OutOfMemoryError"), None),

    # Lifecycle close
    ("TERMINATE", "info", _r(r"Terminating pod\s+(?P<pod>\S+)"), None),
    ("STATUS",    "ok",   _r(r"Status:\s*(?P<status>\S+)"), None),
]


@dataclass
class ParseResult:
    events: list[Event]
    state: dict
    stats: dict
    raw_tail: list[str]


_TS_RE = re.compile(r"\b(\d{2}):(\d{2}):(\d{2})\b")


def _line_seconds(line: str) -> int | None:
    """Extract HH:MM:SS from a log line and return seconds-since-midnight."""
    m = _TS_RE.search(line)
    if not m:
        return None
    h, mn, s = (int(g) for g in m.groups())
    return h * 3600 + mn * 60 + s


def parse_log(text: str, tail_n: int = 60, file_mtime: float | None = None) -> ParseResult:
    state = {
        "session_id": SESSION_ID,
        "model_id": "",
        "variants": "",
        "format": "exl3",
        "head_bits": "",
        "hf_org": "",
        "pod_id": "",
        "gpu": "",
        "cost_per_hr": 0.0,
        "remote_py": "",
        "torch_ver": "",
        "current_layer": None,
        "current_expert": None,
        "tensors_quantized": 0,
        "head_quantized": False,
        "eta": "",
        "bootstrap_complete_at": None,
        "download_started": False,
        "download_done": False,
        "quantize_started": False,
        "upload_started": False,
        "hf_url": "",
        "terminated": False,
        "stage": "init",
        "iteration_count": 0,
        # Server-computed durations (in seconds) — keep dashboard refresh-safe.
        "pod_runtime_sec": 0.0,
        "pod_spent": 0.0,
        "quant_elapsed_sec": 0.0,
        "layer_avg_sec": 0.0,
        "tensor_rate_per_sec": 0.0,
        "source_hf_url": "",
        "predicted_hf_url": "",
        # Largest layer / expert index ever observed in QUANT_TENSOR lines.
        # Lets the front-end size the matrix without hardcoding 48×256.
        "max_layer_observed": 0,
        "max_expert_observed": 0,
    }
    events: list[Event] = []
    # Walk-clock seconds-since-midnight, anchored on the most recent stamped line.
    last_seen_clock: int | None = None
    pod_create_clock: int | None = None
    bootstrap_clock: int | None = None
    quant_first_clock: int | None = None
    quant_last_clock: int | None = None
    layer_first_clock: dict[int, int] = {}
    layer_first_seen_order: list[int] = []

    lines = text.splitlines()
    for ln_no, line in enumerate(lines, 1):
        clk = _line_seconds(line)
        if clk is not None:
            last_seen_clock = clk
        for kind, sev, regex, extractor in RULES:
            m = regex.search(line)
            if not m:
                continue
            fields = extractor(m, line) if extractor else m.groupdict()

            # State updates
            if kind == "JOB":
                state["model_id"] = fields.get("model", "")
                state["variants"] = fields.get("variants", "")
                state["format"] = fields.get("format", "exl3")
                state["head_bits"] = fields.get("head_bits", "") or ""
                # Treat the legacy "(personal)" placeholder as empty so the
                # predicted URL doesn't render literal parens as an org.
                hf_org_raw = fields.get("hf_org", "") or ""
                state["hf_org"] = "" if hf_org_raw == "(personal)" else hf_org_raw
            elif kind == "GPU_TRY":
                state["iteration_count"] += 1
                try: state["cost_per_hr"] = float(fields.get("rate") or 0.0)
                except ValueError: pass
            elif kind == "POD_NEW":
                state["pod_id"] = fields.get("pod_id", "")
                state["gpu"] = (fields.get("gpu") or "").strip()
                state["stage"] = "provisioning"
                # Anchor pod runtime on the timestamp of THIS line (the create
                # log line carries HH:MM:SS); fall back to last_seen if missing.
                if pod_create_clock is None:
                    pod_create_clock = last_seen_clock
            elif kind == "SSH":
                state["stage"] = "bootstrapping"
            elif kind == "INTERP":
                state["remote_py"] = fields.get("py", "")
                state["torch_ver"] = fields.get("torch", "")
            elif kind == "BOOT_OK" and state["bootstrap_complete_at"] is None:
                state["bootstrap_complete_at"] = time.time()
                bootstrap_clock = last_seen_clock
                state["stage"] = "downloading"
            elif kind == "DOWNLOAD_REPO":
                # Backfill model_id from the [download] {org/repo} line when
                # the [job] header line is missing (true for jobs launched
                # before that line was added).
                if not state["model_id"]:
                    state["model_id"] = fields.get("repo", "")
                state["download_started"] = True
                state["stage"] = "downloading"
            elif kind == "DOWNLOAD":
                state["download_started"] = True
                state["stage"] = "downloading"
                if "complete" in (fields.get("msg") or "").lower():
                    state["download_done"] = True
                    state["stage"] = "quantizing"
            elif kind == "UPLOAD":
                state["upload_started"] = True
                msg = (fields.get("msg") or "")
                # Catch "[upload] {variant} -> {org}/{repo} ..." — the
                # repo_id is known the moment upload begins, so we can
                # populate hf_url even if the trailing URL line never
                # flushed (older quant.py builds didn't echo it).
                m = re.search(r"->\s*([A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+)", msg)
                if m:
                    state["hf_url"] = f"https://huggingface.co/{m.group(1)}"
                    # The upload line is authoritative — overwrite any
                    # placeholder org that may have been parsed from the
                    # [job] header so predicted_hf_url stops rendering it.
                    state["hf_org"] = m.group(1).split("/")[0]
                if "complete" in msg.lower():
                    state["stage"] = "complete"
                elif state["stage"] != "complete":
                    state["stage"] = "uploading"
            elif kind == "HF_URL":
                state["hf_url"] = fields.get("url", "")
                state["stage"] = "complete"
            elif kind == "QUANT_TENSOR":
                state["quantize_started"] = True
                state["tensors_quantized"] += 1
                if state["stage"] not in ("uploading", "complete"):
                    state["stage"] = "quantizing"
                # Anchor against the most recent stamped log line; -- Quantized
                # rows themselves are unstamped but the surrounding logger lines
                # carry HH:MM:SS, so last_seen_clock is the right approximation.
                if quant_first_clock is None and last_seen_clock is not None:
                    quant_first_clock = last_seen_clock
                if last_seen_clock is not None:
                    quant_last_clock = last_seen_clock
                try:
                    layer = int(fields.get("layer") or -1)
                    if layer >= 0:
                        state["current_layer"] = layer
                        if layer > state["max_layer_observed"]:
                            state["max_layer_observed"] = layer
                        if layer not in layer_first_clock and last_seen_clock is not None:
                            layer_first_clock[layer] = last_seen_clock
                            layer_first_seen_order.append(layer)
                except ValueError:
                    pass
                try:
                    expert = int(fields.get("expert") or -1)
                    if expert >= 0:
                        state["current_expert"] = expert
                        if expert > state["max_expert_observed"]:
                            state["max_expert_observed"] = expert
                except ValueError:
                    pass
            elif kind == "QUANT_HEAD":
                state["head_quantized"] = True
            elif kind == "ETA":
                state["eta"] = (fields.get("eta") or "").strip()
            elif kind == "TERMINATE":
                state["terminated"] = True
                if state["stage"] not in ("complete",):
                    state["stage"] = "terminated"

            # Only record one event per line — first matching rule wins.
            # Filter QUANT_TENSOR (50k+ noise) and NET_RESET (handled by
            # the retry layer; only adds ledger spam) out of the visible
            # events stream. They still affect state/stats above.
            if kind not in ("QUANT_TENSOR", "NET_RESET"):
                events.append(Event(kind=kind, severity=sev, message=line.strip(),
                                    line_no=ln_no, fields=fields))
            break

    # Derived stats — all anchored on log timestamps + file mtime so they
    # survive a browser refresh (no client-side wall clock involved).
    DAY = 24 * 3600

    def _elapsed(a: int | None, b: int | None) -> float | None:
        if a is None or b is None:
            return None
        return float((b - a) % DAY)

    now_clock = None
    if file_mtime:
        lt = time.localtime(file_mtime)
        now_clock = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec

    # Pod runtime + spend (in state, not stats — these are foreground numbers).
    if pod_create_clock is not None and now_clock is not None and not state["terminated"]:
        rt = _elapsed(pod_create_clock, now_clock) or 0.0
        state["pod_runtime_sec"] = rt
        if state["cost_per_hr"]:
            state["pod_spent"] = rt / 3600 * state["cost_per_hr"]

    # Quant elapsed + tensor rate.
    if quant_first_clock is not None and state["tensors_quantized"] > 0:
        anchor_end = now_clock if now_clock is not None else quant_last_clock
        d = _elapsed(quant_first_clock, anchor_end)
        if d and d >= 1:
            state["quant_elapsed_sec"] = d
            state["tensor_rate_per_sec"] = state["tensors_quantized"] / d

    # Sec/layer: anchor on quant_first_clock + completed-layer count.
    # current_layer is the index we're working on, so layers_done = current_layer.
    if (state["current_layer"] is not None and state["current_layer"] > 0
            and state["quant_elapsed_sec"] > 0):
        state["layer_avg_sec"] = state["quant_elapsed_sec"] / state["current_layer"]

    stats: dict = {
        "layer_count_observed": len(layer_first_seen_order),
        "pod_create_clock": pod_create_clock,
        "now_clock": now_clock,
    }

    # Derived URLs — visible the moment we know the model_id; the predicted
    # output URL is a best-guess that lights up "live" only after upload.
    # Convention: {org}/{basename}-{variant}bpw-{format} — one repo per bpw,
    # matches the upload slug in runpod_provider._QUANT_SCRIPT.
    if state["model_id"]:
        state["source_hf_url"] = f"https://huggingface.co/{state['model_id']}"
        basename = state["model_id"].split("/")[-1]
        # Only emit predicted_hf_url when the real namespace is known.
        # Otherwise the dashboard would render "{your-account}/..." for
        # the entire pre-upload window (~3 hours).
        if state["hf_org"]:
            variant = (state["variants"].split(",")[0].strip()
                       if state["variants"] else "")
            if variant:
                state["predicted_hf_url"] = (
                    f"https://huggingface.co/{state['hf_org']}/"
                    f"{basename}-{state['format']}-{variant}bpw"
                )
            else:
                state["predicted_hf_url"] = (
                    f"https://huggingface.co/{state['hf_org']}/"
                    f"{basename}-{state['format']}"
                )
        else:
            state["predicted_hf_url"] = ""
    else:
        state["source_hf_url"] = ""
        state["predicted_hf_url"] = ""

    # Severity tally
    sev_count = {"info": 0, "ok": 0, "warn": 0, "bad": 0}
    for e in events:
        sev_count[e.severity] = sev_count.get(e.severity, 0) + 1
    stats["sev_count"] = sev_count

    raw_tail = [ln for ln in lines if ln.strip()][-tail_n:]
    return ParseResult(events=events, state=state, stats=stats, raw_tail=raw_tail)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="BLOCKQUANT // LIVE OPS")


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _payload(path: Path, tail_n: int = 60) -> dict:
    text = _read_log(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = None
    parsed = parse_log(text, tail_n=tail_n, file_mtime=mtime)
    # Only the most recent 25 typed events to keep payload light.
    visible_events = [
        {"kind": e.kind, "severity": e.severity, "message": e.message,
         "line_no": e.line_no, "fields": e.fields}
        for e in parsed.events[-25:]
    ]
    return {
        "session_id": SESSION_ID,
        "state": parsed.state,
        "stats": parsed.stats,
        "events": visible_events,
        "tail": parsed.raw_tail,
    }


_balance_cache: dict = {"value": None, "fetched_at": 0.0}


def _runpod_balance() -> dict | None:
    """Fetch RunPod credit balance + current burn rate.
    Cached for 60 s so we don't hammer the GraphQL endpoint on every
    SSE tick. Returns ``None`` if unfetchable so the UI shows '—'."""
    if time.time() - _balance_cache["fetched_at"] < 60:
        return _balance_cache["value"]
    try:
        import os as _os
        api_key = _os.environ.get("RUNPOD_API_KEY", "")
        if not api_key:
            # Lazy dotenv load — only needed if launched without the
            # env already populated (rare, but defensive).
            try:
                from dotenv import load_dotenv as _ld
                _ld(REPO_ROOT / ".env", override=False)
                api_key = _os.environ.get("RUNPOD_API_KEY", "")
            except Exception:
                pass
        if not api_key:
            _balance_cache["value"] = None
        else:
            import runpod as _rp
            from runpod.api.graphql import run_graphql_query
            _rp.api_key = api_key
            q = "query myself { myself { clientBalance currentSpendPerHr } }"
            r = run_graphql_query(q)
            me = ((r or {}).get("data", {}) or {}).get("myself", {}) or {}
            _balance_cache["value"] = {
                "balance": me.get("clientBalance"),
                "spend_per_hr": me.get("currentSpendPerHr"),
            }
    except Exception as e:
        _balance_cache["value"] = {"error": str(e)[:120]}
    _balance_cache["fetched_at"] = time.time()
    return _balance_cache["value"]


def _multi_payload(tail_n: int = 60) -> dict:
    """Build a payload covering every known job. Single-job mode returns a
    shape that's a strict superset of the single-payload format (state/stats/
    events/tail mirror the default job), so old clients keep working."""
    jobs_map = _discover_logs() or {_derive_job_id(DEFAULT_LOG): DEFAULT_LOG}
    jobs: dict[str, dict] = {}
    mtimes: dict[str, float] = {}
    for jid, path in jobs_map.items():
        jobs[jid] = _payload(path, tail_n=tail_n)
        try:
            mtimes[jid] = path.stat().st_mtime
        except FileNotFoundError:
            mtimes[jid] = 0.0
    # Default = the most recently-active job. Falls back to first key when
    # nothing has been written yet.
    default_id = (max(mtimes, key=mtimes.get) if mtimes else next(iter(jobs)))
    base = jobs[default_id]
    return {
        "session_id": SESSION_ID,
        "state": base["state"],
        "stats": base["stats"],
        "events": base["events"],
        "tail": base["tail"],
        "jobs": jobs,
        "default_job_id": default_id,
        "job_mtimes": mtimes,
        "runpod_balance": _runpod_balance(),
    }


@app.get("/api/payload")
async def api_payload(log: str = "", tail: int = 60):
    if log:
        # Explicit single-log request — preserve old single-payload shape.
        return JSONResponse(_payload(Path(log), tail_n=tail))
    return JSONResponse(_multi_payload(tail_n=tail))


@app.get("/stream")
async def stream(request: Request, log: str = ""):
    explicit_path = Path(log) if log else None

    async def gen():
        while True:
            if await request.is_disconnected():
                break
            # Always send the full payload — the server-derived runtime/spend
            # values change every second based on file mtime, and bandwidth here
            # is trivial (~5 KB/event × N jobs).
            payload = (_payload(explicit_path) if explicit_path
                       else _multi_payload())
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# HTML — operator console aesthetic
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BlockQuant Field Journal — __SESSION__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..900,0..100;1,9..144,300..900,0..100&family=Source+Serif+4:ital,opsz,wght@0,8..60,300..700;1,8..60,300..700&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    /* Warm charcoal — newsprint at dusk, not midnight cyberpunk. */
    --paper:        #14120e;
    --paper-warm:   #1a1813;
    --paper-raised: #211e18;
    --paper-shadow: #0c0a07;

    /* Quiet hairline rules. No accent rules. */
    --rule:         #2a2620;
    --rule-soft:    #201d18;

    /* One accent, used very sparingly. Slightly muted from neon. */
    --vermillion:   #c63010;
    --vermillion-d: #8a1808;

    /* Settled / quantized tone. */
    --sage:         #6b8a76;

    /* Cream type, soft on dark. */
    --ink:          #ece7d5;
    --ink-soft:     #a89e84;
    --ink-mute:     #6b6555;
    --ink-faint:    #3d382c;

    /* Fonts */
    --display:      'Fraunces', Georgia, serif;
    --body:         'Source Serif 4', 'Source Serif Pro', Georgia, serif;
    --mono:         'JetBrains Mono', ui-monospace, Menlo, monospace;
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0; padding: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: var(--body);
    font-size: 14px;
    line-height: 1.65;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    text-rendering: geometricPrecision;
  }

  .page {
    max-width: 1340px;
    margin: 0 auto;
    padding: 32px 56px 56px;
  }

  ::selection { background: var(--ink-faint); color: var(--ink); }

  a { color: var(--ink); text-decoration: underline;
      text-decoration-color: var(--ink-mute);
      text-decoration-thickness: 1px;
      text-underline-offset: 3px;
      transition: text-decoration-color .15s; }
  a:hover { text-decoration-color: var(--vermillion); }

  /* ===== MASTHEAD — narrow nameplate ===== */
  .masthead {
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    padding: 12px 0 10px;
    margin-bottom: 24px;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: baseline;
    gap: 24px;
  }
  .masthead-left, .masthead-right {
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--ink-mute);
    line-height: 1.5;
  }
  .masthead-left b, .masthead-right b {
    color: var(--ink-soft); font-weight: 500;
  }
  .masthead-right { text-align: right; }
  .masthead-title {
    font-family: var(--display);
    font-weight: 500;
    font-size: 30px;
    line-height: 1;
    letter-spacing: -0.01em;
    color: var(--ink);
    font-variation-settings: "opsz" 60, "SOFT" 0;
    margin: 0;
    text-align: center;
  }
  .masthead-title em {
    font-style: italic;
    font-weight: 400;
    color: var(--vermillion);
    font-variation-settings: "opsz" 60, "SOFT" 30;
  }
  .balance-chip {
    display: inline-block;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: var(--ink);
    background: var(--paper-warm);
    border: 1px solid var(--rule);
    border-radius: 2px;
    padding: 2px 8px;
    margin-bottom: 2px;
  }
  .balance-chip.warn { color: var(--vermillion); border-color: var(--vermillion-d); }
  .balance-burn {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--ink-mute);
    margin-left: 6px;
    letter-spacing: 0.04em;
  }
  .balance-burn.live { color: var(--sage); }

  /* ===== FLEET STRIP — top-of-page summary across N jobs ===== */
  .fleet-strip {
    display: flex;
    align-items: stretch;
    gap: 12px;
    padding: 0 0 18px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--rule);
    overflow-x: auto;
  }
  .fleet-strip-label {
    flex: 0 0 auto;
    font-family: var(--mono);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: var(--ink-mute);
    padding: 14px 12px 0 0;
    border-right: 1px solid var(--rule-soft);
    align-self: stretch;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    text-align: center;
  }
  .fleet-strip-items {
    display: flex;
    gap: 10px;
    flex: 1;
  }
  .fleet-item {
    flex: 0 0 220px;
    background: var(--paper-warm);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--rule);
    padding: 10px 12px 8px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, transform 0.12s;
    position: relative;
    font-family: var(--mono);
  }
  .fleet-item:hover {
    background: var(--paper-raised);
    border-color: var(--ink-mute);
  }
  .fleet-item.selected {
    background: var(--paper-raised);
    border-left-color: var(--vermillion);
    border-color: var(--ink-mute);
  }
  .fleet-item.stale {
    opacity: 0.55;
  }
  .fleet-item.complete {
    border-left-color: var(--sage);
  }
  .fleet-item.failed {
    border-left-color: var(--vermillion-d);
  }
  .fleet-item-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
  }
  .fleet-item-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: 0.04em;
  }
  .fleet-item-pct {
    font-size: 11px;
    font-weight: 500;
    color: var(--ink-soft);
    font-variant-numeric: tabular-nums;
  }
  .fleet-item-pips {
    display: inline-flex;
    gap: 4px;
    margin-top: 6px;
  }
  .fleet-item-pip {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--rule);
    transition: background 0.3s ease;
  }
  .fleet-item-pip.active {
    background: var(--vermillion);
    box-shadow: 0 0 6px rgba(198, 48, 16, 0.6);
  }
  .fleet-item-pip.done {
    background: var(--sage);
  }
  .fleet-item-pip.bad {
    background: var(--vermillion-d);
  }
  .fleet-item-meta {
    margin-top: 6px;
    font-size: 10px;
    color: var(--ink-mute);
    letter-spacing: 0.04em;
    display: flex;
    justify-content: space-between;
    gap: 8px;
  }
  .fleet-item-bar {
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 2px;
    background: var(--rule-soft);
    overflow: hidden;
  }
  .fleet-item-bar > span {
    display: block;
    height: 100%;
    background: var(--vermillion);
    width: 0%;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .fleet-item.complete .fleet-item-bar > span {
    background: var(--sage);
  }

  /* ===== HERO BAND ===== */
  .hero {
    margin-bottom: 28px;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--rule);
  }
  .hero[hidden] { display: none; }
  .hero-title {
    font-family: var(--display);
    font-weight: 500;
    font-size: 32px;
    line-height: 1.2;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin: 0 0 6px;
    font-variation-settings: "opsz" 60, "SOFT" 0;
    word-break: break-word;
  }
  .hero-title a { color: inherit; text-decoration: none; }
  .hero-arrow {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink-soft);
    line-height: 1.6;
    margin: 4px 0 14px;
    word-break: break-all;
  }
  .hero-arrow .glyph { color: var(--ink-mute); margin-right: 6px; }
  .hero-arrow a {
    color: var(--ink-soft);
    text-decoration-color: var(--ink-faint);
  }
  .badge {
    display: inline-block;
    margin-left: 8px;
    padding: 0;
    border: none;
    color: var(--ink-mute);
    font-family: var(--mono);
    font-size: 10px;
    font-style: normal;
    text-transform: uppercase;
    letter-spacing: 0.18em;
  }
  .badge::before { content: "· "; }
  .badge.live { color: var(--vermillion); }
  .stage-line {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    display: flex; gap: 22px; flex-wrap: wrap;
  }
  .stage-line .pip {
    display: inline-flex; align-items: baseline; gap: 6px;
  }
  .stage-line .pip .mark {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ink-mute);
  }
  .stage-line .pip.done { color: var(--ink-soft); }
  .stage-line .pip.done .mark { color: var(--sage); }
  .stage-line .pip.active { color: var(--ink); }
  .stage-line .pip.active .mark { color: var(--vermillion); }
  .stage-line .pip.bad { color: var(--vermillion); }

  /* ===== FIGURE LABELS — quiet hairline rules ===== */
  .figure-label {
    font-family: var(--display);
    font-style: italic;
    font-size: 11px;
    font-weight: 500;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid var(--rule-cyan);
    padding-bottom: 6px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .figure-label .num {
    font-family: var(--display);
    font-style: normal;
    font-weight: 600;
    font-variant: small-caps;
    letter-spacing: 0.12em;
    color: var(--ink-soft);
  }
  .figure-label .meta {
    font-family: var(--mono);
    font-style: normal;
    font-size: 10px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.16em;
  }

  /* ===== MAIN GRID ===== */
  .grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 40px;
    margin-bottom: 32px;
  }

  /* ===== STATS COLUMN (left) ===== */
  .stats-col { display: flex; flex-direction: column; gap: 26px; }

  .runtime-spend {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--rule-cyan);
  }
  .runtime-spend .stat-label {
    font-family: var(--mono);
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: var(--ink-mute);
    margin-bottom: 4px;
  }
  .runtime-spend .stat-value {
    font-family: var(--display);
    font-size: 38px;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.02em;
    line-height: 1;
    font-variation-settings: "opsz" 60, "SOFT" 0;
    font-variant-numeric: lining-nums tabular-nums;
  }
  .runtime-spend .spend { color: var(--ink); }

  /* Instrument tag — tight 4-line mono block, no decoration */
  .instrument-tag {
    background: var(--paper-warm);
    border: 1px solid var(--rule);
    padding: 14px 16px;
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 6px 14px;
    font-family: var(--mono);
    font-size: 11.5px;
    line-height: 1.6;
  }
  .instrument-tag .k {
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 9.5px;
    align-self: baseline;
    padding-top: 2px;
  }
  .instrument-tag .v {
    color: var(--ink);
    word-break: break-all;
  }
  .instrument-tag .v.dim { color: var(--ink-mute); font-style: italic; }

  /* ===== EXPERT MATRIX — quiet inset (centerpiece) ===== */
  .matrix-wrap {
    background: var(--paper-warm);
    border: 1px solid var(--rule);
    padding: 22px;
  }
  .matrix-viewport {
    background: var(--paper-shadow);
    border: 1px solid var(--rule-soft);
    padding: 8px;
  }
  .matrix-svg {
    display: block;
    width: 100%;
    height: auto;
    image-rendering: pixelated;
  }
  /* Smooth fade between cell colours instead of instant flips. */
  .matrix-svg rect {
    transition: fill 350ms ease-out;
  }
  /* Tactile easing on the progress bar fill. */
  .progress-fill {
    transition: width 600ms cubic-bezier(0.4, 0, 0.2, 1) !important;
  }
  /* Subtle pulse when a counter just bumped (added/removed by JS tween) */
  .counter-bumping {
    text-shadow: 0 0 12px rgba(255, 69, 37, 0.35);
    transition: text-shadow 200ms ease-out;
  }

  .matrix-caption {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-top: 14px;
    display: flex; flex-wrap: wrap; gap: 22px; align-items: center;
  }
  .matrix-caption .key {
    display: inline-flex; align-items: center; gap: 8px;
  }
  .matrix-caption .swatch {
    width: 10px; height: 10px;
    display: inline-block;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  /* Inline single-line stats under matrix */
  .matrix-stats {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid var(--rule);
    display: flex;
    flex-wrap: wrap;
    gap: 24px 32px;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.5;
  }
  .matrix-stats .ms-stat {
    display: flex; flex-direction: column; gap: 2px;
  }
  .matrix-stats .ms-label {
    font-size: 9.5px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.2em;
  }
  .matrix-stats .ms-val {
    font-family: var(--display);
    font-size: 22px;
    font-weight: 500;
    color: var(--ink);
    line-height: 1.05;
    letter-spacing: -0.01em;
    font-variation-settings: "opsz" 36, "SOFT" 0;
    font-variant-numeric: lining-nums tabular-nums;
  }
  .matrix-stats .ms-val.accent { color: var(--vermillion); }
  .matrix-stats .ms-val.mono {
    font-family: var(--mono);
    font-size: 14px;
  }
  .matrix-stats .ms-val.dim { color: var(--ink-mute); font-style: italic; font-weight: 400; }

  /* ===== LEDGER — readable on dark ===== */
  .ledger { margin-top: 40px; }
  .ledger-wrap {
    max-height: 320px;
    overflow-y: auto;
    border: 1px solid var(--rule);
    background: var(--paper-warm);
    padding: 4px 0;
    scrollbar-width: thin;
    scrollbar-color: var(--rule) var(--paper-warm);
  }
  .ledger-wrap::-webkit-scrollbar { width: 8px; }
  .ledger-wrap::-webkit-scrollbar-track { background: var(--paper-warm); }
  .ledger-wrap::-webkit-scrollbar-thumb { background: var(--rule); }

  .ledger-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.65;
  }
  .ledger-table thead th {
    font-family: var(--display);
    font-style: italic;
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    text-align: left;
    color: var(--ink-soft);
    border-bottom: 1px solid var(--rule);
    padding: 8px 16px;
    background: var(--paper);
    position: sticky; top: 0;
  }
  .ledger-table tbody tr {
    border-bottom: 1px solid var(--rule-soft);
  }
  .ledger-table tbody tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.012);
  }
  .ledger-table tbody tr:hover {
    background: var(--cyan-soft);
  }
  .ledger-table td {
    padding: 6px 16px;
    vertical-align: baseline;
    color: var(--ink);
  }
  .ledger-table .col-no {
    color: var(--ink-mute);
    font-variant-numeric: tabular-nums;
    width: 56px;
  }
  .ledger-table .col-time {
    width: 88px;
    color: var(--ink-mute);
    font-variant-numeric: tabular-nums;
  }
  .ledger-table .col-kind {
    width: 140px;
    font-family: var(--display);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.14em;
    color: var(--ink);
  }
  .ledger-table .col-kind.severity-bad  { color: var(--vermillion); font-style: italic; }
  .ledger-table .col-kind.severity-warn { color: #f0c060; font-style: italic; }
  .ledger-table .col-kind.severity-ok   { color: var(--sage); }
  .ledger-table .col-msg { color: var(--ink); }
  .ledger-table .field {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--cyan);
    margin-right: 12px;
  }
  .ledger-table .field-key { color: var(--ink-mute); }

  /* ===== OUTPUT / ALERTS — compact strip ===== */
  .output-section {
    margin-top: 36px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
  }
  .output-pending {
    font-family: var(--display);
    font-style: italic;
    color: var(--ink-mute);
    font-size: 15px;
  }
  #hf-out a { font-family: var(--mono); font-size: 12.5px; }
  .alert-clear {
    font-family: var(--display);
    font-style: italic;
    color: var(--sage);
    font-size: 15px;
  }
  .alert {
    font-family: var(--mono);
    font-size: 11px;
    color: #ff7f6a;
    border-left: 2px solid var(--vermillion);
    padding: 6px 10px;
    margin-bottom: 4px;
    background: rgba(255, 69, 37, 0.06);
    word-break: break-word;
  }

  /* ===== COLOPHON ===== */
  .colophon {
    margin-top: 48px;
    padding-top: 18px;
    border-top: 1px solid var(--rule);
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 24px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    align-items: center;
  }
  .colophon-mark {
    font-family: var(--display);
    font-style: italic;
    font-weight: 400;
    font-size: 13px;
    text-transform: none;
    letter-spacing: 0;
    color: var(--ink-soft);
  }
  .colophon .mid { text-align: center; }
  .colophon .signal-pip {
    display: inline-block; width: 5px; height: 5px;
    background: var(--ink-soft); border-radius: 50%;
    vertical-align: 1px; margin: 0 6px;
  }

  /* responsive */
  @media (max-width: 980px) {
    .grid { grid-template-columns: 1fr; }
    .page { padding: 24px; }
    .hero-title { font-size: 26px; }
    .output-section { grid-template-columns: 1fr; }
    .ledger-table .col-kind { width: 110px; }
    .runtime-spend .stat-value { font-size: 30px; }
  }
</style>
</head>
<body>
<div class="page">

  <header class="masthead">
    <div class="masthead-left">
      <b id="masthead-date">— — —</b><br>
      Operator <b>blockblockblock</b>
    </div>
    <h1 class="masthead-title">Block<em>Quant</em></h1>
    <div class="masthead-right">
      <span id="rp-balance" class="balance-chip" title="RunPod credit balance">Balance —</span>
      <span id="rp-burn" class="balance-burn"></span><br>
      Vol. I · No. <b><span id="iter-count">—</span></b> &nbsp;·&nbsp;
      <span id="session-id-short">SESSION __SESSION__</span>
    </div>
  </header>

  <!-- FLEET STRIP — only shown when 2+ jobs are tracked -->
  <nav class="fleet-strip" id="fleet-strip" hidden>
    <div class="fleet-strip-label">Fleet</div>
    <div class="fleet-strip-items" id="fleet-strip-items"></div>
  </nav>

  <!-- HERO — model + URLs + one-line stage indicator -->
  <section class="hero" id="hero" hidden>
    <h2 class="hero-title">
      <a id="hero-model" href="#" target="_blank"></a>
    </h2>
    <div class="hero-arrow" id="hero-arrow" hidden>
      <span class="glyph">→</span>
      <a id="hero-output" href="#" target="_blank"></a>
      <span id="hero-output-badge" class="badge"></span>
    </div>
    <div class="stage-line" id="stage-line">
      <span class="pip" id="st-bootstrap"><span class="mark">○</span><span>bootstrap</span></span>
      <span class="pip" id="st-download"><span class="mark">○</span><span>download</span></span>
      <span class="pip" id="st-quantize"><span class="mark">○</span><span>quantize</span></span>
      <span class="pip" id="st-upload"><span class="mark">○</span><span>upload</span></span>
    </div>
  </section>

  <div class="grid">

    <!-- LEFT: compact sidebar -->
    <aside class="stats-col">

      <div class="runtime-spend">
        <div>
          <div class="stat-label">Runtime</div>
          <div id="pod-runtime" class="stat-value">—</div>
        </div>
        <div>
          <div class="stat-label">Spend</div>
          <div id="pod-spent" class="stat-value spend">$—</div>
        </div>
      </div>

      <div class="instrument-tag" aria-label="Compute node">
        <span class="k">Pod</span>     <span id="pod-id" class="v dim">awaiting</span>
        <span class="k">GPU</span>     <span id="pod-gpu" class="v dim">—</span>
        <span class="k">Stack</span>   <span id="pod-interp" class="v dim">—</span>
        <span class="k">Rate</span>    <span id="pod-rate" class="v dim">—</span>
      </div>

      <div>
        <div class="figure-label">
          <span class="num">Exhibit A.</span>
          <span class="meta">Published Output</span>
        </div>
        <div id="hf-out"><span class="output-pending">— awaiting upload —</span></div>
      </div>

      <div>
        <div class="figure-label">
          <span class="num">Exhibit B.</span>
          <span class="meta" id="sev-meta">no alerts</span>
        </div>
        <div id="errs"><span class="alert-clear">no anomalies recorded</span></div>
      </div>

    </aside>

    <!-- RIGHT: matrix + inline stats -->
    <main>
      <div class="figure-label">
        <span class="num">Figure I.</span>
        <span class="meta">Expert Matrix · Live</span>
      </div>

      <div class="matrix-wrap">
        <div class="matrix-viewport">
          <svg id="matrix" class="matrix-svg" viewBox="0 0 1024 192" preserveAspectRatio="xMidYMid meet"></svg>
        </div>

        <div class="matrix-caption">
          <span class="key"><span class="swatch" style="background:#26221c"></span>pending</span>
          <span class="key"><span class="swatch" style="background:var(--vermillion)"></span>measuring</span>
          <span class="key"><span class="swatch" style="background:var(--sage)"></span>quantized</span>
          <span id="matrix-shape" style="margin-left:auto; font-size:10.5px; color:var(--ink-mute);">— layers × — experts</span>
        </div>

        <div class="matrix-stats">
          <div class="ms-stat">
            <span class="ms-label">Layer</span>
            <span class="ms-val"><span id="layer">—</span> <span class="dim" id="layer-total" style="font-size:14px;color:var(--ink-mute);">/ —</span></span>
          </div>
          <div class="ms-stat">
            <span class="ms-label">Expert</span>
            <span id="expert" class="ms-val">—</span>
          </div>
          <div class="ms-stat">
            <span class="ms-label">Tensors</span>
            <span id="tensors" class="ms-val">0</span>
          </div>
          <div class="ms-stat">
            <span class="ms-label">Tensor Rate</span>
            <span id="tensor-rate" class="ms-val">— /s</span>
          </div>
          <div class="ms-stat">
            <span class="ms-label">Sec / Layer</span>
            <span id="layer-avg" class="ms-val">—</span>
          </div>
          <div class="ms-stat" style="margin-left:auto;">
            <span class="ms-label">Time Remaining</span>
            <span id="eta" class="ms-val" style="font-family:var(--mono); font-size:15px;">computing</span>
          </div>
          <div class="ms-stat">
            <span class="ms-label">Progress</span>
            <span id="bar-pct" class="ms-val">0.0%</span>
          </div>
        </div>
      </div>
    </main>
  </div>

  <!-- LEDGER -->
  <section class="ledger">
    <div class="figure-label">
      <span class="num">Ledger.</span>
      <span class="meta" id="events-meta">— entries</span>
    </div>
    <div class="ledger-wrap">
      <table class="ledger-table">
        <thead>
          <tr>
            <th class="col-no">№</th>
            <th class="col-time">Time</th>
            <th class="col-kind">Kind</th>
            <th class="col-msg">Description</th>
          </tr>
        </thead>
        <tbody id="events">
          <tr><td colspan="4" style="padding:12px 16px;font-style:italic;color:var(--ink-mute);">awaiting first event…</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <footer class="colophon">
    <span>Set in <em class="colophon-mark">Fraunces</em> &amp; <em class="colophon-mark">JetBrains Mono</em></span>
    <span class="mid" id="logsrc">streaming default log</span>
    <span>signal <span class="signal-pip"></span></span>
  </footer>
</div>

<script>
const $ = id => document.getElementById(id);
const pad = n => n.toString().padStart(2, "0");

function fmtDuration(secs) {
  if (!secs || secs < 0) return "—";
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}h ${pad(m)}m`;
  return `${m}m ${pad(s)}s`;
}

function fmtMonth() {
  const d = new Date();
  return d.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
}
$("masthead-date").textContent = fmtMonth();

const params = new URLSearchParams(location.search);
const logQs = params.get("log") ? `?log=${encodeURIComponent(params.get("log"))}` : "";
$("logsrc").textContent = params.get("log") ? `streaming ${params.get("log")}` : "streaming default log";

// Manual overrides for jobs whose log lacks the [job] header line — pass
// ?model=org/repo&variants=4.5&hf_org=youraccount in the URL.
const OVERRIDE_MODEL    = params.get("model") || "";
const OVERRIDE_VARIANTS = params.get("variants") || "";
const OVERRIDE_HFORG    = params.get("hf_org") || "";

// Architecture defaults. Either the log tells us (max_layer_observed +
// max_expert_observed in the payload), or the URL overrides them, or
// these reasonable Qwen3-MoE defaults stand in until we have data.
let TOTAL_LAYERS = parseInt(params.get("layers") || "48", 10);
let EXPERTS_PER_LAYER = parseInt(params.get("experts") || "256", 10);
const LAYERS_FROM_URL = params.has("layers");
const EXPERTS_FROM_URL = params.has("experts");

// ===== Build expert matrix (rebuildable) =====
// Auto-resizes when the log reveals a model with different layer / expert
// counts than the defaults. URL params (?layers=N&experts=N) lock the size.
const matrixSvg = $("matrix");
const CELL = 4, GAP = 0;
let cells = [];

function buildMatrix(layers, experts) {
  TOTAL_LAYERS = layers;
  EXPERTS_PER_LAYER = experts;
  const W = experts * CELL + Math.max(0, experts - 1) * GAP;
  const H = layers * CELL + Math.max(0, layers - 1) * GAP;
  matrixSvg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  while (matrixSvg.firstChild) matrixSvg.removeChild(matrixSvg.firstChild);
  cells = [];
  for (let layer = 0; layer < layers; layer++) {
    const row = [];
    for (let exp = 0; exp < experts; exp++) {
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", exp * (CELL + GAP));
      r.setAttribute("y", layer * (CELL + GAP));
      r.setAttribute("width", CELL);
      r.setAttribute("height", CELL);
      r.setAttribute("fill", "#1a1a24");
      matrixSvg.appendChild(r);
      row.push(r);
    }
    cells.push(row);
  }
}
buildMatrix(TOTAL_LAYERS, EXPERTS_PER_LAYER);

function maybeResize(maxLayer, maxExpert) {
  // Grow only — never shrink mid-run. Round expert count to a sane power-of-2-ish
  // bucket (Qwen MoE uses 128/256/384/512); for layers, use observed + 1 since
  // we want the just-discovered last layer to fit.
  if (LAYERS_FROM_URL && EXPERTS_FROM_URL) return;
  const targetLayers = LAYERS_FROM_URL
    ? TOTAL_LAYERS
    : Math.max(TOTAL_LAYERS, (maxLayer || 0) + 1);
  const expertBuckets = [64, 128, 160, 256, 384, 512, 1024];
  let targetExperts = EXPERTS_PER_LAYER;
  if (!EXPERTS_FROM_URL && maxExpert != null) {
    targetExperts = expertBuckets.find(b => b > maxExpert) || ((maxExpert || 0) + 1);
    targetExperts = Math.max(targetExperts, EXPERTS_PER_LAYER);
  }
  if (targetLayers !== TOTAL_LAYERS || targetExperts !== EXPERTS_PER_LAYER) {
    buildMatrix(targetLayers, targetExperts);
  }
}

const COLORS = {
  pending: "#1a1a24",
  active:  "#ff4525",
  done:    "#7faf94",
};

// Repaint the matrix in row-stagger via rAF so the browser yields between
// transformer-block rows. With CSS `transition: fill` on each rect the
// effect is a smooth row-by-row settle, not a flash of all 12k cells.
let _paintToken = 0;
function paintMatrix(currentLayer, currentExpert) {
  const myToken = ++_paintToken;
  let L = 0;
  function paintRow() {
    if (myToken !== _paintToken) return;   // superseded by a newer paint
    if (L >= TOTAL_LAYERS) return;
    for (let E = 0; E < EXPERTS_PER_LAYER; E++) {
      let color;
      if (currentLayer == null) color = COLORS.pending;
      else if (L < currentLayer) color = COLORS.done;
      else if (L === currentLayer) {
        if (currentExpert != null && E < currentExpert) color = COLORS.done;
        else if (currentExpert != null && E === currentExpert) color = COLORS.active;
        else color = COLORS.pending;
      } else color = COLORS.pending;
      const cell = cells[L][E];
      if (cell.getAttribute("fill") !== color) cell.setAttribute("fill", color);
    }
    L++;
    requestAnimationFrame(paintRow);
  }
  requestAnimationFrame(paintRow);
}

function setStage(id, state) {
  const el = $(id);
  if (!el) return;
  el.classList.remove("active", "done", "bad");
  if (state) el.classList.add(state);
  // Swap mark glyph based on state — done=✓, active=●, bad=✕, pending=○
  const mark = el.querySelector(".mark");
  if (mark) {
    mark.textContent =
      state === "done" ? "✓" :
      state === "active" ? "●" :
      state === "bad" ? "✕" : "○";
  }
}

// ===== Smooth numeric counter tween =====
// Interpolates element textContent from oldValue → newValue over `duration`
// using ease-out-cubic. Stores the live value on the element so successive
// updates pick up where the previous tween left off.
const _activeTweens = new Map();    // elementId → rAF id
function tweenNumber(elId, toValue, duration = 1500, formatter = null) {
  const el = $(elId);
  if (!el) return;
  const fromValue = parseFloat(el.dataset.tweenValue ?? "NaN");
  const target = Number(toValue);
  if (!Number.isFinite(target)) return;
  // First-ever set OR no change: just write it directly.
  if (!Number.isFinite(fromValue) || fromValue === target) {
    el.dataset.tweenValue = String(target);
    el.textContent = formatter ? formatter(target) : target.toLocaleString();
    return;
  }
  // Cancel any in-flight tween on this element.
  if (_activeTweens.has(elId)) cancelAnimationFrame(_activeTweens.get(elId));
  const t0 = performance.now();
  const delta = target - fromValue;
  const ease = t => 1 - Math.pow(1 - t, 3);   // ease-out-cubic
  el.classList.add("counter-bumping");
  function step(now) {
    const t = Math.min(1, (now - t0) / duration);
    const v = fromValue + delta * ease(t);
    el.textContent = formatter ? formatter(v) : Math.round(v).toLocaleString();
    if (t < 1) {
      _activeTweens.set(elId, requestAnimationFrame(step));
    } else {
      el.dataset.tweenValue = String(target);
      el.textContent = formatter ? formatter(target) : target.toLocaleString();
      _activeTweens.delete(elId);
      setTimeout(() => el.classList.remove("counter-bumping"), 200);
    }
  }
  _activeTweens.set(elId, requestAnimationFrame(step));
}
const fmtPct = v => v.toFixed(1) + "%";
const fmtDur = v => fmtDuration(v);
const fmtDollar = v => "$" + v.toFixed(2);

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

let lastSeenEvents = 0;
const ROMAN = ["i","ii","iii","iv","v","vi","vii","viii","ix","x"];

function renderEvent(e, idx) {
  const fields = Object.entries(e.fields || {})
    .filter(([k, v]) => v != null && v !== "")
    .slice(0, 3)
    .map(([k, v]) => `<span class="field"><span class="field-key">${escapeHtml(k)}</span>=${escapeHtml(String(v))}</span>`)
    .join("");
  // Try to extract a HH:MM:SS from the raw message
  const tm = (e.message.match(/\b(\d{2}:\d{2}:\d{2})\b/) || [])[1] || "—";
  const sevClass = `severity-${e.severity}`;
  return `<tr>
    <td class="col-no">${(idx + 1).toString().padStart(3, "0")}</td>
    <td class="col-time">${tm}</td>
    <td class="col-kind ${sevClass}">${e.kind}</td>
    <td class="col-msg">${fields || escapeHtml(e.message)}</td>
  </tr>`;
}

// Multi-job selection state — survives across SSE ticks. URL ?job=ID wins.
let selectedJobId = (params.get("job") || "").trim() || null;
let lastRenderedJobId = null;
// Per-tick cache of the entire payload so click handlers can pull the
// freshest snapshot for the tile they're switching to (instead of the
// stale closure they captured at element-creation time).
let latestPayload = null;

function resetDetailPanel() {
  // Wipe ONLY the fields that genuinely belong to the prior job.
  // Hero, pod-info, runtime/spend etc. are re-set unconditionally by
  // renderDetail when the new payload has the data, so leaving them
  // alone here means an init tile renders what it knows immediately
  // (model name, predicted URL, pod-id) rather than blinking through
  // a "—" frame.
  $("hf-out").innerHTML = `<span class="output-pending">— awaiting upload —</span>`;
  ["layer", "expert", "tensors"].forEach(id => {
    const el = $(id); if (el) el.textContent = "—";
  });
  $("eta").textContent = "—";
  $("eta").style.color = "";
  $("bar-pct").textContent = "0%";
  ["st-bootstrap", "st-download", "st-quantize", "st-upload"].forEach(id => {
    const el = $(id);
    if (el) el.classList.remove("active", "done", "bad");
  });
  $("errs").innerHTML = "";
  $("events").innerHTML = "";
  $("events-meta").textContent = "0 entries";
  $("sev-meta").textContent = "no alerts";
  lastSeenEvents = 0;
}

function deriveLabel(jobId, jobPayload) {
  const ms = jobPayload && jobPayload.state ? jobPayload.state.model_id : "";
  const variants = jobPayload && jobPayload.state ? jobPayload.state.variants : "";
  if (variants) {
    const base = ms ? ms.split("/").pop().slice(0, 14) : jobId;
    return `${base} · ${variants} bpw`;
  }
  return jobId;
}

function jobStageClass(s) {
  if (!s) return null;
  if (s.terminated && !s.hf_url) return "failed";
  if (s.stage === "complete" || s.hf_url) return "complete";
  return null;
}

function jobPctOf(s) {
  if (!s) return 0;
  if (s.stage === "complete" || s.hf_url) return 100;
  // Heuristic: bootstrap=5, download=15, quantize=15→85 by layer, upload=85→99.
  if (s.upload_started) return 92;
  if (s.current_layer && s.max_layer_observed) {
    const total = Math.max(s.max_layer_observed + 1, 48);
    const ratio = Math.min(s.current_layer / total, 0.95);
    return Math.round(15 + ratio * 70);
  }
  if (s.download_done) return 18;
  if (s.download_started) return 8;
  if (s.bootstrap_complete_at) return 5;
  return 1;
}

function jobETA(s, st) {
  if (!s) return "";
  if (s.stage === "complete" || s.hf_url) return "complete";
  if (s.eta) return s.eta;
  if (s.upload_started) return "uploading";
  if (s.download_started && !s.download_done) return "downloading";
  if (s.bootstrap_complete_at) return "starting";
  return "init";
}

function renderFleetStrip(p) {
  const jobs = p.jobs || null;
  const strip = $("fleet-strip");
  // Only show strip when there are 2+ jobs.
  if (!jobs || Object.keys(jobs).length < 2) {
    strip.hidden = true;
    return p;  // pass through default state for detail render
  }
  strip.hidden = false;

  const ids = Object.keys(jobs).sort();
  if (!selectedJobId || !jobs[selectedJobId]) {
    selectedJobId = p.default_job_id || ids[0];
  }

  const items = $("fleet-strip-items");
  const existing = new Map(
    Array.from(items.children).map(el => [el.dataset.jobId, el])
  );

  const now = Date.now() / 1000;
  ids.forEach(jid => {
    const jp = jobs[jid];
    const s = jp.state || {};
    const mtime = (p.job_mtimes && p.job_mtimes[jid]) || 0;
    const stale = mtime > 0 && (now - mtime) > 300;
    const stageClass = jobStageClass(s);
    const pct = jobPctOf(s);

    let el = existing.get(jid);
    if (!el) {
      el = document.createElement("div");
      el.className = "fleet-item";
      el.dataset.jobId = jid;
      el.addEventListener("click", () => {
        if (jid === selectedJobId) return;
        selectedJobId = jid;
        const url = new URL(window.location);
        url.searchParams.set("job", jid);
        history.replaceState(null, "", url.toString());
        // Apply selection visually.
        document.querySelectorAll(".fleet-item").forEach(n =>
          n.classList.toggle("selected", n.dataset.jobId === jid));
        // Wipe stale detail fields, then re-render from the FRESHEST
        // snapshot (latestPayload), not the stale closure-captured one.
        resetDetailPanel();
        lastRenderedJobId = jid;
        const fresh = (latestPayload && latestPayload.jobs
                       && latestPayload.jobs[jid]) || jobs[jid];
        renderDetail(fresh);
      });
      el.innerHTML = `
        <div class="fleet-item-row">
          <span class="fleet-item-label"></span>
          <span class="fleet-item-pct"></span>
        </div>
        <div class="fleet-item-pips">
          <span class="fleet-item-pip" data-stage="bootstrap"></span>
          <span class="fleet-item-pip" data-stage="download"></span>
          <span class="fleet-item-pip" data-stage="quantize"></span>
          <span class="fleet-item-pip" data-stage="upload"></span>
        </div>
        <div class="fleet-item-meta">
          <span class="fleet-item-eta"></span>
          <span class="fleet-item-tag"></span>
        </div>
        <div class="fleet-item-bar"><span></span></div>
      `;
      items.appendChild(el);
    }

    el.classList.toggle("selected", jid === selectedJobId);
    el.classList.toggle("stale", stale);
    el.classList.toggle("complete", stageClass === "complete");
    el.classList.toggle("failed", stageClass === "failed");

    el.querySelector(".fleet-item-label").textContent = deriveLabel(jid, jp);
    el.querySelector(".fleet-item-pct").textContent = `${pct}%`;
    el.querySelector(".fleet-item-eta").textContent = jobETA(s);
    el.querySelector(".fleet-item-tag").textContent = jid;
    el.querySelector(".fleet-item-bar > span").style.width = `${pct}%`;

    // Stage pips: bootstrap → download → quantize → upload.
    const pipStates = {
      bootstrap: s.bootstrap_complete_at ? "done"
                 : (s.pod_id ? "active" : null),
      download:  s.download_done ? "done"
                 : (s.download_started ? "active" : null),
      quantize:  (s.upload_started || s.hf_url) ? "done"
                 : (s.tensors_quantized > 0 ? "active" : null),
      upload:    s.hf_url ? "done" : (s.upload_started ? "active" : null),
    };
    // Completion override: a finished run gets all pips green even if
    // the parser missed an earlier "complete" line in an old log file.
    const completed = s.stage === "complete" || !!s.hf_url;
    if (completed) {
      pipStates.bootstrap = "done";
      pipStates.download  = "done";
      pipStates.quantize  = "done";
      pipStates.upload    = "done";
    }
    if (s.terminated && !s.hf_url) pipStates.quantize = "bad";
    el.querySelectorAll(".fleet-item-pip").forEach(pip => {
      const stage = pip.dataset.stage;
      pip.classList.remove("active", "done", "bad");
      if (pipStates[stage]) pip.classList.add(pipStates[stage]);
    });
  });

  // Return the selected job's payload for the detail view to consume.
  return jobs[selectedJobId];
}

function renderRunpodBalance(p) {
  const b = p && p.runpod_balance;
  const chip = $("rp-balance");
  const burn = $("rp-burn");
  if (!chip || !burn) return;
  if (!b || b.error || b.balance == null) {
    chip.textContent = "Balance —";
    chip.classList.remove("warn");
    burn.textContent = b && b.error ? "(api err)" : "";
    burn.classList.remove("live");
    return;
  }
  const bal = Number(b.balance) || 0;
  chip.textContent = `Balance $${bal.toFixed(2)}`;
  chip.classList.toggle("warn", bal < 25);
  const spend = Number(b.spend_per_hr) || 0;
  if (spend > 0.001) {
    burn.textContent = `· $${spend.toFixed(2)}/hr`;
    burn.classList.add("live");
  } else {
    burn.textContent = "· idle";
    burn.classList.remove("live");
  }
}

function update(p) {
  if (p.heartbeat) return;
  latestPayload = p;
  renderRunpodBalance(p);
  // Multi-job mode: render strip and pull the selected job's payload.
  // Single-job mode: strip stays hidden, payload passed through.
  const detailPayload = renderFleetStrip(p) || p;
  // Reset the panel on job switch so we don't inherit the previous job's
  // hf_url, pod info, or events.
  if (selectedJobId !== lastRenderedJobId) {
    resetDetailPanel();
    lastRenderedJobId = selectedJobId;
  }
  renderDetail(detailPayload);
}

function renderDetail(p) {
  if (!p || p.heartbeat) return;
  const s = p.state, st = p.stats || {};

  // Apply URL overrides (for jobs whose log doesn't carry the [job] header)
  if (!s.model_id && OVERRIDE_MODEL) {
    s.model_id = OVERRIDE_MODEL;
    s.source_hf_url = `https://huggingface.co/${OVERRIDE_MODEL}`;
    const basename = OVERRIDE_MODEL.split("/").pop();
    const orgOrUser = OVERRIDE_HFORG || "{your-account}";
    s.predicted_hf_url = `https://huggingface.co/${orgOrUser}/${basename}-${s.format || "exl3"}`;
  }
  if (!s.variants && OVERRIDE_VARIANTS) s.variants = OVERRIDE_VARIANTS;

  // Hero — only render if we actually know the model. No placeholder strings.
  if (s.model_id) {
    $("hero").hidden = false;
    const basename = s.model_id.split("/").pop();
    const heroA = $("hero-model");
    heroA.textContent = basename;
    heroA.href = s.source_hf_url || `https://huggingface.co/${s.model_id}`;
  }
  // Prefer the REAL URL the moment we know it; only fall back to the
  // predicted slug while upload is still in flight.
  const linkedUrl = s.hf_url || s.predicted_hf_url;
  if (linkedUrl) {
    $("hero-arrow").hidden = false;
    const outA = $("hero-output");
    const display = linkedUrl.replace(/^https:\/\/huggingface\.co\//, "");
    outA.textContent = display + (s.variants ? ` · ${s.variants} bpw` : "");
    outA.href = linkedUrl;
    const badge = $("hero-output-badge");
    if (s.hf_url) {
      badge.textContent = "live";
      badge.classList.add("live");
    } else {
      badge.textContent = "pending";
      badge.classList.remove("live");
    }
  }

  // Compact instrument tag
  if (s.pod_id) {
    $("pod-id").textContent = s.pod_id;
    $("pod-id").classList.remove("dim");
  }
  if (s.gpu) {
    $("pod-gpu").textContent = s.gpu;
    $("pod-gpu").classList.remove("dim");
  }
  if (s.remote_py || s.torch_ver) {
    $("pod-interp").textContent = `${s.remote_py || "—"} · torch ${s.torch_ver || "—"}`;
    $("pod-interp").classList.remove("dim");
  }
  if (s.cost_per_hr) {
    $("pod-rate").textContent = `$${s.cost_per_hr.toFixed(2)} / hour`;
    $("pod-rate").classList.remove("dim");
  }

  // Big runtime + spend (server-derived, refresh-safe) — tween smoothly.
  if (s.pod_runtime_sec > 0) {
    tweenNumber("pod-runtime", s.pod_runtime_sec, 1500, fmtDur);
  }
  if (s.pod_spent > 0) {
    tweenNumber("pod-spent", s.pod_spent, 1500, fmtDollar);
  }

  $("iter-count").textContent = s.iteration_count || 1;

  // Stage pips
  setStage("st-bootstrap",
    s.bootstrap_complete_at ? "done" : (s.pod_id ? "active" : null));
  setStage("st-download",
    s.tensors_quantized > 0 || s.download_done ? "done"
    : (s.bootstrap_complete_at ? "active" : null));
  setStage("st-quantize",
    s.upload_started || s.hf_url ? "done"
    : (s.tensors_quantized > 0 ? "active" : null));
  setStage("st-upload",
    s.hf_url ? "done" : (s.upload_started ? "active" : null));
  if (s.terminated && !s.hf_url) {
    setStage("st-quantize", "bad");
  }

  // Auto-resize the matrix when the log reveals a non-default model shape.
  if (s.max_layer_observed || s.max_expert_observed) {
    maybeResize(s.max_layer_observed, s.max_expert_observed);
  }

  // Inline matrix-stats — when the run is over, freeze the live readouts
  // and replace ETA + Time Remaining with the final outcome.
  // The local log can lose the last few layers if the local poller exits
  // before the remote script finishes flushing — when state.hf_url is
  // present we know the run actually completed, so display the truthful
  // total instead of the last-observed layer.
  const done = s.stage === "complete" || !!s.hf_url;

  $("layer-total").textContent = `/ ${TOTAL_LAYERS}`;
  $("matrix-shape").textContent = `${TOTAL_LAYERS} layers × ${EXPERTS_PER_LAYER} experts`;

  if (done) {
    $("layer").textContent = TOTAL_LAYERS;
    $("expert").textContent = "all";
    $("tensors").textContent = "—";
    $("eta").textContent = "done";
    $("eta").style.color = "var(--sage)";
    $("layer-avg").textContent = s.layer_avg_sec > 0 ? fmtDuration(s.layer_avg_sec) : "—";
    $("tensor-rate").textContent = s.tensor_rate_per_sec
      ? `${s.tensor_rate_per_sec.toFixed(1)} /s` : "—";
  } else {
    // Tween counters so they visibly tick up between SSE polls instead
    // of teleporting whenever the syncer drains a fresh batch.
    if (s.current_layer != null) tweenNumber("layer", s.current_layer, 800, v => Math.round(v));
    if (s.current_expert != null) tweenNumber("expert", s.current_expert, 800, v => Math.round(v));
    tweenNumber("tensors", s.tensors_quantized || 0, 1500, v => Math.round(v).toLocaleString());
    $("eta").textContent = s.eta || "computing";
    if (s.tensor_rate_per_sec) {
      tweenNumber("tensor-rate", s.tensor_rate_per_sec, 1000, v => v.toFixed(1) + " /s");
    }
    if (s.layer_avg_sec > 0) {
      tweenNumber("layer-avg", s.layer_avg_sec, 1000, fmtDur);
    }
  }

  // Progress bar — tween the percentage label; the bar fill itself is
  // CSS-transitioned via `.progress-fill { transition: width ...}`.
  const pct = done
    ? 100
    : (s.current_layer != null ? Math.min(100, ((s.current_layer + 1) / TOTAL_LAYERS) * 100) : 0);
  tweenNumber("bar-pct", pct, 800, fmtPct);

  // Matrix: paint as fully done when run is complete so all cells settle to sage.
  if (done) {
    paintMatrix(TOTAL_LAYERS, EXPERTS_PER_LAYER);
  } else {
    paintMatrix(s.current_layer, s.current_expert);
  }

  // Output URL (sidebar — also separate from hero predicted URL)
  if (s.hf_url) {
    $("hf-out").innerHTML = `<a href="${s.hf_url}" target="_blank">${escapeHtml(s.hf_url)}</a>`;
  }

  // Alerts
  const tally = (st.sev_count || {});
  const badCount = tally.bad || 0;
  $("sev-meta").textContent = badCount === 0 ? "no alerts" : `${badCount} alerts`;
  const bad = (p.events || []).filter(e => e.severity === "bad");
  if (bad.length) {
    $("errs").innerHTML = bad.slice(-5).map(e =>
      `<div class="alert">${escapeHtml(e.message)}</div>`).join("");
  }

  // Ledger
  if ((p.events || []).length) {
    const tbody = $("events");
    tbody.innerHTML = p.events.map((e, i) => renderEvent(e, i)).join("");
    $("events-meta").textContent = `${p.events.length} entries`;
    if (p.events.length > lastSeenEvents) {
      const wrap = tbody.closest(".ledger-wrap");
      const nearBottom = wrap.scrollTop + wrap.clientHeight >= wrap.scrollHeight - 60;
      if (nearBottom) wrap.scrollTop = wrap.scrollHeight;
      lastSeenEvents = p.events.length;
    }
  }
}

const evt = new EventSource("/stream" + logQs);
evt.onmessage = e => update(JSON.parse(e.data));
evt.onerror = () => {
  console.warn("stream disconnected — reload to retry");
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML.replace("__SESSION__", SESSION_ID)


def main():
    global DEFAULT_LOG, JOB_LOGS, LOGS_DIR, LOG_PATTERN
    # Windows cp1252 stdout chokes on Unicode (→ · etc.); switch to UTF-8
    # so our startup banner doesn't crash before uvicorn even starts.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--log", nargs="*", default=[],
        help="Log file(s) to tail. Pass multiple paths for fleet view.",
    )
    parser.add_argument(
        "--logs-dir", default="",
        help="Directory to auto-discover logs in (re-scanned each request).",
    )
    parser.add_argument(
        "--log-pattern", default="runpod-*.log",
        help="Glob applied inside --logs-dir.",
    )
    args = parser.parse_args()

    log_paths = [Path(p) for p in args.log] if args.log else []
    if args.logs_dir:
        LOGS_DIR = Path(args.logs_dir)
        LOG_PATTERN = args.log_pattern
    if log_paths:
        JOB_LOGS = {_derive_job_id(p): p for p in log_paths}
        DEFAULT_LOG = log_paths[0]
    elif not LOGS_DIR:
        # No args supplied — keep legacy behaviour.
        JOB_LOGS = {_derive_job_id(DEFAULT_LOG): DEFAULT_LOG}

    discovered = _discover_logs() or {_derive_job_id(DEFAULT_LOG): DEFAULT_LOG}
    print(f"BLOCKQUANT // LIVE OPS  ·  http://{args.host}:{args.port}")
    print(f"Session: {SESSION_ID}")
    if len(discovered) == 1:
        print(f"Tailing: {next(iter(discovered.values()))}")
    else:
        print(f"Fleet mode — {len(discovered)} jobs:")
        for jid, path in discovered.items():
            print(f"  · {jid}  →  {path}")
        if LOGS_DIR:
            print(f"  (auto-discovering {LOG_PATTERN} in {LOGS_DIR})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
