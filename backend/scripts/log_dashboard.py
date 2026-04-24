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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO_ROOT / "backend" / "logs" / "runpod-qwen35b-4.5.log"

SESSION_ID = "-".join(f"{secrets.randbelow(10000):04d}" for _ in range(5))


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

    # Network / SSH errors
    ("NET_RESET", "warn",
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
                state["hf_org"] = fields.get("hf_org", "") or ""
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
            # Filter QUANT_TENSOR out of the visible events stream (would be 50k+).
            if kind not in ("QUANT_TENSOR",):
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
        org_or_user = state["hf_org"] or "{your-account}"
        # Use the first declared variant for the predicted URL.
        variant = (state["variants"].split(",")[0].strip()
                   if state["variants"] else "")
        if variant:
            state["predicted_hf_url"] = (
                f"https://huggingface.co/{org_or_user}/{basename}-{state['format']}-{variant}bpw"
            )
        else:
            state["predicted_hf_url"] = (
                f"https://huggingface.co/{org_or_user}/{basename}-{state['format']}"
            )
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


@app.get("/api/payload")
async def api_payload(log: str = "", tail: int = 60):
    path = Path(log) if log else DEFAULT_LOG
    return JSONResponse(_payload(path, tail_n=tail))


@app.get("/stream")
async def stream(request: Request, log: str = ""):
    path = Path(log) if log else DEFAULT_LOG

    async def gen():
        while True:
            if await request.is_disconnected():
                break
            # Always send the full payload — the server-derived runtime/spend
            # values change every second based on file mtime, and bandwidth here
            # is trivial (~5 KB/event).
            payload = _payload(path)
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
      Vol. I · No. <b><span id="iter-count">—</span></b><br>
      <span id="session-id-short">SESSION __SESSION__</span>
    </div>
  </header>

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

function paintMatrix(currentLayer, currentExpert) {
  for (let L = 0; L < TOTAL_LAYERS; L++) {
    for (let E = 0; E < EXPERTS_PER_LAYER; E++) {
      let color;
      if (currentLayer == null) color = COLORS.pending;
      else if (L < currentLayer) color = COLORS.done;
      else if (L === currentLayer) {
        if (currentExpert != null && E < currentExpert) color = COLORS.done;
        else if (currentExpert != null && E === currentExpert) color = COLORS.active;
        else color = COLORS.pending;
      } else color = COLORS.pending;
      cells[L][E].setAttribute("fill", color);
    }
  }
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

function update(p) {
  if (p.heartbeat) return;

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

  // Big runtime + spend (server-derived, refresh-safe)
  if (s.pod_runtime_sec > 0) {
    $("pod-runtime").textContent = fmtDuration(s.pod_runtime_sec);
  }
  if (s.pod_spent > 0) {
    $("pod-spent").textContent = `$${s.pod_spent.toFixed(2)}`;
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
    $("layer").textContent = s.current_layer ?? "—";
    $("expert").textContent = s.current_expert ?? "—";
    $("tensors").textContent = (s.tensors_quantized || 0).toLocaleString();
    $("eta").textContent = s.eta || "computing";
    if (s.tensor_rate_per_sec) {
      $("tensor-rate").textContent = `${s.tensor_rate_per_sec.toFixed(1)} /s`;
    }
    if (s.layer_avg_sec > 0) {
      $("layer-avg").textContent = fmtDuration(s.layer_avg_sec);
    }
  }

  // Progress bar: 100% when done, otherwise based on observed layer.
  const pct = done
    ? 100
    : (s.current_layer != null ? Math.min(100, ((s.current_layer + 1) / TOTAL_LAYERS) * 100) : 0);
  $("bar-pct").textContent = pct.toFixed(1) + "%";

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
    global DEFAULT_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="log file to tail")
    args = parser.parse_args()
    DEFAULT_LOG = Path(args.log)
    print(f"BLOCKQUANT // LIVE OPS  ·  http://{args.host}:{args.port}")
    print(f"Session: {SESSION_ID}")
    print(f"Tailing: {DEFAULT_LOG}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
