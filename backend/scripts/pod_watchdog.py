#!/usr/bin/env python3
"""Reap RunPod pods that nothing is driving any more.

Report-only unless --enforce, so a misjudgement costs a log line and not a
six-hour quant. Every branch that cannot *prove* a pod is abandoned keeps it:
unparseable name, unreadable age, API hiccup, live controller, any process
mentioning the pod -- all resolve to keep.

This is the backstop, not the first line. src/index.js already reaps
`bq-<pid>-*` pods with a dead controller every two minutes, but only while the
bot itself is up, and only for names matching that pattern. What it cannot
catch is what leaks in practice: pods created by hand for one-off research
(`exliberate7`, `klverify`, `bq-phase3`), which own no controller process and
so fall through every check the bot makes. Those, plus anything at all if the
bot is down, are this script's job.

The pod that matters is bq-<pid>-<n>-<ts>, created by run_runpod_job.py and
run_catbench_job.py. The <pid> in the name is the controller's, which is what
makes "is anyone still driving this" answerable at all: look the pid up, check
it is really that controller and not a recycled number, and leave the pod
alone if it is alive. Names outside that pattern get an age ceiling instead,
because there is nothing else to go on.

Install as a VPS cron job. The script reads .env itself; do not source it from
cron, which is how the previous watchdog spent 905 runs dying on a KeyError:

  */10 * * * * /root/blockquant/backend/venv/bin/python \
      /root/blockquant/backend/scripts/pod_watchdog.py >> /root/watchdog.log 2>&1
"""
import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# backend/scripts/ -> repo root. Guarded because a watchdog that dies at
# import time is worth less than no watchdog at all.
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent

# Pods nothing here is allowed to touch, ever. The broadcast node is
# intentionally long-lived; see wallet-derive ops/README.md.
EXEMPT_PREFIXES = ("btc-node",)

# A live one of these driving a pod means the pod is working, whatever its age.
CONTROLLER_SCRIPTS = ("run_runpod_job.py", "run_catbench_job.py", "run_parallel_quants.py")

# bq-<controller pid>-<ts % 100000>-<ts>, assembled by run_*_job.py's run_tag
# plus the provider's own "-{ts}" suffix.
RUN_TAG = re.compile(r"^bq-(\d+)-(\d+)-(\d+)$")

# "Rented by User: Sun Aug 09 2026 21:18:26 GMT+0000 (...)"
RENTED = re.compile(r"([A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{4} \d{2}:\d{2}:\d{2}) GMT([+-]\d{4})")

KEEP, REAP = "keep", "reap"


def _env(name, default, cast=float):
    return cast(os.environ.get(name, default))


@dataclass
class Config:
    """Read at construction, not import, so cron's environment actually lands."""

    # A bq- pod whose controller is gone is finished either way, but the bot's
    # own reaper already covers that case fast. Hang back and give a restarted
    # controller room to reclaim the pod before assuming the worst.
    bq_orphan_min_age: float = field(default_factory=lambda: _env("WATCHDOG_BQ_MIN_AGE", 45 * 60))
    # Ad-hoc pods have no owning process to check, so age is all there is.
    adhoc_max_age: float = field(default_factory=lambda: _env("WATCHDOG_ADHOC_MAX_AGE", 2 * 3600))
    # Shout about anything unattended this long, whether or not it gets reaped.
    alert_age: float = field(default_factory=lambda: _env("WATCHDOG_ALERT_AGE", 30 * 60))
    # Consecutive runs a pod must look abandoned before it is touched.
    strikes: int = field(default_factory=lambda: _env("WATCHDOG_STRIKES", 3, int))
    state: Path = field(default_factory=lambda: Path(_env(
        "WATCHDOG_STATE", ROOT / "backend/logs/pod_watchdog_state.json", str)))
    holds: Path = field(default_factory=lambda: Path(_env(
        "WATCHDOG_HOLDS", ROOT / "backend/logs/pod_watchdog_holds.txt", str)))


@dataclass
class Verdict:
    action: str
    reason: str
    age: float | None = None
    strikes: int = 0
    alert: bool = False


@dataclass
class Proc:
    pid: int
    cmdline: str
    started: float | None = None


@dataclass
class Pod:
    id: str
    name: str
    cost: float = 0.0
    rented_at: float | None = None
    gpu_util: int | None = None
    cpu_pct: int | None = None
    raw: dict = field(default_factory=dict)


def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ} [watchdog] {msg}", flush=True)


def parse_run_tag(name):
    """(controller pid, create ts) for a bq- pod name, else None."""
    m = RUN_TAG.match(name or "")
    if not m:
        return None
    pid, short, ts = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if ts % 100000 != short:  # not one of ours, whatever else it looks like
        return None
    return pid, ts


def parse_rented_at(last_status_change):
    m = RENTED.search(last_status_change or "")
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%a %b %d %Y %H:%M:%S %z").timestamp()
    except ValueError:
        return None


def pod_age(pod, now):
    """Seconds billing has been running, or None if we cannot tell.

    Two clocks disagree by design: the name carries creation, lastStatusChange
    moves on a restart. Take the younger one -- underestimating age only ever
    keeps a pod alive longer.
    """
    ages = []
    if pod.rented_at:
        ages.append(now - pod.rented_at)
    tag = parse_run_tag(pod.name)
    if tag:
        ages.append(now - tag[1])
    ages = [a for a in ages if a >= 0]
    return min(ages) if ages else None


def _boot_time():
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    return None


def read_procs():
    """pid -> Proc from /proc. Empty dict if unreadable, which reads as 'unsure'."""
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return {}
    try:
        btime, clk = _boot_time(), os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        btime, clk = None, 100
    procs = {}
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue  # exited mid-scan
        started = None
        if btime:
            try:
                stat = (entry / "stat").read_text()
                # field 22 (starttime), counted after the comm field's ")"
                started = btime + int(stat[stat.rfind(")") + 2:].split()[19]) / clk
            except (OSError, IndexError, ValueError):
                pass
        procs[int(entry.name)] = Proc(int(entry.name), cmdline, started)
    return procs


def controller_alive(pid, created_ts, procs):
    """True only if pid is a blockquant controller that predates the pod.

    The start-time check is the pid-reuse guard: a controller cannot have
    launched a pod before it existed, so a process that started after the pod
    is a recycled number wearing a dead controller's pid. Unknown start time
    counts as alive, because guessing wrong the other way kills a live job.
    """
    proc = procs.get(pid)
    if not proc:
        return False
    if not any(s in proc.cmdline for s in CONTROLLER_SCRIPTS):
        return False
    if proc.started is not None and created_ts and proc.started > created_ts + 60:
        return False
    return True


def referenced_by(pod, procs):
    """Any process naming this pod: detached script, a person in a shell, us.

    Whole-token match, not substring: pod ids are short enough that a bare
    `in` hits things like agetty's "linux", and a stray match here silently
    grants a leaked pod immunity forever.
    """
    toks = [t for t in (pod.id, pod.name) if t and len(t) >= 6]
    if not toks:
        return None
    pat = re.compile("|".join(rf"(?<![\w-]){re.escape(t)}(?![\w-])" for t in toks))
    for proc in procs.values():
        if pat.search(proc.cmdline):
            return proc.pid
    return None


def read_holds(path):
    """Pod ids or names an operator has claimed. One per line, # comments."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return set()
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}


def classify(pod, procs, now, cfg, holds, prior_strikes=0):
    if any(pod.name.startswith(p) for p in EXEMPT_PREFIXES):
        return Verdict(KEEP, "exempt prefix")
    if pod.id in holds or pod.name in holds:
        return Verdict(KEEP, "held by operator")

    age = pod_age(pod, now)
    tag = parse_run_tag(pod.name)

    if tag and controller_alive(tag[0], tag[1], procs):
        return Verdict(KEEP, f"controller pid {tag[0]} alive", age)

    pid = referenced_by(pod, procs)
    if pid:
        return Verdict(KEEP, f"referenced by pid {pid}", age)

    if not procs:
        return Verdict(KEEP, "process table unreadable", age, alert=True)
    if age is None:
        return Verdict(KEEP, "age unknown", None, alert=True)

    if tag:
        limit, why = cfg.bq_orphan_min_age, f"controller pid {tag[0]} gone"
    else:
        limit, why = cfg.adhoc_max_age, "ad-hoc pod, nothing driving it"

    alert = age >= cfg.alert_age
    if age < limit:
        return Verdict(KEEP, f"{why}, under {limit / 60:.0f}m ceiling", age, alert=alert)

    # Never act on a single sighting. Three consecutive runs of a */10 cron is
    # twenty-odd minutes of the pod looking abandoned every time we looked.
    strikes = prior_strikes + 1
    if strikes < cfg.strikes:
        return Verdict(KEEP, f"{why}, strike {strikes}/{cfg.strikes}", age, strikes, True)
    return Verdict(REAP, f"{why}, {age / 3600:.1f}h old", age, strikes, True)


def _pod(p):
    rt = p.get("runtime") or {}
    gpus = rt.get("gpus") or []
    return Pod(
        id=p["id"],
        name=p.get("name") or "",
        cost=p.get("costPerHr") or 0.0,
        rented_at=parse_rented_at(p.get("lastStatusChange")),
        gpu_util=gpus[0].get("gpuUtilPercent") if gpus else None,
        cpu_pct=(rt.get("container") or {}).get("cpuPercent"),
        raw=p,
    )


def fetch_graphql(api_key):
    """GraphQL, not runpod.get_pods() -- the SDK drops runtime and we want it."""
    import requests

    q = """query { myself { pods {
      id name costPerHr desiredStatus lastStatusChange
      runtime { gpus { gpuUtilPercent } container { cpuPercent } }
    } } }"""
    r = requests.post("https://api.runpod.io/graphql", json={"query": q},
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"runpod graphql: {body['errors']}")
    return [_pod(p) for p in (body["data"]["myself"]["pods"] or [])
            if p.get("desiredStatus") == "RUNNING"]


def fetch_rest(api_key):
    """Fallback list. GraphQL has a history of timing out on rpa_ keys, which
    is why _terminate_stray_pods uses REST. No runtime or rental clock here, so
    only bq- names carry an age and everything else gets kept."""
    req = urllib.request.Request("https://rest.runpod.io/v1/pods",
                                 headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    rows = body if isinstance(body, list) else body.get("data", [])
    return [_pod(p) for p in rows
            if (p.get("desiredStatus") or "RUNNING") == "RUNNING"]


def fetch_pods(api_key):
    try:
        return fetch_graphql(api_key), "graphql"
    except Exception as exc:
        log(f"WARN graphql list failed ({exc}); falling back to REST")
        return fetch_rest(api_key), "rest"


def load_state(path):
    try:
        return json.loads(path.read_text()).get("pods", {})
    except (OSError, ValueError):
        return {}


def save_state(path, pods):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pods": pods}, indent=1))
    except OSError as exc:
        log(f"WARN could not write state {path}: {exc}")


def describe(pod, v):
    age = f"{v.age / 60:.0f}m" if v.age is not None else "age?"
    # GPU idle is not evidence: a healthy pod sits at 0% for the whole download
    # and the whole upload. Printed to inform a human, never to decide.
    util = f"gpu {pod.gpu_util}% cpu {pod.cpu_pct}%" if pod.gpu_util is not None else "no runtime"
    return f"{pod.id} {pod.name or '<unnamed>'} [{age}, ${pod.cost:.2f}/hr, {util}] {v.reason}"


def terminate(api_key, pod_id):
    import runpod

    runpod.api_key = api_key
    runpod.terminate_pod(pod_id)


def main():
    ap = argparse.ArgumentParser(description="Reap RunPod pods nothing is driving any more.")
    ap.add_argument("--enforce", action="store_true",
                    help="actually terminate; without it nothing is touched")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        log("ERROR RUNPOD_API_KEY not set (looked in env and .env); doing nothing")
        raise SystemExit(1)

    cfg = Config()
    try:
        pods, via = fetch_pods(api_key)
    except Exception as exc:
        log(f"ERROR could not list pods, leaving everything alone: {exc}")
        raise SystemExit(1)

    procs = read_procs()
    holds = read_holds(cfg.holds)
    prior = load_state(cfg.state)
    now = time.time()

    mode = "ENFORCE" if args.enforce else "report-only"
    log(f"{mode} via {via}: {len(pods)} running pod(s), "
        f"{len(procs)} process(es), {len(holds)} hold(s)")

    state, reaped = {}, 0
    for pod in pods:
        v = classify(pod, procs, now, cfg, holds, prior.get(pod.id, {}).get("strikes", 0))
        line = describe(pod, v)
        if v.action == REAP:
            burned = (v.age or 0) / 3600 * pod.cost
            if not args.enforce:
                log(f"WOULD REAP {line} (${burned:.2f} burned) -- report-only, left running")
                state[pod.id] = {"strikes": v.strikes, "name": pod.name}
                continue
            log(f"REAP {line} (${burned:.2f} burned)")
            try:
                terminate(api_key, pod.id)
                reaped += 1
            except Exception as exc:
                log(f"  FAILED to terminate {pod.id}: {exc}")
                state[pod.id] = {"strikes": v.strikes, "name": pod.name}
        else:
            log(f"{'ALERT keep' if v.alert else 'keep'} {line}")
            if v.strikes:
                state[pod.id] = {"strikes": v.strikes, "name": pod.name}

    save_state(cfg.state, state)
    if args.enforce and reaped:
        log(f"terminated {reaped} pod(s)")


if __name__ == "__main__":
    main()
