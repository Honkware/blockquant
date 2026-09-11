"""Does a controller actually survive the way pm2 kills things?

The claim that `spawn(..., {detached: true})` was enough turned out to be wrong
and cost a paid /catbench run, so this asserts it against the real mechanism
rather than reading the code. pm2 6.x defaults treekill:true: it enumerates
descendants by PPID and signals every one of them. `tree_kill` below is that
walk. The negative control runs the OLD spawn through the same walk and must
die, otherwise this file proves nothing.
"""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DETACHED = ROOT / "src" / "services" / "detached.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(not NODE, reason="node not installed")

# Stays alive after spawning, like the bot does, so there is a tree to kill.
REPARENTED = """
import {{ spawnDetached }} from '{mod}';
const h = spawnDetached({{
  command: '/bin/sh',
  args: ['-c', 'sleep {secs}; exit {code}'],
  cwd: process.cwd(),
  env: process.env,
  logPath: process.argv[2],
  kind: 'test',
  meta: {{ modelId: 'org/model' }},
}});
process.stdout.write(JSON.stringify(h) + '\\n');
setInterval(() => {{}}, 1000);
"""

# What runpodCli/catbenchCli did before: detached, unref'd, still a child.
PLAIN = """
import {{ spawn }} from 'node:child_process';
const c = spawn('/bin/sh', ['-c', 'sleep {secs}'], {{ detached: true, stdio: 'ignore' }});
c.unref();
process.stdout.write(JSON.stringify({{ pid: c.pid }}) + '\\n');
setInterval(() => {{}}, 1000);
"""


ARGV = """
import {{ spawnDetached }} from '{mod}';
const h = spawnDetached({{
  command: '/bin/sh',
  args: ['-c', 'printf %s "$1" > "$2"', 'bq', {arg}, {out}],
  cwd: process.cwd(),
  env: process.env,
  logPath: process.argv[2],
}});
process.stdout.write(JSON.stringify(h) + '\\n');
"""


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def descendants(pid):
    """Every process under pid, by PPID, the way pm2's TreeKill walks it."""
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    kids = [int(p) for p in out.split()]
    return kids + [d for k in kids for d in descendants(k)]


def tree_kill(pid, sig=signal.SIGKILL):
    """Signal pid and everything pm2 would consider part of its tree."""
    targets = descendants(pid) + [pid]
    for target in targets:
        try:
            os.kill(target, sig)
        except ProcessLookupError:
            pass
    return targets


def launch(tmp_path, source, name):
    script = tmp_path / name
    script.write_text(source)
    env = {**os.environ, "BQ_RUN_DIR": str(tmp_path / "controllers")}
    proc = subprocess.Popen(
        [NODE, str(script), str(tmp_path / "run.log")],
        stdout=subprocess.PIPE, text=True, env=env, cwd=str(ROOT),
    )
    handle = json.loads(proc.stdout.readline())
    return proc, handle


def cleanup(proc, *pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()
    proc.wait(timeout=10)


def test_survives_a_pm2_style_tree_kill(tmp_path):
    src = REPARENTED.format(mod=DETACHED, secs=30, code=0)
    proc, handle = launch(tmp_path, src, "reparented.mjs")
    pid = handle["pid"]
    try:
        # Reparented before spawnDetached returned: not in the tree at all.
        assert descendants(proc.pid) == []
        tree_kill(proc.pid)
        time.sleep(1)
        assert alive(pid), "controller died with the bot"
    finally:
        cleanup(proc, pid)


def test_the_old_spawn_did_not(tmp_path):
    """Negative control. Without this the test above proves nothing."""
    proc, handle = launch(tmp_path, PLAIN.format(secs=30), "plain.mjs")
    pid = handle["pid"]
    try:
        assert pid in descendants(proc.pid)
        tree_kill(proc.pid)
        time.sleep(1)
        assert not alive(pid)
    finally:
        cleanup(proc, pid)


def test_exit_status_outlives_the_parent(tmp_path):
    """Completion detection without the Node exit event: the wrapper records
    the status, and it is there whether or not anyone was watching."""
    src = REPARENTED.format(mod=DETACHED, secs=2, code=7)
    proc, handle = launch(tmp_path, src, "reparented.mjs")
    pid = handle["pid"]
    try:
        tree_kill(proc.pid)  # bot gone before the run finishes
        status = Path(handle["statusPath"])
        for _ in range(150):
            if status.exists():
                break
            time.sleep(0.1)
        assert status.read_text() == "7"
        assert not alive(pid)
    finally:
        cleanup(proc, pid)


def test_a_signalled_controller_reads_as_signalled(tmp_path):
    """A shell reports a killed child as 128+signum, which is what keeps a
    cancelled run terminal instead of respawning a pod."""
    src = REPARENTED.format(mod=DETACHED, secs=30, code=0)
    proc, handle = launch(tmp_path, src, "reparented.mjs")
    pid = handle["pid"]
    try:
        for child in descendants(pid):
            os.kill(child, signal.SIGTERM)
        status = Path(handle["statusPath"])
        for _ in range(100):
            if status.exists():
                break
            time.sleep(0.1)
        assert status.read_text() == str(128 + signal.SIGTERM)
    finally:
        cleanup(proc, pid)


def test_decode_status_maps_signals_and_codes(tmp_path):
    probe = tmp_path / "decode.mjs"
    probe.write_text(
        f"import {{ decodeStatus }} from '{DETACHED}';\n"
        "console.log(JSON.stringify(['0','1','137','x'].map(decodeStatus)));\n"
    )
    out = subprocess.run([NODE, str(probe)], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [
        {"code": 0, "signal": None},
        {"code": 1, "signal": None},
        {"code": None, "signal": "SIGKILL"},
        None,
    ]


def test_arguments_are_not_reinterpreted(tmp_path):
    """The wrapper is a shell now, and /quant's --test-prompt is user text that
    lands in argv. "$@" keeps every argument one argument."""
    evil = 'a cat; touch ' + str(tmp_path / "pwned") + ' #$(whoami)'
    out = tmp_path / "arg.txt"
    src = ARGV.format(mod=DETACHED, arg=json.dumps(evil), out=json.dumps(str(out)))
    proc, handle = launch(tmp_path, src, "argv.mjs")
    try:
        for _ in range(100):
            if out.exists():
                break
            time.sleep(0.1)
        assert out.read_text() == evil
        assert not (tmp_path / "pwned").exists()
    finally:
        cleanup(proc, handle["pid"])


def test_registry_reports_a_live_controller(tmp_path):
    """What `npm run status` and the boot warning read."""
    src = REPARENTED.format(mod=DETACHED, secs=30, code=0)
    proc, handle = launch(tmp_path, src, "reparented.mjs")
    pid = handle["pid"]
    probe = tmp_path / "list.mjs"
    probe.write_text(
        f"import {{ list }} from '{DETACHED}';\nconsole.log(JSON.stringify(list()));\n"
    )
    try:
        env = {**os.environ, "BQ_RUN_DIR": str(tmp_path / "controllers")}
        out = subprocess.run([NODE, str(probe)], capture_output=True, text=True,
                             check=True, env=env).stdout
        rows = json.loads(out)
        assert [(r["kind"], r["pid"], r["running"]) for r in rows] == [("test", pid, True)]
        assert rows[0]["meta"]["modelId"] == "org/model"
    finally:
        cleanup(proc, pid)
