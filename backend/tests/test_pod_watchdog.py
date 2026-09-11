"""What the pod watchdog is allowed to kill, and everything it is not.

The script it replaces had two faults worth pinning down forever. It sourced
.env from cron without exporting, so the api key never reached the python
child and 905 consecutive runs died on a KeyError. And its scope rule was
"terminate every pod not named btc-node* whenever no runpod_controller.py is
alive", which on the night it was found would have killed three production
quants hours into their conversions, because blockquant's controllers are
named run_runpod_job.py and run_catbench_job.py and neither one has ever been
called runpod_controller.py.

So the tests that matter most here are the negative ones: a working job
survives, an unreadable anything survives, and nothing dies on first sight.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pod_watchdog as wd  # noqa: E402

NOW = 1786310305.0
POD_TS = int(NOW)
# bq-<pid>-<ts % 100000>-<ts>, the real name observed on the account.
LIVE_NAME = f"bq-4119069-{POD_TS % 100000}-{POD_TS}"


def bq_name(ts, pid=4119069):
    return f"bq-{pid}-{int(ts) % 100000}-{int(ts)}"


def pod(name, pid="p1", cost=0.44, rented=None, **kw):
    return wd.Pod(id=pid, name=name, cost=cost, rented_at=rented, **kw)


def controller(pid=4119069, script="run_runpod_job.py", started=NOW - 60):
    cmd = f"/root/blockquant/backend/venv/bin/python /root/blockquant/backend/scripts/{script} --model x"
    return {pid: wd.Proc(pid, cmd, started)}


def cfg(**kw):
    base = dict(bq_orphan_min_age=45 * 60, adhoc_max_age=2 * 3600,
                alert_age=30 * 60, strikes=3,
                state=Path("/nonexistent/s.json"), holds=Path("/nonexistent/h.txt"))
    base.update(kw)
    return wd.Config(**base)


def verdict(p, procs=None, now=NOW, strikes=0, holds=frozenset(), c=None):
    return wd.classify(p, procs if procs is not None else {1: wd.Proc(1, "init")},
                       now, c or cfg(), set(holds), strikes)


# ── Name and clock parsing ──────────────────────────────────────────────────

def test_a_real_pod_name_yields_its_controller_pid_and_creation_time():
    assert wd.parse_run_tag(LIVE_NAME) == (4119069, POD_TS)


@pytest.mark.parametrize("name", ["", "btc-node-1", "exliberate7", "klverify",
                                  "bq-phase3", "blockquant-1786310305", "bq-4119069"])
def test_names_that_are_not_ours_yield_no_controller(name):
    assert wd.parse_run_tag(name) is None


def test_a_real_pod_name_parses_even_though_the_fields_disagree():
    # Observed in production: bq-4130775-20914-1786320953. 1786320953 % 100000
    # is 20953, not 20914, because run_tag is stamped when the controller starts
    # and the pod's own timestamp is appended ~39s later when it is rented.
    # Cross-checking them dropped every real pod into the ad-hoc bucket, where a
    # multi-hour quant would eventually blow the age ceiling and be killed.
    assert wd.parse_run_tag("bq-4130775-20914-1786320953") == (4130775, 1786320953)


def test_a_bq_name_with_an_implausible_timestamp_is_not_ours():
    assert wd.parse_run_tag("bq-4119069-99999-42") is None


def test_the_rental_clock_is_read_out_of_last_status_change():
    at = wd.parse_rented_at("Rented by User: Sun Aug 09 2026 21:18:26 GMT+0000 (UTC)")
    assert at == pytest.approx(1786310306, abs=1)


@pytest.mark.parametrize("s", [None, "", "Rented by User: whenever", "Stopped by user"])
def test_an_unreadable_rental_clock_is_none_rather_than_a_guess(s):
    assert wd.parse_rented_at(s) is None


def test_age_takes_the_younger_of_the_two_clocks():
    # Name says 2h old, lastStatusChange says 10m. Believe the one that keeps it.
    p = pod(f"bq-1-{POD_TS % 100000}-{POD_TS}", rented=NOW + 7200 - 600)
    assert wd.pod_age(p, NOW + 7200) == pytest.approx(600)


def test_a_pod_with_no_clock_at_all_has_no_age():
    assert wd.pod_age(pod("exliberate7"), NOW) is None


# ── Is anyone driving this pod ──────────────────────────────────────────────

def test_a_running_controller_counts_as_alive():
    assert wd.controller_alive(4119069, POD_TS, controller()) is True


def test_catbench_counts_as_a_controller_too():
    procs = controller(script="run_catbench_job.py")
    assert wd.controller_alive(4119069, POD_TS, procs) is True


def test_a_missing_pid_is_not_alive():
    assert wd.controller_alive(4119069, POD_TS, {}) is False


def test_an_unrelated_process_wearing_the_pid_is_not_a_controller():
    procs = {4119069: wd.Proc(4119069, "sshd: root@notty", NOW - 60)}
    assert wd.controller_alive(4119069, POD_TS, procs) is False


def test_a_recycled_pid_that_started_after_the_pod_is_not_its_controller():
    # A controller cannot have launched a pod before the controller existed.
    procs = controller(started=POD_TS + 3600)
    assert wd.controller_alive(4119069, POD_TS, procs) is False


def test_an_unknown_start_time_is_given_the_benefit_of_the_doubt():
    assert wd.controller_alive(4119069, POD_TS, controller(started=None)) is True


# ── The night it was found: a live job must survive ─────────────────────────

def test_a_live_quant_is_kept_however_old_it_gets():
    # Nine hours in, controller still working. The old script killed this.
    p = pod(LIVE_NAME, rented=NOW - 9 * 3600)
    v = verdict(p, controller(), now=NOW + 9 * 3600)
    assert v.action == wd.KEEP
    assert "alive" in v.reason


def test_no_runpod_controller_py_anywhere_does_not_condemn_anything():
    # The old script's whole trigger. None of these names is a blockquant
    # controller, and the pod is still working.
    procs = controller(script="run_catbench_job.py")
    assert verdict(pod(LIVE_NAME), procs).action == wd.KEEP


def test_a_pod_named_on_a_detached_command_line_is_kept():
    procs = {99: wd.Proc(99, "bash -c 'ssh pod-xyz789 ./converge.sh'", NOW - 60)}
    v = verdict(pod("klverify", pid="pod-xyz789", rented=NOW - 8 * 3600), procs)
    assert v.action == wd.KEEP and "referenced by pid 99" in v.reason


def test_a_pod_id_buried_inside_another_word_is_not_a_reference():
    # "linux" in agetty's command line once matched a pod whose id was "x".
    procs = {1002: wd.Proc(1002, "/sbin/agetty -o -- --noclear - linux", NOW - 99)}
    assert wd.referenced_by(pod("klverify", pid="pod-linux-1"), procs) is None


def test_a_too_short_pod_token_never_matches_anything():
    procs = {1: wd.Proc(1, "some command mentioning x and y", NOW)}
    assert wd.referenced_by(wd.Pod(id="x", name=""), procs) is None


def test_an_operator_hold_outranks_every_ceiling():
    p = pod("exliberate7", rented=NOW - 30 * 3600)
    assert verdict(p, holds={"exliberate7"}).action == wd.KEEP
    assert verdict(p, holds={"p1"}).action == wd.KEEP


def test_the_broadcast_node_is_never_touched():
    p = pod("btc-node-mainnet", rented=NOW - 400 * 3600)
    assert verdict(p).action == wd.KEEP


# ── Unsure means leave it alone ─────────────────────────────────────────────

def test_an_unreadable_process_table_stops_the_watchdog_reaping():
    p = pod("exliberate7", rented=NOW - 30 * 3600)
    v = verdict(p, procs={})
    assert v.action == wd.KEEP and v.alert


def test_a_pod_of_unknown_age_is_kept_and_shouted_about():
    v = verdict(pod("mystery-pod"))
    assert v.action == wd.KEEP and v.alert and v.reason == "age unknown"


# ── What it does reap ───────────────────────────────────────────────────────

def test_an_orphaned_quant_under_the_ceiling_is_only_watched():
    p = pod(bq_name(NOW - 20 * 60), rented=NOW - 20 * 60)
    v = verdict(p)
    assert v.action == wd.KEEP and "ceiling" in v.reason


def test_an_orphaned_quant_past_the_ceiling_is_reaped_on_the_third_strike():
    p = pod(bq_name(NOW - 2 * 3600), rented=NOW - 2 * 3600)
    assert verdict(p, strikes=0).action == wd.KEEP
    assert verdict(p, strikes=1).action == wd.KEEP
    v = verdict(p, strikes=2)
    assert v.action == wd.REAP and v.strikes == 3


def test_strike_counting_says_where_it_is_up_to():
    v = verdict(pod(bq_name(NOW - 2 * 3600), rented=NOW - 2 * 3600), strikes=1)
    assert "strike 2/3" in v.reason


def test_tonights_leaked_research_pod_is_caught_once_it_is_old_enough():
    p = pod("exliberate7", rented=NOW - 3 * 3600)
    v = verdict(p, strikes=2)
    assert v.action == wd.REAP and "ad-hoc" in v.reason


def test_a_young_research_pod_is_left_alone():
    assert verdict(pod("bq-phase3", rented=NOW - 5 * 60)).action == wd.KEEP


def test_an_unattended_pod_is_flagged_well_before_it_is_reaped():
    v = verdict(pod("exliberate7", rented=NOW - 40 * 60))
    assert v.action == wd.KEEP and v.alert


# ── Holds file ──────────────────────────────────────────────────────────────

def test_the_holds_file_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "holds.txt"
    f.write_text("# mine for the evening\n\nexliberate7\n  pod-abc \n")
    assert wd.read_holds(f) == {"exliberate7", "pod-abc"}


def test_a_missing_holds_file_is_simply_no_holds(tmp_path):
    assert wd.read_holds(tmp_path / "nope.txt") == set()


# ── End to end, and the env fault that started all this ─────────────────────

@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    """main() with the network stubbed out and state in a tmp dir."""
    def go(pods, argv, key="rpa_do_not_leak", state_seed=None):
        monkeypatch.setattr(wd, "ROOT", tmp_path)  # no .env here to load
        monkeypatch.setenv("RUNPOD_API_KEY", key) if key else monkeypatch.delenv(
            "RUNPOD_API_KEY", raising=False)
        monkeypatch.setenv("WATCHDOG_STATE", str(tmp_path / "state.json"))
        monkeypatch.setenv("WATCHDOG_HOLDS", str(tmp_path / "holds.txt"))
        if state_seed:
            (tmp_path / "state.json").write_text(state_seed)
        monkeypatch.setattr(wd, "fetch_pods", lambda k: (pods, "graphql"))
        monkeypatch.setattr(wd, "read_procs", lambda: {1: wd.Proc(1, "init")})
        killed = []
        monkeypatch.setattr(wd, "terminate", lambda k, pid: killed.append(pid))
        monkeypatch.setattr(sys, "argv", ["pod_watchdog.py", *argv])
        code = 0
        try:
            wd.main()
        except SystemExit as exc:
            code = exc.code
        return killed, capsys.readouterr().out, code
    return go


def test_the_api_key_comes_from_dotenv_not_an_exported_shell_var(run):
    # The old cron did `. .env && script`, which sets a shell variable the
    # python child never sees. Nothing here depends on the caller's exports.
    killed, out, code = run([], ["--enforce"])
    assert code == 0 and "report-only" not in out


def test_a_missing_key_stops_the_run_instead_of_raising_keyerror(run):
    killed, out, code = run([], [], key=None)
    assert code == 1 and "RUNPOD_API_KEY not set" in out and "KeyError" not in out


def test_report_only_is_the_default_and_kills_nothing(run):
    p = pod("exliberate7", rented=time.time() - 3 * 3600)
    killed, out, _ = run([p], [], state_seed='{"pods": {"p1": {"strikes": 2}}}')
    assert killed == [] and "WOULD REAP" in out and "left running" in out


def test_enforce_terminates_the_same_pod_report_only_only_named(run):
    p = pod("exliberate7", rented=time.time() - 3 * 3600)
    killed, out, _ = run([p], ["--enforce"], state_seed='{"pods": {"p1": {"strikes": 2}}}')
    assert killed == ["p1"] and "REAP" in out


def test_strikes_survive_between_runs(run, tmp_path):
    p = pod("exliberate7", rented=time.time() - 3 * 3600)
    run([p], [])
    import json
    assert json.loads((tmp_path / "state.json").read_text())["pods"]["p1"]["strikes"] == 1


def test_a_pod_that_recovers_loses_its_strikes(run, tmp_path):
    import json
    young = pod("bq-phase3", rented=time.time() - 60)
    run([young], [], state_seed='{"pods": {"p1": {"strikes": 2}}}')
    assert json.loads((tmp_path / "state.json").read_text())["pods"] == {}


def test_the_key_is_never_printed(run):
    p = pod("exliberate7", rented=time.time() - 3 * 3600)
    _, out, _ = run([p], ["--enforce"], key="rpa_supersecretvalue")
    assert "rpa_supersecretvalue" not in out
