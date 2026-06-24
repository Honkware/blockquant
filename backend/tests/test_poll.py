"""Tests for the bounded remote poll loop."""
from blockquant import poll


class FakeProvider:
    def __init__(self, running, progress, gone=None, ssh_error=False):
        self._running = list(running)
        self._progress = list(progress)
        self._gone = list(gone) if gone is not None else []
        self._ssh_error = ssh_error

    def pod_is_gone(self, _id):
        # Defaults to "still here" so existing tests keep their old behavior.
        return self._gone.pop(0) if self._gone else False

    def is_pipeline_running(self, _id):
        if self._ssh_error:
            raise ConnectionError("ssh reset")
        return self._running.pop(0) if self._running else False

    def get_progress(self, _id, lines=30):
        if self._ssh_error:
            raise ConnectionError("ssh reset")
        return self._progress.pop(0) if self._progress else ""


def _clock(step):
    """Virtual clock that advances by `step` seconds on each sleep."""
    state = {"t": 0.0}

    def now():
        return state["t"]

    def sleep(_):
        state["t"] += step

    return now, sleep


def test_done_on_normal_exit():
    now, sleep = _clock(step=10)
    prov = FakeProvider(running=[True, True, False], progress=["a", "b"])
    outcome = poll.poll_remote(prov, "pod", poll_interval=1, _now=now, _sleep=sleep)
    assert outcome == "done"


def test_max_runtime_trips():
    now, sleep = _clock(step=100)
    prov = FakeProvider(running=[True] * 10, progress=["a", "b", "c"])
    outcome = poll.poll_remote(
        prov, "pod", poll_interval=1, max_runtime=50, stall_timeout=10_000,
        _now=now, _sleep=sleep,
    )
    assert outcome == "max_runtime"


def test_stall_trips_when_output_stops():
    now, sleep = _clock(step=100)
    prov = FakeProvider(running=[True] * 10, progress=["x", "x", "x"])
    outcome = poll.poll_remote(
        prov, "pod", poll_interval=1, max_runtime=10_000, stall_timeout=50,
        _now=now, _sleep=sleep,
    )
    assert outcome == "stalled"


def test_changing_progress_keeps_it_alive():
    now, sleep = _clock(step=30)
    prov = FakeProvider(running=[True, True, True, False], progress=["a", "b", "c"])
    outcome = poll.poll_remote(
        prov, "pod", poll_interval=1, max_runtime=10_000, stall_timeout=50,
        _now=now, _sleep=sleep,
    )
    assert outcome == "done"


def test_on_progress_receives_new_output():
    now, sleep = _clock(step=10)
    prov = FakeProvider(running=[True, True, False], progress=["a", "b"])
    seen = []
    poll.poll_remote(
        prov, "pod", poll_interval=1, on_progress=seen.append,
        _now=now, _sleep=sleep,
    )
    assert seen == ["a", "b"]


def test_pod_gone_ends_run_even_when_ssh_is_dead():
    # The pod self-terminated post-upload: SSH is dead so is_pipeline_running
    # would just error forever, but the control plane reports the pod GONE. The
    # loop must end as "done" right away instead of hanging out the stall
    # timeout (the bug that froze the embed at 95% for hours).
    now, sleep = _clock(step=10)
    prov = FakeProvider(running=[], progress=[], gone=[True], ssh_error=True)
    outcome = poll.poll_remote(
        prov, "pod", poll_interval=1, max_runtime=10_000, stall_timeout=50,
        _now=now, _sleep=sleep,
    )
    assert outcome == "done"


def test_ssh_blip_with_live_pod_does_not_end_early():
    # SSH errors on every call but the pod is still alive per the API: a network
    # blip, not a finished run. The loop must NOT declare "done"; it rides it out
    # until stall_timeout (preserves the original network-drop tolerance).
    now, sleep = _clock(step=100)
    prov = FakeProvider(running=[], progress=[], gone=[False] * 10, ssh_error=True)
    outcome = poll.poll_remote(
        prov, "pod", poll_interval=1, max_runtime=10_000, stall_timeout=50,
        _now=now, _sleep=sleep,
    )
    assert outcome == "stalled"


def test_done_marker_ends_run_while_process_lingers():
    # is_pipeline_running stays True (the self-terminate backstop keeps matching
    # pgrep) but the log shows [done]: the run is finished, so exit on the marker
    # rather than waiting the backstop out.
    now, sleep = _clock(step=10)
    prov = FakeProvider(running=[True, True, True, True], progress=["working", "[done]"])
    outcome = poll.poll_remote(
        prov, "pod", poll_interval=1, max_runtime=10_000, stall_timeout=10_000,
        _now=now, _sleep=sleep,
    )
    assert outcome == "done"
