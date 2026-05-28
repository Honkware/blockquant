"""Tests for the bounded remote poll loop."""
from blockquant import poll


class FakeProvider:
    def __init__(self, running, progress):
        self._running = list(running)
        self._progress = list(progress)

    def is_pipeline_running(self, _id):
        return self._running.pop(0) if self._running else False

    def get_progress(self, _id, lines=30):
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
