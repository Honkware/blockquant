"""Printing a sliding tail window without repeating yourself.

get_progress is `grep | tail -n N`. While the filtered log is shorter than N
the window only grows, and a prefix test is enough. Once it passes N the window
SLIDES, the prefix test fails every poll, and the whole window gets reprinted.
That was invisible until the [sc] heartbeat started adding a line every 30s:
the controller log for one job hit 94 lines carrying 62 distinct ones, and the
same repeats go to the Discord embed.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def unseen():
    sys.argv = ["run_runpod_job.py"]
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("rj", ROOT / "scripts/run_runpod_job.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m._unseen_lines


def test_nothing_new_prints_nothing(unseen):
    assert unseen(["a", "b"], ["a", "b"]) == []


def test_a_growing_window_prints_only_the_growth(unseen):
    assert unseen(["a", "b"], ["a", "b", "c"]) == ["c"]


def test_a_slid_window_prints_only_the_growth(unseen):
    # THE case. tail -n 3 over a log that gained one line: the new window does
    # not start with the old one, and the prefix test reprinted all three.
    assert unseen(["a", "b", "c"], ["b", "c", "d"]) == ["d"]


def test_a_window_that_slid_several_lines(unseen):
    assert unseen(["a", "b", "c"], ["c", "d", "e"]) == ["d", "e"]


def test_no_overlap_at_all_prints_everything(unseen):
    # Polls far enough apart that the window turned over completely; the lines
    # in between are genuinely lost, and reprinting what we have is right.
    assert unseen(["a", "b"], ["y", "z"]) == ["y", "z"]


def test_the_first_poll_prints_everything(unseen):
    assert unseen([], ["a", "b"]) == ["a", "b"]


def test_repeated_identical_lines_are_not_swallowed(unseen):
    # An overlap search must not treat a genuine repeat as already-seen. The
    # heartbeat carries elapsed seconds partly so this stays rare, but a stage
    # can legitimately print the same line twice.
    assert unseen(["x", "a"], ["a", "a"]) == ["a"]


def test_a_full_window_of_heartbeats(unseen):
    prev = [f"[sc] sc_trace: {i}s working" for i in range(0, 450, 30)]
    cur = prev[1:] + ["[sc] sc_trace: 450s working"]
    assert unseen(prev, cur) == ["[sc] sc_trace: 450s working"]
