"""Bounded polling for remote pod pipelines.

The poll loop waits for the in-pod process to exit. With no bound, a hung
remote (a deadlocked quant, a stuck download, a wedged process) keeps the pod
billing until the account runs dry. This caps total runtime and notices a
stalled run, where no new log output has arrived for a while, so the caller
can terminate the pod.
"""
from __future__ import annotations

import time

# Defaults are generous on purpose. A large MoE quant can run for several
# hours, and the download phase can go quiet for many minutes, so the stall
# window stays wide enough that a slow download does not trip it.
DEFAULT_MAX_RUNTIME_S = 8 * 3600
DEFAULT_STALL_TIMEOUT_S = 60 * 60


def poll_remote(
    provider,
    instance_id,
    *,
    poll_interval: int = 15,
    max_runtime: int = DEFAULT_MAX_RUNTIME_S,
    stall_timeout: int = DEFAULT_STALL_TIMEOUT_S,
    on_progress=None,
    _now=time.time,
    _sleep=time.sleep,
) -> str:
    """Poll until the remote process exits or a bound trips.

    Returns "done" on a normal exit, "max_runtime" when the run passes
    max_runtime, or "stalled" when no new log output arrives within
    stall_timeout. The caller is responsible for terminating the pod in
    every case.

    ``_now`` and ``_sleep`` are injectable for tests.
    """
    start = _now()
    last_change = start
    last_progress = ""
    not_running = 0
    while True:
        # Authoritative liveness via the control plane (not SSH). The pod
        # self-terminates on success, so once it is GONE or terminal the run is
        # over. Checking this FIRST is what stops the controller hanging for the
        # whole stall_timeout (hours) after the pod tears itself down once the
        # upload finishes: the SSH checks below would just keep erroring, but the
        # API knows the pod is gone. pod_is_gone fails closed, so a flaky API
        # call can't end a live run early.
        if provider.pod_is_gone(instance_id):
            return "done"
        # is_pipeline_running / get_progress run over SSH, which can hit a
        # transient reset (a local network blip reset every pod's connection at
        # once, and the old code crashed out and terminated pods MID-UPLOAD).
        # Treat an SSH error as "still running, no news" and let stall_timeout
        # decide, so a blip can't kill a job: the pod uploads to HF on its own
        # and the controller reconnects and sees it finish. The pod_is_gone
        # check above already rules out a genuinely vanished pod.
        try:
            running = provider.is_pipeline_running(instance_id)
        except Exception:
            running = True
        # Require TWO consecutive "not running" reads before declaring done. A
        # reconnect right after a network reset can return a spurious empty/
        # garbled result that looks like "done" while the quant is still going,
        # which would end the poll early and fail the job. One real completion
        # reads done twice in a row; a glitch does not.
        if not running:
            not_running += 1
            if not_running >= 2:
                return "done"
        else:
            not_running = 0
        _sleep(poll_interval)
        now = _now()
        try:
            progress = provider.get_progress(instance_id)
        except Exception:
            progress = None
        if progress and progress != last_progress:
            last_progress = progress
            last_change = now
            if on_progress:
                on_progress(progress)
        # A terminal marker in the remote log means the run finished even though
        # its process can linger (the detached self-terminate backstop keeps
        # matching pgrep for ~5 min). Exit as soon as we see it rather than
        # waiting the backstop out or falling through to the pod-gone check.
        if progress and "[done]" in progress:
            return "done"
        if now - start > max_runtime:
            return "max_runtime"
        if now - last_change > stall_timeout:
            return "stalled"
