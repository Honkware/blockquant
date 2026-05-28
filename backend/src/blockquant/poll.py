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
    while provider.is_pipeline_running(instance_id):
        _sleep(poll_interval)
        now = _now()
        progress = provider.get_progress(instance_id)
        if progress and progress != last_progress:
            last_progress = progress
            last_change = now
            if on_progress:
                on_progress(progress)
        if now - start > max_runtime:
            return "max_runtime"
        if now - last_change > stall_timeout:
            return "stalled"
    return "done"
