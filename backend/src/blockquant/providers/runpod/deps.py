"""Lazy SDK imports for the RunPod provider."""

_runpod = None
_paramiko = None


def _ensure_runpod():
    global _runpod
    if _runpod is None:
        import runpod
        _runpod = runpod
    return _runpod


def _ensure_paramiko():
    global _paramiko
    if _paramiko is None:
        import paramiko
        _paramiko = paramiko
    return _paramiko
