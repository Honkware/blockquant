"""The pod's SSH sessions must carry the image's load-bearing ENV.

sshd on a RunPod pod runs outside the container's process tree, so an SSH
session inherits none of the Docker ENV. Both of these failed silently: torch
recompiled exllamav3's CUDA extension on every pod because TORCH_CUDA_ARCH_LIST
was unset (missing the cache the image built), and qbench could not see the
wikitext-2 warmed into the image because HF_DATASETS_CACHE was unset.
"""
from __future__ import annotations

import pathlib
import re

from blockquant.providers.runpod.provider import RunPodProvider

DOCKERFILE = pathlib.Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.runpod"
PROVIDER = (pathlib.Path(__file__).resolve().parents[2]
            / "src" / "blockquant" / "providers" / "runpod" / "provider.py")


def _dockerfile_env(name: str) -> str:
    m = re.search(rf'^ENV {name}="?([^"\n]+)"?', DOCKERFILE.read_text(), re.M)
    assert m, f"{name} is not set in the Dockerfile"
    return m.group(1).strip()


def test_pod_env_matches_the_image():
    """POD_ENV is a second copy of what the Dockerfile sets, so it can drift.
    A wrong arch list is worse than none: it would silently rebuild."""
    for name, value in RunPodProvider.POD_ENV.items():
        assert _dockerfile_env(name) == value, (
            f"{name} is {value!r} in provider.py but {_dockerfile_env(name)!r} in the Dockerfile"
        )


def test_every_remote_exec_carries_the_env():
    """Wrapped at the paramiko boundary, not at each caller, so a path added
    later cannot skip it."""
    sites = re.findall(r"client\.exec_command\(\s*([^,\n]+)", PROVIDER.read_text())
    assert sites, "no exec_command call sites found -- did the helper move?"
    for s in sites:
        assert "_with_pod_env" in s, f"a remote exec skips the pod env: {s}"


def test_the_export_is_shell_safe():
    """The arch list contains semicolons; unquoted they would end the command."""
    out = RunPodProvider._with_pod_env("echo hi")
    assert out.endswith("; echo hi")
    assert "'8.0;8.6;8.9;9.0;12.0+PTX'" in out
