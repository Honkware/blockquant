"""Guards on the vendored copies of turboderp's self-calibration scripts.

The whole point of backend/src/blockquant/selfcal/VENDOR_MANIFEST.json is that the
next person to bump EXLLAMAV3_REF can tell, mechanically, which of those files are
pristine upstream and which carry a local patch they have to re-apply. That only
works if the manifest cannot silently drift from the files, hence this module.

Everything here is offline except the drift check against GitHub, which is marked
slow: CI should not fail because raw.githubusercontent.com had a bad minute.
"""

import hashlib
import json
from pathlib import Path

import pytest

SELFCAL_DIR = Path(__file__).resolve().parents[1] / "src" / "blockquant" / "selfcal"
MANIFEST_PATH = SELFCAL_DIR / "VENDOR_MANIFEST.json"
DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.runpod"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _files() -> dict:
    return _manifest()["files"]


def _upstream_bytes(path: Path) -> bytes:
    """File contents without the '#' attribution header we prepend at vendoring time.

    Stripping it is what lets an unmodified file hash to the upstream sha256, so the
    manifest can record one hash that is checkable both on disk and against GitHub.
    """
    lines = path.read_bytes().splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith(b"#"):
        i += 1
    if i < len(lines) and lines[i].strip() == b"":
        i += 1
    return b"".join(lines[i:])


def test_manifest_exists():
    assert MANIFEST_PATH.is_file(), f"no vendor manifest at {MANIFEST_PATH}"


def test_every_vendored_file_is_listed():
    listed = set(_files())
    on_disk = {
        str(p.relative_to(SELFCAL_DIR))
        for p in SELFCAL_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert on_disk == listed, (
        f"selfcal/ and the manifest disagree: only on disk {sorted(on_disk - listed)}, "
        f"only in manifest {sorted(listed - on_disk)}"
    )


def test_vendored_files_match_the_manifest():
    for name, meta in _files().items():
        path = SELFCAL_DIR / name
        assert path.is_file(), f"{name}: listed in the manifest but missing from selfcal/"
        actual = hashlib.sha256(_upstream_bytes(path)).hexdigest()
        expected = meta.get("local_sha256", meta["sha256"])
        if "local_sha256" in meta:
            hint = (
                "the recorded local patch was edited or lost. Re-apply it and update "
                "local_sha256, or update local_patch to describe the new diff."
            )
        else:
            hint = (
                "this file is supposed to be pristine upstream. Either revert the edit, "
                "or record it as a local_sha256 + local_patch pair in the manifest."
            )
        assert actual == expected, f"{name}: {actual[:16]} != manifest {expected[:16]} -- {hint}"


def test_local_patches_are_declared_and_real():
    for name, meta in _files().items():
        if "local_sha256" not in meta:
            assert "local_patch" not in meta, (
                f"{name}: has a local_patch note but no local_sha256, so nothing checks it"
            )
            continue
        assert meta.get("local_patch"), f"{name}: patched but no local_patch note saying why"
        assert meta["local_sha256"] != meta["sha256"], (
            f"{name}: local_sha256 equals the upstream sha256, so the patch is gone -- "
            f"drop local_sha256/local_patch from the manifest"
        )


def test_sources_point_at_the_pinned_ref():
    ref = _manifest()["_ref"]
    for name, meta in _files().items():
        assert meta["ref"] == ref, f"{name}: ref {meta['ref']} is not the manifest's {ref}"
        assert ref in meta["source"], (
            f"{name}: source URL does not pin the ref, so re-fetching it would give "
            f"whatever is on that branch today: {meta['source']}"
        )


def test_ref_matches_the_image():
    """The scripts reach into exllamav3.modules internals, so they are only valid
    against the exllamav3 the image installs. A silent bump of one without the other
    is the failure this catches."""
    args = [
        line.split("=", 1)[1].strip()
        for line in DOCKERFILE.read_text().splitlines()
        if line.startswith("ARG EXLLAMAV3_REF=")
    ]
    assert len(args) == 1, f"expected one ARG EXLLAMAV3_REF in {DOCKERFILE}, found {len(args)}"
    arg = args[0]
    assert arg == _manifest()["_ref"], (
        f"Dockerfile.runpod pins exllamav3 {arg} but selfcal/ was vendored from "
        f"{_manifest()['_ref']}. Re-vendor the scripts or revert the bump."
    )


def test_vendored_scripts_compile():
    """Compiled in-process rather than with py_compile so the check leaves no
    __pycache__ behind in src/."""
    for name in _files():
        path = SELFCAL_DIR / name
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            pytest.fail(f"{name} does not parse ({exc}) -- truncated copy, or a botched patch")


@pytest.mark.slow
def test_upstream_has_not_moved():
    """Fetches the pinned ref from GitHub. A commit sha cannot change under us, so a
    failure here means the manifest's sha256 was recorded wrong, not that turbo pushed."""
    import urllib.request

    for name, meta in _files().items():
        try:
            with urllib.request.urlopen(meta["source"], timeout=20) as resp:
                body = resp.read()
        except Exception as exc:  # network, not a vendoring problem
            pytest.skip(f"could not fetch {name}: {exc}")
        assert hashlib.sha256(body).hexdigest() == meta["sha256"], (
            f"{name}: upstream at the pinned ref does not match the manifest's sha256"
        )
