"""Where a CatBench run goes after the pod is gone.

catbench_store.py is stdlib at module scope (Pillow and huggingface_hub are
imported inside the functions that need them), so it loads directly rather than
through the AST trick test_catbench.py uses on the controller.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "catbench_store.py"


@pytest.fixture(scope="module")
def store():
    spec = importlib.util.spec_from_file_location("catbench_store", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png(tmp_path, name, size=(64, 64), colour=(240, 140, 40)):
    from PIL import Image
    p = tmp_path / name
    Image.new("RGBA", size, colour + (255,)).save(p, "PNG")
    return p


def _payload(tmp_path, model_id="Qwen/Qwen3-8B"):
    return {
        "model_id": model_id,
        "display_name": model_id.split("/")[-1],
        "loader": "exl3",
        "engine": "0.0.43",
        "format": "exl3",
        "run_date": "2026-08-09T12:00:00Z",
        "prompts": {"svg": "Create a detailed SVG image of a cute kitten."},
        "svg_source": "<svg><circle r='3'/></svg>",
        "python_source": "import matplotlib.pyplot as plt\n",
        "svg_png": str(_png(tmp_path, "svg.png")),
        "python_png": str(_png(tmp_path, "python.png")),
    }


# ── Keys have to agree with upstream, and with the JS side ──────────────────

def test_norm_key_matches_the_upstream_rule(store):
    assert store.norm_key("North-Mini-Code-1.0_30B") == "north-mini-code-1.0-30b"
    assert store.norm_key("Muse Spark 1.1") == "muse-spark-1.1"
    assert store.model_key("zai-org/GLM-5.2") == "glm-5.2"


def test_a_re_run_of_the_same_model_reuses_its_key(store):
    models = {"qwen3-8b": {"model_id": "Qwen/Qwen3-8B"}}
    assert store.pick_key("Qwen/Qwen3-8B", models) == "qwen3-8b"


def test_a_different_org_with_the_same_stem_does_not_overwrite(store):
    models = {"qwen3-8b": {"model_id": "Qwen/Qwen3-8B"}}
    assert store.pick_key("someone/Qwen3-8B", models) == "someone--qwen3-8b"


def test_a_fresh_key_is_just_the_stem(store):
    assert store.pick_key("Qwen/Qwen3-8B", {}) == "qwen3-8b"


# ── Only whole runs get in ──────────────────────────────────────────────────

def test_a_complete_payload_is_storable(store, tmp_path):
    assert store.check_payload(_payload(tmp_path)) == ""


@pytest.mark.parametrize("drop", ["svg_source", "python_source", "svg_png", "python_png"])
def test_a_missing_half_is_refused(store, tmp_path, drop):
    p = _payload(tmp_path)
    p[drop] = ""
    assert drop in store.check_payload(p)


def test_an_empty_render_is_refused(store, tmp_path):
    p = _payload(tmp_path)
    Path(p["python_png"]).write_bytes(b"")
    assert "python_png" in store.check_payload(p)


# ── What lands on disk ──────────────────────────────────────────────────────

def test_the_mirror_gets_both_rasters_and_both_sources(store, tmp_path):
    mirror = tmp_path / "mirror"
    payload = _payload(tmp_path)
    manifest = {"models": {}}
    entry = store.build_entry(payload, "qwen3-8b", {})
    sizes = store.write_mirror(mirror, "qwen3-8b", entry, payload, manifest)

    for rel in ("assets/qwen3-8b-svg.jpg", "assets/qwen3-8b-python.jpg",
                "assets/qwen3-8b.svg", "assets/qwen3-8b.py"):
        assert (mirror / rel).is_file(), rel
        assert sizes[rel] > 0
    # Rasters are JPEG, matching upstream and the dataset README.
    assert (mirror / "assets/qwen3-8b-svg.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert (mirror / "assets/qwen3-8b.py").read_text() == payload["python_source"]

    on_disk = json.loads((mirror / "manifest.json").read_text())
    assert on_disk["models"]["qwen3-8b"]["model_id"] == "Qwen/Qwen3-8B"


def test_the_entry_keeps_upstream_field_names(store, tmp_path):
    entry = store.build_entry(_payload(tmp_path), "qwen3-8b", {})
    for field in ("display_name", "svg", "python_render", "svg_source",
                  "python_source", "_first_seen"):
        assert entry[field], field
    # And the bits upstream has no room for.
    assert entry["loader"] == "exl3"
    assert entry["engine"] == "0.0.43"
    assert entry["run_date"] == "2026-08-09T12:00:00Z"


def test_a_re_run_keeps_the_original_first_seen(store, tmp_path):
    prev = {"_first_seen": "2026-01-01T00:00:00Z"}
    entry = store.build_entry(_payload(tmp_path), "qwen3-8b", prev)
    assert entry["_first_seen"] == "2026-01-01T00:00:00Z"
    assert entry["run_date"] != entry["_first_seen"]


def test_an_oversized_source_is_truncated_not_stored_whole(store, tmp_path):
    payload = _payload(tmp_path)
    payload["python_source"] = "#" * (store.MAX_SOURCE_BYTES + 5000)
    entry = store.build_entry(payload, "k", {})
    sizes = store.write_mirror(tmp_path / "m", "k", entry, payload, {"models": {}})
    assert sizes["assets/k.py"] == store.MAX_SOURCE_BYTES


# ── Nothing reaches HuggingFace unless a repo is named ──────────────────────

def test_no_repo_means_no_upload(store, tmp_path, monkeypatch, capsys):
    payload = _payload(tmp_path)
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(store, "push", lambda *a, **k: pytest.fail("pushed with no repo"))
    monkeypatch.setattr(sys, "argv", ["catbench_store.py", "--payload", str(path),
                                      "--mirror", str(tmp_path / "mirror")])
    assert store.main() == 0
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["ok"] and receipt["uploaded"] is False
    assert receipt["key"] == "qwen3-8b"


def test_dry_run_writes_the_mirror_but_pushes_nothing(store, tmp_path, monkeypatch, capsys):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(_payload(tmp_path)))
    monkeypatch.setattr(store, "push", lambda *a, **k: pytest.fail("pushed on a dry run"))
    monkeypatch.setattr(store, "remote_manifest", lambda repo, token: None)
    monkeypatch.setattr(sys, "argv", ["catbench_store.py", "--payload", str(path),
                                      "--mirror", str(tmp_path / "mirror"),
                                      "--repo", "Honkware/catbench-results", "--dry-run"])
    assert store.main() == 0
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["dry_run"] and receipt["uploaded"] is False
    assert (tmp_path / "mirror" / "manifest.json").is_file()
    assert set(receipt["files"]) == {
        "assets/qwen3-8b-svg.jpg", "assets/qwen3-8b-python.jpg",
        "assets/qwen3-8b.svg", "assets/qwen3-8b.py", "manifest.json",
    }


def test_a_run_that_did_not_finish_is_not_stored(store, tmp_path, capsys):
    payload = _payload(tmp_path)
    payload["python_source"] = ""
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    import sys as _sys
    argv = ["catbench_store.py", "--payload", str(path), "--mirror", str(tmp_path / "mirror")]
    old, _sys.argv = _sys.argv, argv
    try:
        assert store.main() == 1
    finally:
        _sys.argv = old
    assert not (tmp_path / "mirror").exists()
    assert json.loads(capsys.readouterr().out.strip())["ok"] is False


def test_a_dataset_we_cannot_read_is_not_overwritten(store, tmp_path, monkeypatch, capsys):
    """A failed read must not turn into a push that truncates the dataset."""
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(_payload(tmp_path)))

    def boom(repo, token):
        raise RuntimeError("could not read manifest.json: 503")

    monkeypatch.setattr(store, "remote_manifest", boom)
    monkeypatch.setattr(store, "push", lambda *a, **k: pytest.fail("pushed over an unread remote"))
    monkeypatch.setattr(sys, "argv", ["catbench_store.py", "--payload", str(path),
                                      "--mirror", str(tmp_path / "mirror"),
                                      "--repo", "Honkware/catbench-results",
                                      "--hf-token", "hf_test"])
    assert store.main() == 0
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    # Cached locally so the bot does not re-rent a pod, but not claimed as pushed.
    assert receipt["ok"] and receipt["uploaded"] is False and "503" in receipt["note"]
    assert (tmp_path / "mirror" / "assets" / "qwen3-8b-svg.jpg").is_file()


def test_an_entry_the_dataset_missed_is_pushed_on_the_next_run(store, tmp_path, monkeypatch, capsys):
    mirror = tmp_path / "mirror"
    # An earlier run that landed on disk but never reached HF.
    earlier = _payload(tmp_path, "org/Earlier")
    entry = store.build_entry(earlier, "earlier", {})
    store.write_mirror(mirror, "earlier", entry, earlier, {"models": {}})

    pushed = {}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(_payload(tmp_path)))
    monkeypatch.setattr(store, "remote_manifest", lambda repo, token: {"models": {}})
    monkeypatch.setattr(store, "push",
                        lambda repo, token, m, files, msg: pushed.update(files=files))
    monkeypatch.setattr(sys, "argv", ["catbench_store.py", "--payload", str(path),
                                      "--mirror", str(mirror),
                                      "--repo", "Honkware/catbench-results",
                                      "--hf-token", "hf_test"])
    assert store.main() == 0
    assert "assets/earlier-svg.jpg" in pushed["files"]
    assert "assets/earlier.py" in pushed["files"]
    models = json.loads((mirror / "manifest.json").read_text())["models"]
    assert set(models) == {"earlier", "qwen3-8b"}


def test_a_manifest_never_points_at_an_asset_we_did_not_push(store, tmp_path, monkeypatch, capsys):
    mirror = tmp_path / "mirror"
    (mirror / "assets").mkdir(parents=True)
    # An entry whose assets are gone: it cannot be re-uploaded, so it must not
    # ride along into the pushed manifest either.
    (mirror / "manifest.json").write_text(json.dumps({"models": {"ghost": {
        "model_id": "org/Ghost", "svg": "assets/ghost-svg.jpg",
        "python_render": "assets/ghost-python.jpg",
        "svg_source": "assets/ghost.svg", "python_source": "assets/ghost.py"}}}))

    pushed = {}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(_payload(tmp_path)))
    monkeypatch.setattr(store, "remote_manifest", lambda repo, token: {"models": {}})
    monkeypatch.setattr(store, "push",
                        lambda repo, token, m, files, msg: pushed.update(files=files))
    monkeypatch.setattr(sys, "argv", ["catbench_store.py", "--payload", str(path),
                                      "--mirror", str(mirror), "--repo", "r/r",
                                      "--hf-token", "hf_test"])
    assert store.main() == 0
    assert not any("ghost" in f for f in pushed["files"])
    assert "ghost" not in json.loads((mirror / "manifest.json").read_text())["models"]


def test_remote_entries_survive_a_local_write(store, tmp_path, monkeypatch, capsys):
    """A rebuilt box has an empty mirror; that must not truncate the dataset."""
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(_payload(tmp_path)))
    monkeypatch.setattr(store, "remote_manifest", lambda repo, token: {
        "models": {"kimi-k3": {"model_id": "moonshot/Kimi-K3", "svg": "assets/kimi-k3-svg.jpg"}}
    })
    monkeypatch.setattr(store, "push", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["catbench_store.py", "--payload", str(path),
                                      "--mirror", str(tmp_path / "mirror"),
                                      "--repo", "Honkware/catbench-results",
                                      "--hf-token", "hf_test"])
    assert store.main() == 0
    models = json.loads((tmp_path / "mirror" / "manifest.json").read_text())["models"]
    assert set(models) == {"kimi-k3", "qwen3-8b"}
