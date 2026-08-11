#!/usr/bin/env python3
"""Persist one CatBench run: local mirror, then the HuggingFace dataset.

The bot hands this a payload describing a run that already passed grading, and
it does the two writes that make the run permanent:

  1. ``data/catbench/`` on the VPS: manifest.json plus the four assets. Read
     back with no network, so a repeat lookup costs nothing.
  2. ``<owner>/catbench-results`` on HuggingFace, when --repo is given. That is
     the durable copy: it survives the box, and the images get a URL Discord
     embeds the way it embeds upstream's.

Layout matches the dataset README, which matches upstream's manifest shape::

    manifest.json                 index of every entry
    assets/<key>-svg.jpg          the SVG, rasterized
    assets/<key>-python.jpg       what the matplotlib script drew
    assets/<key>.svg              raw SVG the model emitted
    assets/<key>.py               raw script the model emitted

Runs on the VPS, never on a pod. The pod hands its artifacts back through the
controller's result JSON and never sees HF_TOKEN after the download step, so
persistence costs the sandbox nothing.

Usage::

    python backend/scripts/catbench_store.py --payload run.json \\
        --mirror data/catbench --repo Honkware/catbench-results

    python backend/scripts/catbench_store.py --payload run.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

# The rasters are line art and flat fill, so quality buys little; the cap is
# there because one 1.8 MB kitten in the dataset is one too many.
JPEG_QUALITY = 90
MAX_EDGE_PX = 1280
# A source longer than this is not a kitten, it is a model that never stopped.
MAX_SOURCE_BYTES = 256 * 1024


def norm_key(stem: str) -> str:
    """Upstream's key rule (demos/CatBench/index.html:normKey), and ours.

    Kept byte-identical to normKey in src/services/catbench.js: a key that
    disagrees across the two stores is a key that misses.
    """
    return re.sub(r"[_\s]+", "-", str(stem).lower())


def model_key(model_id: str) -> str:
    return norm_key(str(model_id).split("/")[-1])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def pick_key(model_id: str, models: dict) -> str:
    """The manifest key for this model, avoiding a stem collision.

    Two orgs can publish the same stem, and upstream's key format cannot tell
    them apart. First one in keeps the plain key; the next gets its owner
    folded in, so a second org's kitten never overwrites the first's.
    """
    base = model_key(model_id)
    if models.get(base, {}).get("model_id", model_id) == model_id:
        return base
    owner = norm_key(str(model_id).split("/")[0]) if "/" in model_id else "x"
    key = f"{owner}--{base}"
    n = 2
    while models.get(key, {}).get("model_id", model_id) != model_id:
        key = f"{owner}--{base}-{n}"
        n += 1
    return key


def to_jpeg(png_path: Path, out_path: Path) -> int:
    """Flatten a PNG onto white and write a JPEG. Returns bytes written.

    The dataset stores jpg because upstream does and the raw sources are kept
    alongside anyway: the raster is the thumbnail, the source is the evidence.
    """
    from PIL import Image

    with Image.open(png_path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, "white")
            flat.paste(im, mask=im.split()[-1])
            im = flat
        else:
            im = im.convert("RGB")
        long_edge = max(im.size)
        if long_edge > MAX_EDGE_PX:
            scale = MAX_EDGE_PX / long_edge
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                           Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    return out_path.stat().st_size


def check_payload(p: dict) -> str:
    """Why this payload is not a storable result, or "" when it is.

    Grading happens in the bot, before a pod's output ever reaches here; this
    is the second lock on the same door. A half-run must not enter the store,
    because a cached half-run is a run nobody will ever retry.
    """
    if not str(p.get("model_id", "")).strip():
        return "no model_id"
    for field in ("svg_source", "python_source"):
        if not str(p.get(field) or "").strip():
            return f"no {field}"
    for field in ("svg_png", "python_png"):
        path = p.get(field)
        if not path or not Path(path).is_file() or not Path(path).stat().st_size:
            return f"no {field}"
    return ""


def load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"models": {}}
    if not isinstance(data.get("models"), dict):
        data["models"] = {}
    return data


def remote_manifest(repo: str, token: str) -> dict | None:
    """The dataset's current manifest, or None when it has none yet.

    Read before every write so two bots, or a rebuilt box with an empty mirror,
    merge instead of truncating each other's entries.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(repo, "manifest.json", repo_type="dataset",
                               token=token or None, force_download=True)
    except EntryNotFoundError:
        return None
    except Exception as e:
        raise RuntimeError(f"could not read {repo}/manifest.json: {e}") from e
    return load_manifest(Path(path))


def build_entry(p: dict, key: str, prev: dict) -> dict:
    """One manifest entry: upstream's fields, then what upstream has no room for.

    display_name / svg / python_render / svg_source / python_source / _first_seen
    are upstream's, spelled the way upstream spells them, so this manifest can
    be read by the same parser and contributed back without a translation step.
    The rest is what upstream does not record and we need: which repo this
    actually was, which loader read it, at what version, when, and asked what.
    """
    model_id = p["model_id"]
    return {
        "display_name": p.get("display_name") or str(model_id).split("/")[-1],
        "svg": f"assets/{key}-svg.jpg",
        "python_render": f"assets/{key}-python.jpg",
        "svg_source": f"assets/{key}.svg",
        "python_source": f"assets/{key}.py",
        "_first_seen": prev.get("_first_seen") or _now(),
        "model_id": model_id,
        "loader": p.get("loader") or "",
        "engine": p.get("engine") or "",
        "format": p.get("format") or "",
        "run_date": p.get("run_date") or _now(),
        "prompts": p.get("prompts") or {},
        # Whether CatBench's own renderer got a picture out of this script, as
        # answered on the pod. Recorded because contributing the run upstream
        # depends on it and re-asking would cost another pod. Absent on entries
        # written before the check existed, which is not the same as False.
        **({"upstream_render_ok": bool(p["upstream_render_ok"])}
           if p.get("upstream_render_ok") is not None else {}),
    }


def write_mirror(mirror: Path, key: str, entry: dict, p: dict, manifest: dict) -> dict:
    """Assets and manifest onto local disk. Returns {path_in_repo: bytes}."""
    assets = mirror / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}

    sizes[entry["svg"]] = to_jpeg(Path(p["svg_png"]), mirror / entry["svg"])
    sizes[entry["python_render"]] = to_jpeg(Path(p["python_png"]), mirror / entry["python_render"])
    for field, text in (("svg_source", p["svg_source"]), ("python_source", p["python_source"])):
        blob = str(text).encode("utf-8")[:MAX_SOURCE_BYTES]
        (mirror / entry[field]).write_bytes(blob)
        sizes[entry[field]] = len(blob)

    manifest["models"][key] = entry
    manifest["generated"] = _now()
    manifest.setdefault("source", "blockquant /catbench")
    body = json.dumps(manifest, indent=1, sort_keys=False).encode("utf-8")
    tmp = mirror / "manifest.json.tmp"
    tmp.write_bytes(body)
    tmp.replace(mirror / "manifest.json")
    sizes["manifest.json"] = len(body)
    return sizes


def push(repo: str, token: str, mirror: Path, paths: list[str], message: str) -> None:
    """One commit for the whole entry, so the manifest never points at an
    asset that has not landed yet."""
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(mirror / rel))
           for rel in paths]
    HfApi(token=token).create_commit(
        repo_id=repo, repo_type="dataset", operations=ops, commit_message=message,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Persist one CatBench run")
    ap.add_argument("--payload", required=True, help="JSON describing the run")
    ap.add_argument("--mirror", default="data/catbench", help="Local mirror directory")
    ap.add_argument("--repo", default="", help="HF dataset repo id; empty = mirror only")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write the mirror and print the commit, upload nothing")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    bad = check_payload(payload)
    if bad:
        print(json.dumps({"ok": False, "error": f"not a storable result: {bad}"}))
        return 1

    mirror = Path(args.mirror).expanduser().resolve()
    manifest = load_manifest(mirror / "manifest.json")
    catchup: list[str] = []
    note = ""
    can_push = bool(args.repo)

    if args.repo:
        try:
            remote = remote_manifest(args.repo, args.hf_token)
        except RuntimeError as e:
            # A dataset we cannot read is a dataset we must not overwrite. Keep
            # the mirror write so the bot still caches; the entry gets pushed by
            # the catch-up below on the next run that does reach HF.
            remote, can_push, note = None, False, str(e)
        if remote is not None:
            # Remote is the record. A local entry it has never seen only rejoins
            # if its assets are still on disk, so a pushed manifest can never
            # point at a file that was never pushed.
            for key, prev in manifest["models"].items():
                if key in remote["models"]:
                    continue
                rels = [prev.get(f) for f in ("svg", "python_render", "svg_source", "python_source")]
                if not all(r and (mirror / r).is_file() for r in rels):
                    continue
                remote["models"][key] = prev
                catchup += rels
            manifest = remote

    key = pick_key(payload["model_id"], manifest["models"])
    entry = build_entry(payload, key, manifest["models"].get(key, {}))
    sizes = write_mirror(mirror, key, entry, payload, manifest)
    files = [entry["svg"], entry["python_render"], entry["svg_source"],
             entry["python_source"], *catchup, "manifest.json"]

    uploaded = False
    if can_push and not args.dry_run:
        if not args.hf_token:
            print(json.dumps({"ok": False, "error": "HF_TOKEN is required to push"}))
            return 1
        push(args.repo, args.hf_token, mirror, files, f"catbench: {payload['model_id']}")
        uploaded = True

    print(json.dumps({
        "ok": True, "key": key, "entry": entry, "files": files, "note": note,
        "bytes": sizes, "total_bytes": sum(sizes.values()),
        "repo": args.repo, "uploaded": uploaded, "dry_run": bool(args.dry_run),
        "mirror": str(mirror),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
