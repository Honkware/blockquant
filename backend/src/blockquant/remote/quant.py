#!/usr/bin/env python3
"""Remote quantization entrypoint — runs inside the RunPod pod.

Reads ``/root/bq-config.json``, downloads the source weights from
HuggingFace, runs ExLlamaV3's ``convert.py`` against the requested
bits-per-weight, optionally uploads the results back to HuggingFace,
then writes a final summary to ``/root/bq-result.json`` for the local
poller to pick up.

This file is shipped two ways:
  1. Baked into the prebuilt Docker image at ``/opt/blockquant/quant.py``
     (production path — see ``docker/Dockerfile.runpod``).
  2. SFTP'd to ``/root/quant.py`` by ``RunPodProvider.run_pipeline`` for
     pods running a non-baked base image (fallback path).

The two paths produce identical behaviour. Tests live in
``backend/tests/remote/test_quant.py``.
"""
from __future__ import annotations

import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

CONFIG_PATH = "/root/bq-config.json"
RESULT_PATH = "/root/bq-result.json"

# cards.py + card_template.md are shipped next to this script (SFTP'd by the
# provider or baked into the image), so make the script's own dir importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _dir_size_gb(path: Path) -> float | None:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / 1e9 if total else None
    except OSError:
        return None


def emit_result(payload: dict) -> None:
    """Write the final result JSON for the local poller."""
    try:
        with open(RESULT_PATH, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[fatal] could not write result: {e}", flush=True)


def _arm_self_terminate_backstop(pod_id: str, api_key: str, grace_seconds: float) -> None:
    """Spawn a detached process that terminates the pod after a grace window.

    This lets quant.py exit right after [done] so a live controller can drain
    the log, fetch the result, and terminate the pod itself first. The backstop
    fires only if that has not happened within the grace window (controller
    died), so it cannot orphan the pod and does not race a live controller.

    Uses urllib + the v1 REST DELETE with Bearer auth; the legacy GraphQL
    endpoint returns 403 for rpa_ keys. No dependency on the runpod SDK.
    """
    import subprocess
    # Pass pod id + key via env, NOT argv -- argv shows up in `ps`/proc/cmdline.
    code = (
        "import os, time, urllib.request as u;"
        f"time.sleep({float(grace_seconds)});"
        "u.urlopen(u.Request('https://rest.runpod.io/v1/pods/' + os.environ['BQ_POD'], "
        "method='DELETE', headers={'Authorization': 'Bearer ' + os.environ['BQ_KEY']}), timeout=20)"
    )
    subprocess.Popen(
        [sys.executable, "-c", code],
        env={**os.environ, "BQ_POD": pod_id, "BQ_KEY": api_key},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


_CLIP_MEAN_STD = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
_HALF_MEAN_STD = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

# What each vision family's own preprocessor_config.json says, keyed by the
# vision_config->model_type prefix. The pixel budgets are upstream's, restated
# as merged-token counts (pixels / (patch * merge)^2) so they hold for a model
# whose patch size differs: Qwen2-VL ships 3136/12845056 at factor 28,
# Qwen3-VL 65536/16777216 at factor 32, GLM-4V 12544/9633792 at factor 28.
# (processor_type, mean/std, min_tokens, max_tokens)
_VL_PREP_FAMILIES = (
    ("qwen3_5", ("Qwen2VLImageProcessorFast", _HALF_MEAN_STD, 64, 16384)),
    ("qwen3_vl", ("Qwen2VLImageProcessorFast", _HALF_MEAN_STD, 64, 16384)),
    ("glm4v", ("Glm4vImageProcessor", _CLIP_MEAN_STD, 16, 12288)),
    ("qwen2", ("Qwen2VLImageProcessorFast", _CLIP_MEAN_STD, 4, 16384)),
)
_VL_PREP_FALLBACK = _VL_PREP_FAMILIES[-1][1]


def _vision_preprocessor_config(model_dir: Path) -> None:
    """Give a VL source dir the preprocessor_config.json exllamav3 requires.

    A config.json with a vision_config makes exllamav3 open the file
    unconditionally -- there is no language-model-only path anywhere in it --
    so a merge that dropped the file is unloadable, FileNotFoundError before
    the first shard. convert.py copies every non-tensor file into the output,
    so whatever is here is what the published quant ships.

    It has to be *right*, not merely present. The 56px stub we used to write
    loaded fine and then broke every image: size->shortest/longest_edge are
    read as min/max_pixels, total pixel counts rather than edge lengths, and a
    56-pixel ceiling floors smart_resize to a zero-size grid ("height and width
    must be > 0"). Prefer the model's own numbers out of processor_config.json,
    which HF-style VL repos carry alongside; otherwise derive them from
    vision_config.
    """
    prep = model_dir / "preprocessor_config.json"
    if prep.exists():
        return
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    vis = cfg.get("vision_config")
    if not isinstance(vis, dict):
        return

    own = _image_processor_block(model_dir)
    if own:
        prep.write_text(json.dumps(own, indent=2))
        print("[preprocess] wrote preprocessor_config.json from the model's own "
              "processor_config.json", flush=True)
        return

    model_type = str(vis.get("model_type") or cfg.get("model_type") or "")
    proc_type, (mean, std), min_tok, max_tok = next(
        (v for k, v in _VL_PREP_FAMILIES if model_type.startswith(k)), _VL_PREP_FALLBACK
    )
    patch = int(vis.get("patch_size", 14))
    merge = int(vis.get("spatial_merge_size", 2))
    factor = patch * merge
    prep.write_text(json.dumps({
        # Pixel counts, not edge lengths. exllamav3 reads them as min/max_pixels.
        "size": {"shortest_edge": min_tok * factor ** 2,
                 "longest_edge": max_tok * factor ** 2},
        "patch_size": patch,
        "temporal_patch_size": int(vis.get("temporal_patch_size", 2)),
        "merge_size": merge,
        "image_mean": mean,
        "image_std": std,
        "image_processor_type": proc_type,
    }, indent=2))
    print(f"[preprocess] no preprocessor_config.json; derived one for {model_type or 'vl'} "
          f"(patch {patch}, merge {merge})", flush=True)


def _image_processor_block(model_dir: Path) -> dict | None:
    """The image_processor half of a processor_config.json, if it is complete.

    exllamav3 only looks in processor_config.json for Mistral3; every other
    arch wants the split-out file, so hoist the block across when we have it.
    """
    try:
        blk = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    blk = blk.get("image_processor")
    if not isinstance(blk, dict):
        return None
    need = ("size", "patch_size", "temporal_patch_size", "merge_size",
            "image_mean", "image_std", "image_processor_type")
    if not all(k in blk for k in need):
        return None
    size = blk["size"]
    if not (isinstance(size, dict) and "shortest_edge" in size and "longest_edge" in size):
        return None
    return blk


def _ensure_fast_tokenizer(model_dir: Path) -> None:
    """exllamav3's Tokenizer loads tokenizer.json (the fast format). Some models
    ship only the legacy vocab.json + merges.txt (e.g. LocateAnything's Qwen2
    tokenizer), so build tokenizer.json from them via transformers when absent.
    """
    if (model_dir / "tokenizer.json").exists():
        return
    from transformers import AutoTokenizer
    last = None
    for kw in ({"use_fast": True}, {"use_fast": True, "trust_remote_code": True}):
        try:
            tok = AutoTokenizer.from_pretrained(str(model_dir), **kw)
            tok.save_pretrained(str(model_dir))
            print("[tokenizer] built tokenizer.json from legacy vocab/merges", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            last = e
    print(f"[tokenizer] WARN could not build tokenizer.json: {last}", flush=True)


# Config fields exllamav3 reads as ints. Fine-tunes/merges sometimes emit them
# as floats (e.g. "original_max_position_embeddings": 16384.0), and a float where
# an int is expected makes ext.rope() raise "incompatible function arguments"
# (this killed pagestorm-14b / ministral3 -- diagnosed by turboderp).
_INT_CFG_FIELDS = {
    "original_max_position_embeddings", "max_position_embeddings",
    "head_dim", "hidden_size", "intermediate_size", "moe_intermediate_size",
    "shared_expert_intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "num_experts", "num_experts_per_tok", "num_local_experts",
    "sliding_window", "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id",
}


def _sanitize_config(model_dir: Path) -> None:
    """Coerce known integer config fields that ship as floats back to ints, incl.
    nested configs (llama_4_scaling, rope_parameters, text_config, ...). Only
    rewrites integral floats of allow-listed int fields, so genuine floats
    (rope_theta, scaling factors) are untouched."""
    cfg_path = model_dir / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return
    fixed = []

    def walk(o, pfx=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in _INT_CFG_FIELDS and isinstance(v, float) and v == int(v):
                    o[k] = int(v)
                    fixed.append(f"{pfx}{k}: {v} -> {int(v)}")
                else:
                    walk(v, f"{pfx}{k}.")
        elif isinstance(o, list):
            for x in o:
                walk(x, pfx)

    walk(cfg)
    if fixed:
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[config] coerced float int-fields: {', '.join(fixed)}", flush=True)


def _model_has_tensor_prefix(model_dir: Path, prefix: str) -> bool:
    """True if any model tensor key starts with `prefix`. Reads the safetensors
    index when present, else scans each .safetensors header. Fails OPEN (True) so
    we never wrongly strip something on an unreadable model."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        try:
            return any(k.startswith(prefix) for k in json.loads(idx.read_text()).get("weight_map", {}))
        except Exception:
            return True
    import struct
    files = list(model_dir.glob("*.safetensors"))
    if not files:
        return True
    for sf in files:
        try:
            with open(sf, "rb") as f:
                hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            if any(k.startswith(prefix) for k in hdr):
                return True
        except Exception:
            return True
    return False


def _disable_missing_mtp(model_dir: Path) -> None:
    """Qwen3.5 declares an MTP draft head (mtp_num_hidden_layers > 0), and
    exllamav3 quantizes it -- but most fine-tunes ship no mtp.* weights, so the
    convert dies on 'Required tensor mtp....weight not found'. If the weights
    aren't there, zero mtp_num_hidden_layers so exllamav3 skips MTP and quantizes
    the base model. (Diagnosed on Qwythos-9B.)"""
    cfg_path = model_dir / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if _model_has_tensor_prefix(model_dir, "mtp"):
        return
    fixed = []

    def walk(o, pfx=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "mtp_num_hidden_layers" and isinstance(v, int) and v > 0:
                    o[k] = 0
                    fixed.append(f"{pfx}{k}: {v} -> 0")
                else:
                    walk(v, f"{pfx}{k}.")

    walk(cfg)
    if fixed:
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[config] no mtp.* weights present; disabled MTP: {', '.join(fixed)}", flush=True)


def _rfn_count(rfn_path: Path) -> int:
    """How many tensors sc_rfn_probe found, or 0 if unreadable."""
    try:
        d = json.loads(rfn_path.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            # Key presence, not truthiness: an empty results list means the
            # probe found nothing, which is 0 -- an `or` chain falls through it
            # and returns the length of the wrapper dict instead.
            for k in ("results", "tensors"):
                if k in d:
                    return len(d[k])
        return len(d)
    except Exception:
        return 0


def _measured_all(measure_path: Path, rfn_path: Path) -> bool:
    """True once sc_measure has a result for EVERY tensor the probe found.

    Non-empty is not enough. sc_measure streams results and resumes from a
    partial file on its own, so a crash at tensor 50 of 151 leaves a file that
    reads as finished and hands sc_optimize a measurement covering a third of
    the model -- which it will happily build a recipe from. The probe walks the
    same module tree immediately beforehand, so its entry count is the expected
    total: 151 and 151 on the first real run.

    Unreadable rfn.json means no expected count, so fall back to re-running the
    stage rather than trusting a number we do not have.
    """
    exp = _rfn_count(rfn_path)
    if not exp:
        return False
    try:
        res = json.loads(measure_path.read_text(encoding="utf-8")).get("results") or []
    except Exception:
        return False
    return len(res) >= exp


def _heartbeat(proc, name: str, every: float = 30.0) -> list[str]:
    """Report a running stage into the log, and return its last lines.

    These stages used to run under capture_output, which meant one line at the
    start and silence until they exited -- and the controller's stall clock
    advances on the progress text CHANGING, so a stage outliving stall_timeout
    looked exactly like a hung pod and got a working one terminated.

    Not a straight passthrough: these are turboderp's scripts and they write
    tqdm bars, which are carriage returns rather than lines, so this reads raw
    and splits on both. Every fragment would be the flood the progress filter
    exists to keep out of the log, so it emits the newest one every `every`
    seconds. The [sc] prefix is what gets it past _PROGRESS_MARKERS.
    """
    keep: deque[str] = deque(maxlen=12)
    buf = ""
    start = last = time.monotonic()
    fd = proc.stdout.fileno()
    while True:
        # Wake on output OR on the clock. Reading straight off the pipe blocks
        # until the child writes, so a stage that goes quiet -- the only case
        # that actually looks like a hung pod -- would report nothing at all.
        # That was the first version of this, and it emitted one line in three
        # minutes while sc_trace loaded a model.
        ready, _, _ = select.select([fd], [], [], max(0.0, every - (time.monotonic() - last)))
        if ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace")
            parts = re.split(r"[\r\n]", buf)
            buf = parts.pop()
            for frag in parts:
                if frag.strip():
                    keep.append(frag.strip())
        now = time.monotonic()
        if now - last >= every:
            # The live state of a tqdm bar is the UNTERMINATED fragment: it only
            # gets a \r when the next update lands, so reporting completed ones
            # meant reporting the previous update, or nothing on a stage sitting
            # on a single bar.
            cur = buf.strip() or (keep[-1] if keep else "working")
            # Elapsed is not decoration. get_progress is `grep | tail`, so a
            # repeated identical line leaves the controller's progress text
            # unchanged and its stall clock frozen -- the same failure by a
            # different route. This guarantees every beat differs.
            print(f"[sc] {name}: {int(now - start)}s {cur[:150]}", flush=True)
            last = now
    if buf.strip():
        keep.append(buf.strip())
    return list(keep)


# The self-calibration artifacts depend on the model and the donor, NOT on the
# bitrate: trace/cal/rfn/measure are reused verbatim across every variant, and
# only sc_optimize runs per bitrate. Within one job quant.py already shares
# them. Across jobs they died with the pod, so asking for a 3.5bpw next week
# regenerated 42 minutes of identical work -- 25.5 for sc_trace and 16.3 for
# sc_measure, measured. They total ~10 MB and cal.safetensors is token ids, so
# the size barely moves with the model.
_SC_CACHE_FILES = ("trace.json", "cal.safetensors", "rfn.json", "measure.json")


def _sc_cache_repo(owner: str, model_name: str) -> str:
    """Deterministic, so a pod can ask for it before spending anything.

    A dataset repo, and keyed on the base model rather than any one quant: the
    artifacts outlive any particular bitrate, and filing them inside an SC repo
    would mean deleting that repo takes the cache with it.
    """
    return f"{owner}/{model_name}-exl3-selfcal"


def _sc_cache_key(model_id: str, model_rev: str, donor_repo: str,
                  cal_rows: int, cal_cols: int) -> dict:
    """Everything the artifacts actually depend on.

    Reused under the wrong key this silently optimizes a quant against another
    model's measurements, which no error would ever surface -- so the match is
    exact and anything unrecognised regenerates. exllamav3's version is in here
    because sc_measure's output is its format, not ours.
    """
    try:
        import exllamav3
        exl = getattr(exllamav3, "__version__", "?")
    except Exception:
        exl = "?"
    return {
        "model_id": model_id,
        "model_rev": model_rev or "",
        "donor_repo": donor_repo or "",
        "cal_rows": int(cal_rows),
        "cal_cols": int(cal_cols),
        "exllamav3": exl,
        "schema": 1,
    }


def _sc_cache_restore(api, repo_id: str, key: dict, work_root: Path) -> bool:
    """Pull a matching cache into work_root. True if every artifact landed.

    Nothing is changed in _run_sc_stages to use this: its stages already skip
    what is on disk, so restoring the files IS the reuse. A partial restore is
    no worse than none -- whatever is missing simply gets rebuilt.
    """
    from huggingface_hub import hf_hub_download
    try:
        man = hf_hub_download(repo_id=repo_id, filename="manifest.json",
                              repo_type="dataset", token=api.token)
        have = json.loads(Path(man).read_text(encoding="utf-8"))
    except Exception:
        return False
    if have != key:
        diff = [k for k in key if have.get(k) != key.get(k)]
        print(f"[sc] cache at {repo_id} does not match ({', '.join(diff)}); rebuilding",
              flush=True)
        return False
    work_root.mkdir(parents=True, exist_ok=True)
    got = 0
    for name in _SC_CACHE_FILES:
        try:
            src = hf_hub_download(repo_id=repo_id, filename=name,
                                  repo_type="dataset", token=api.token)
            shutil.copyfile(src, work_root / name)
            got += 1
        except Exception as e:
            print(f"[sc] cache miss for {name}: {type(e).__name__}", flush=True)
    if got:
        print(f"[sc] restored {got}/{len(_SC_CACHE_FILES)} artifacts from {repo_id}",
              flush=True)
    return got == len(_SC_CACHE_FILES)


def _sc_cache_save(api, repo_id: str, key: dict, work_root: Path) -> None:
    """Best-effort: the quant is what matters, the cache is an optimisation."""
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
        for name in _SC_CACHE_FILES:
            f = work_root / name
            if f.exists() and f.stat().st_size > 0:
                api.upload_file(path_or_fileobj=str(f), path_in_repo=name,
                                repo_id=repo_id, repo_type="dataset")
        # Manifest last: it is the thing a later run trusts, so it must not
        # appear before the files it vouches for.
        api.upload_file(path_or_fileobj=json.dumps(key, indent=2).encode(),
                        path_in_repo="manifest.json", repo_id=repo_id,
                        repo_type="dataset")
        print(f"[sc] cached calibration -> {repo_id}", flush=True)
    except Exception as e:
        print(f"[sc] WARN could not cache ({type(e).__name__}: {e})", flush=True)


def _run_sc_stages(model_dir: Path, donor_dir: Path, work_root: Path, bpw: float,
                   head_bits: int, cal_rows: int, cal_cols: int,
                   timings: dict | None = None) -> tuple[Path, Path]:
    """Self-calibration: produce (recipe.yaml, cal.safetensors) for one bitrate.

    Four of turboderp's scripts in order, each a subprocess so a crash in one
    is a stage failure rather than something that takes this process with it.
    Every coupling below cost a pod to discover -- see backend/scripts/
    sc_chain_smoke.py, which runs the same sequence small:

      - sc_measure's -tr takes sc_trace's -co (packed safetensors), NOT its -o
        (the qbench trace JSON). Hand it the JSON and it falls back to the
        bundled corpus without saying so, which optimizes the quant against the
        wrong distribution.
      - donor_dir must be a quant of THIS model. sc_rfn_probe walks both module
        trees together, and sc_measure runs the donor's trace through this
        model's embedding, so a foreign tokenizer indexes out of range.
      - cal_rows/cal_cols must match what the conversion will ask for. convert
        crops the file to its own --cal_rows x --cal_cols and refuses anything
        smaller, so the caller passes one pair to both.
      - A stage is judged on the artifact it leaves. sc_measure can write its
        output and still exit non-zero.
    """
    sc = _selfcal_dir()
    if sc is None:
        raise RuntimeError("vendored selfcal scripts not found on this pod")
    work_root.mkdir(parents=True, exist_ok=True)
    trace = work_root / "trace.json"
    cal = work_root / "cal.safetensors"
    rfn = work_root / "rfn.json"
    measure = work_root / "measure.json"
    recipe = work_root / f"recipe-{bpw}.yaml"

    def stage(name: str, args: list[str], produces: Path, done=None) -> None:
        # Resume: every one of these is expensive and all of them can be
        # re-entered, so a retried job does not redo what already landed.
        # `done` is for a stage that creates its file up front and fills it in
        # as it goes -- existence there means "started", not "finished".
        ok = done or (lambda p: p.stat().st_size > 0)
        if produces.exists() and ok(produces):
            print(f"[sc] {name} already done -> {produces.name}", flush=True)
            if timings is not None:
                timings[name] = 0.0   # reused, not run
            return
        _t0 = time.monotonic()
        print(f"[sc] {name} ...", flush=True)
        # -u, because these scripts print without flush=True and their stdout
        # here is a pipe: Python block-buffers it, so a quiet stage's output
        # sits in an 8K buffer for the whole run and the heartbeat has nothing
        # to report. sc_trace prints enough to keep filling the buffer and
        # looked fine; sc_measure printed "Reference pass" and went dark for
        # 13 minutes.
        proc = subprocess.Popen([sys.executable, "-u", str(sc / f"{name}.py"), *args],
                                cwd=str(sc), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        tail = _heartbeat(proc, name)
        rc = proc.wait()
        if not (produces.exists() and ok(produces)):
            raise RuntimeError(f"sc stage {name} left no usable {produces.name} "
                               f"(exit {rc}): " + " | ".join(tail))
        _el = time.monotonic() - _t0
        if timings is not None:
            timings[name] = round(_el, 1)
        # Timed because every estimate of this so far has been a guess, and each
        # one was wrong: the cost band multiplies a 35B baseline by 4, which no
        # measured SC run has ever justified. Recorded per run so the estimate
        # can be fitted from real jobs instead.
        print(f"[sc] {name} done -> {produces.name} ({_el / 60:.1f}m)", flush=True)

    stage("sc_trace", ["-m", str(donor_dir), "-o", str(trace), "-co", str(cal),
                       "-cr", str(cal_rows), "-cc", str(cal_cols)], cal)
    stage("sc_rfn_probe", ["-mq", str(donor_dir), "-mr", str(model_dir),
                           "-o", str(rfn)], rfn)
    # --streaming writes the header and an empty "results" before it measures
    # anything, so the file exists seconds in and stays that way for the whole
    # stage. On size alone a resumed job skips the measurement entirely and a
    # crash halfway reads as success -- either way sc_optimize builds a recipe
    # from nothing and the quant is silently optimized against it.
    stage("sc_measure", ["-m", str(model_dir), "-o", str(measure), "--streaming",
                         "-tr", str(cal)], measure,
          done=lambda p: _measured_all(p, rfn))
    stage("sc_optimize", ["-m", str(measure), "-b", str(bpw), "-hb", str(head_bits),
                          "-rr", str(rfn), "-o", str(recipe)], recipe)
    return recipe, cal


def _selfcal_dir() -> Path | None:
    """Where the vendored selfcal tree is, whichever layout this is running in.

    The pod flattens this file to /opt/blockquant/quant.py (baked, then
    overwritten with the current one) or /root/quant.py, so the repo's
    blockquant/remote/quant.py -> blockquant/selfcal relationship does not hold
    there. Getting it wrong is silent: the import fails into a best-effort
    handler and the number it was going to produce just never appears.
    """
    for cand in (Path(__file__).parent / "selfcal",
                 Path(__file__).parent.parent / "selfcal",
                 Path("/opt/blockquant/selfcal")):
        if (cand / "sc_measure.py").is_file():
            return cand
    return None


def _kl_kernel_usable() -> bool:
    """Whether exllamav3's CUDA compute_kl_div can be trusted on this image.

    It segfaulted on the builds we ran before 1.5.0, and a segfault takes the
    whole pod down mid-job rather than raising something we could catch. So the
    check runs in a subprocess on tiny tensors: a crash there costs a second and
    tells us to use the torch path instead.
    """
    import subprocess
    probe = (
        "import torch;"
        "from exllamav3.util.measures import compute_kl_div;"
        "a=torch.randn(4,128,device='cuda');b=torch.randn(4,128,device='cuda');"
        "r=compute_kl_div(a,b,128);"
        "assert torch.isfinite(r).all();"
        "print('ok')"
    )
    try:
        p = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           timeout=120, text=True)
    except Exception as e:
        print(f"[kl] kernel probe failed to run ({type(e).__name__}); using torch", flush=True)
        return False
    if p.returncode == 0 and "ok" in p.stdout:
        return True
    why = f"exit {p.returncode}" + (f", {p.stderr.strip().splitlines()[-1]}" if p.stderr.strip() else "")
    print(f"[kl] compute_kl_div unusable ({why}); using torch", flush=True)
    return False


def _kl_div_eval(quant_dir: Path, fp16_dir: Path, rows: int = 10,
                 seq_len: int = 2048) -> tuple[dict | None, str]:
    """KL(fp16 || quant) over held-out text, as qbench reports it.

    Returns (stats, method). stats carries kld, kld_median, the p10-p90 spread
    and buckets by reference confidence -- the shape turboderp's qbench emits,
    computed by his own DiffStats, so a number on our card can be read against
    his. The median and the high-confidence buckets are the ones worth quoting:
    per his own note, the mean is dominated by tokens where the reference itself
    is undecided, where any perturbation is amplified.

    Two passes, one model resident at a time: the fp16 writes reference logits
    and per-token confidence to disk, then the quant is streamed against them.
    That is qbench's own arrangement and it is what lets a 35B fp16 fit on an
    80GB card. Best-effort throughout: returns (None, "") so a finished quant
    still uploads without a number.
    """
    import shutil
    import tempfile
    try:
        import torch
        from exllamav3 import Config, Model, Cache, Tokenizer
        _sc = _selfcal_dir()
        if _sc is None:
            print("[kl] WARN vendored qbench not found; no KL this run", flush=True)
            return None, ""
        sys.path.insert(0, str(_sc))
        from eval.qbench.measure import DiffStats, save_reference_row, print_stats
        from eval.qbench.data import QCache, get_test_rows, save_tensors
    except Exception as e:
        print(f"[kl] WARN import failed: {type(e).__name__}: {e}", flush=True)
        return None, ""

    try:
        # qbench's own loader: wiki2 at 10x2048, which is what its example
        # project ships and the only corpus sc_measure implements. We measured
        # openwebtext at 8x8192 before, which is a supported source but nobody
        # else's geometry, so the number could not be read against anyone's.
        # It also applies the chat template (prepend_hf_chat_context) and
        # reports prefix_len, so metrics skip the framing rather than scoring it.
        project = {
            "test_data": {"source": "wiki2", "rows": rows, "length": seq_len,
                          "stride": seq_len},
            # The reference's tokenizer, as qbench's example does. Keying it on
            # the quant instead would re-tokenize per variant, since the cache
            # key is a hash of (dataset spec, tokenizer source).
            "tokenizer": {"source": str(fp16_dir), "template": True},
            "logit_cache": {"dir": str(fp16_dir.parent), "max_size_gb": 50},
        }
        corpus = "wiki2"
        qcache = QCache(project["logit_cache"])
        ids, ranges, trace_vocab = get_test_rows(project, qcache)
        tcfg = Config.from_directory(str(quant_dir))
        tokenizer = Tokenizer.from_config(tcfg)
        vocab = trace_vocab or tokenizer.actual_vocab_size
        method = f"qbench · {corpus} · {rows}×{seq_len}"
        seqs = [ids[i:i + 1, :] for i in range(ids.shape[0])]
    except Exception as e:
        print(f"[kl] WARN test data failed: {type(e).__name__}: {e}", flush=True)
        return None, ""

    # exllamav3 asserts the cache is a multiple of 256, and qbench's rows are
    # not: prepend_hf_chat_context puts a chat prefix in front of each 2048, so
    # the row is 2048 + however long that framing came out.
    row_len = ids.shape[-1]
    cache_len = -(-row_len // 256) * 256

    def _forward_rows(model_dir, on_row) -> None:
        config = Config.from_directory(str(model_dir))
        config.override_dynamic_seq_len(cache_len)
        model = Model.from_config(config)
        cache = Cache(model, max_num_tokens=cache_len)
        model.load()
        try:
            for i, seq in enumerate(seqs):
                # batch_shape is the cache's geometry, not the input's -- the
                # real length comes from input_ids -- and attn.py asserts the
                # seq len is a page multiple, so it takes the rounded value.
                params = {"attn_mode": "flash_attn", "cache": cache,
                          "past_len": 0, "batch_shape": (1, cache_len)}
                logits = model.forward(seq, params=params)
                on_row(i, logits)
                # A hybrid/linear-attn model takes a recurrent state slot per
                # forward: with no "recurrent_states" in params, exllamav3
                # allocates from the cache's pool and this fresh dict per row
                # means it never comes back. The pool is max_batch_size, 16 by
                # default, and kl_rows is 40 -- so row 17 died on "Cannot
                # create new state: no available slots" and the job published
                # with no KL number. Rows are independent (past_len 0), so
                # hand the slots back after each one.
                try:
                    cache.reset_states()
                except AttributeError:
                    pass
        finally:
            try:
                model.unload()
            except Exception:
                pass
            del model, cache, config
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    stage = Path(tempfile.mkdtemp(prefix="klstage-", dir=str(quant_dir.parent)))
    try:
        conf_rows: list = []
        _forward_rows(fp16_dir, lambda i, lg: save_reference_row(str(stage), i, lg, ranges[i], conf_rows))
        save_tensors(str(stage / "conf.safetensors"), {"conf": torch.cat(conf_rows)})

        stats = DiffStats(ids, ranges, vocab, str(stage))
        if _kl_kernel_usable():
            _forward_rows(quant_dir, lambda i, lg: stats(i, lg))
        else:
            # Same quantity, plain torch: compute_kl_div documents itself as
            # F.kl_div(log_softmax(input), softmax(target)).sum(-1), which is
            # what this is. Fills the same accumulators so results() is identical.
            from eval.qbench.data import load_tensor
            from exllamav3.util.measures import compute_target_log_probs

            def _cmp(r, logits):
                a, b = ranges[r]
                lg = logits[:, a:b, :].float()
                lg.clamp_(min=-200.0)
                tgt = ids[r, a + 1:b].view(1, -1).to(lg.device)
                lp = compute_target_log_probs(lg[:, :-1, :], tgt, min(vocab, lg.shape[-1]))
                fin = torch.isfinite(lp)
                stats.logprob_sum += lp[fin].sum().item()
                stats.logprob_count += fin.sum().item()
                stats.total_count += tgt.numel()
                ref = load_tensor(str(stage / f"row_{r:06d}.safetensors"), "logits")
                ref = ref.to(lg.device).float()
                kv = min(vocab, lg.shape[-1], ref.shape[-1])
                qi, fi = lg.squeeze(0)[..., :kv], ref.squeeze(0)[..., :kv]
                kl = (torch.softmax(fi, dim=-1)
                      * (torch.log_softmax(fi, dim=-1) - torch.log_softmax(qi, dim=-1))
                      ).sum(-1)
                stats.kl_toks.append(kl.flatten().float().cpu())
                del ref

            _forward_rows(quant_dir, _cmp)

        res = stats.results()
        if "kld" not in res:
            return None, ""
        print_stats(f"{quant_dir.name}", res)
        return res, method
    except Exception as e:
        print(f"[kl] WARN eval failed: {type(e).__name__}: {e}", flush=True)
        return None, ""
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _sample_generate(quant_dir: Path, prompt: str, max_new_tokens: int = 256) -> str | None:
    """Run a single prompt through the freshly quantized model for the card/embed.

    Loads the quant (small, fits easily) and generates one greedy completion so
    the requester sees how this bpw actually responds. Applies the model's chat
    template when it has one, else treats the prompt as raw text. Best-effort:
    returns None on any failure so a finished quant still uploads.
    """
    # "Draw an SVG" prompts (the catbench) need room to emit the whole drawing;
    # short Q&A still stops early on the eos token, so a bigger cap is free there.
    if "svg" in prompt.lower():
        max_new_tokens = max(max_new_tokens, 2048)
    try:
        import torch
        from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler
    except Exception as e:
        print(f"[sample] WARN import failed: {type(e).__name__}: {e}", flush=True)
        return None

    model = None
    try:
        # Chat-template the prompt with transformers when the model ships one,
        # so instruct models reply in-character; base models get the raw prompt.
        # Collect stop tokens too, so generation ends at the turn boundary
        # instead of running the full budget and rambling past <|im_end|>.
        text, special = prompt, False
        stop: list = []
        end_strs = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>")
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(str(quant_dir))
            if getattr(hf_tok, "chat_template", None):
                text = hf_tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False,
                )
                special = True
            if hf_tok.eos_token_id is not None:
                stop.append(hf_tok.eos_token_id)
            for t in end_strs:
                tid = hf_tok.convert_tokens_to_ids(t)
                if isinstance(tid, int) and tid >= 0 and tid != hf_tok.unk_token_id:
                    stop.append(tid)
        except Exception:
            text, special = prompt, False

        config = Config.from_directory(str(quant_dir))
        model = Model.from_config(config)
        cache = Cache(model, max_num_tokens=4096)
        model.load()
        tokenizer = Tokenizer.from_config(config)
        gen = Generator(model, cache, tokenizer)
        out = gen.generate(
            prompt=text, max_new_tokens=max_new_tokens, sampler=GreedySampler(),
            completion_only=True, encode_special_tokens=special, add_bos=not special,
            stop_conditions=(list(dict.fromkeys(stop)) or None),
        )
        resp = out if isinstance(out, str) else (out[0] if out else "")
        resp = (resp or "").strip()
        # Belt-and-suspenders: drop anything past a turn-end marker the stop
        # conditions didn't catch, and trim a trailing special token.
        for t in end_strs + ("<|endoftext|>",):
            if t in resp:
                resp = resp.split(t, 1)[0].strip()
        return resp or None
    except Exception as e:
        print(f"[sample] WARN generation failed: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        try:
            if model is not None:
                model.unload()
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


_UPLOAD_TRIES = 4
# Nothing here comes good on a retry: bad request, bad token, no write access,
# missing repo, payload rejected. Everything else (429, 5xx, a dropped socket)
# is worth another go.
_UPLOAD_TERMINAL = {400, 401, 403, 404, 413}


def _upload_folder_hb(api, path, repo_id, variant) -> None:
    """upload_folder on a worker thread with a 20s heartbeat -- pushing tens of
    GB is silent for minutes, the last quiet phase that could trip the
    controller's stall watchdog. Raises on failure (after join).

    Retried, because the pod is torn down afterwards: one transient error used
    to throw away a conversion that had already succeeded. upload_folder asks
    the remote what it already has and pushes only the rest, so a retry resumes
    a part-uploaded repo rather than starting it again.
    """
    import threading
    for attempt in range(1, _UPLOAD_TRIES + 1):
        done, err = threading.Event(), {}

        def _do():
            try:
                api.upload_folder(folder_path=path, repo_id=repo_id, repo_type="model")
            except Exception as exc:
                err["exc"] = exc
            finally:
                done.set()

        t = threading.Thread(target=_do, daemon=True)
        t.start()
        secs = 0
        while not done.wait(20):
            secs += 20
            print(f"[upload] {variant} pushing... {secs}s", flush=True)
        t.join()
        exc = err.get("exc")
        if exc is None:
            return
        code = getattr(getattr(exc, "response", None), "status_code", None)
        if code in _UPLOAD_TERMINAL or attempt == _UPLOAD_TRIES:
            raise exc
        wait = 30 * 2 ** (attempt - 1)
        print(f"[upload] {variant} attempt {attempt}/{_UPLOAD_TRIES} failed "
              f"({code or type(exc).__name__}); resuming in {wait}s", flush=True)
        time.sleep(wait)


def _written_quant_config(out_dir) -> dict:
    """quantization_config the converter just wrote, or {} if unreadable.

    Head bits, codebook and calibration size are all optional on the request
    now -- unset means exllamav3 picks. Cards state those numbers, so read what
    was actually produced instead of repeating upstream's defaults here and
    diverging the day one of them moves.
    """
    try:
        cfg = json.loads((Path(out_dir) / "config.json").read_text(encoding="utf-8"))
        return cfg.get("quantization_config") or {}
    except Exception:
        return {}


def _repo_quant_config(repo_id: str, hf_token: str) -> dict:
    """A published quant's own quantization_config, or {} if unreadable."""
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo_id, "config.json", token=hf_token or None)
        return json.loads(Path(p).read_text(encoding="utf-8")).get("quantization_config") or {}
    except Exception:
        return {}


def _repo_codebook(repo_id: str, hf_token: str, default: str = "mcg") -> str:
    """Codebook a published quant was made with, from its own config.json.

    ExLlamaV3 records it in quantization_config. Older quants predate the key
    and were all mcg, which is also the converter's default.
    """
    return str(_repo_quant_config(repo_id, hf_token).get("codebook") or default)


def _finalize_cards(outputs, model_id, model_name, owner, hf_token,
                    head_bits, cal_rows, codebook, model_dir,
                    name_parts: dict | None = None) -> None:
    """Render each variant's card with the full cross-variant table and push
    README.md to its repo. Runs after the serial upload+delete, so the out_dir
    is gone -- sizes/KL come from the recs and the card goes up via the API."""
    import cards
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    try:
        model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        model_config = {}

    quant_rows = [{
        "variant": o["variant"], "head_bits": o.get("_head_bits", head_bits),
        "cal_rows": o.get("_cal_rows", cal_rows),
        "vision_bits": o.get("_vision_bits"),
        # Without this the Mode column calls a self-calibrated quant "plain",
        # which is the one distinction the column exists to draw.
        "sc": bool((o.get("_name_parts") or {}).get("sc")),
        "repo_id": o.get("hf_repo_id") or cards.exl3_repo_id(
            owner, model_name, o["variant"], **(o.get("_name_parts") or {})),
        "size_gb": o.get("_size_gb"),
        "url": o.get("hf_url") or "https://huggingface.co/" + cards.exl3_repo_id(
            owner, model_name, o["variant"], **(o.get("_name_parts") or {})),
        "kl_div": o.get("kl_div"),
        "kl_method": o.get("kl_method"),
    } for o in outputs]

    license_id = cards.fetch_license(model_id, hf_token or None)
    collection_url = cards.collection_url_for(owner=owner, base_name=model_name, token=hf_token)

    for o in outputs:
        repo_id = o.get("hf_repo_id") or cards.exl3_repo_id(
            owner, model_name, o["variant"], **(o.get("_name_parts") or name_parts or {}))
        card = cards.render_exl3_card(
            base_repo=model_id, repo_id=repo_id, variant=o["variant"],
            head_bits=o.get("_head_bits", head_bits),
            vision_bits=o.get("_vision_bits"),
            cal_rows=o.get("_cal_rows", cal_rows),
            size_gb=o.get("_size_gb"),
            model_config=model_config, quant_rows=quant_rows,
            collection_url=collection_url, license_id=license_id,
            quantized_by=owner, codebook=o.get("_codebook", codebook),
        )
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model")
        print(f"[card] {o['variant']} written", flush=True)


def _backfill_sibling_kl(*, outputs, model_id, model_name, owner, hf_token,
                         head_bits, cal_rows, codebook, kl_rows, model_dir,
                         scratch_dir, max_eval=8) -> None:
    """Retroactively fill KL for existing sibling quants of the same base.

    The fp16 source is already on the pod, so any {owner}/{model_name}-exl3-Xbpw
    repo missing a KL number only needs its (small) quant downloaded to measure.
    Writes bq_quality.json into each, then re-renders every card in the
    collection so the Quants table carries KL for all bpws. Fully best-effort:
    the finished quant has already uploaded by the time this runs.
    """
    import shutil
    import cards
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi(token=hf_token)
    rx = cards.exl3_slug_rx(model_name)
    new_variants = {o["variant"] for o in outputs}

    def _repo_size_gb(repo_id: str):
        try:
            info = api.model_info(repo_id, files_metadata=True)
            total = sum((s.size or 0) for s in info.siblings
                        if s.rfilename.endswith(".safetensors"))
            return (total / 1e9) or None
        except Exception:
            return None

    # variant -> {repo, kl, size_gb}. Seed with this run's new variants.
    table = {}
    for o in outputs:
        repo = f"{owner}/{model_name}-exl3-{o['variant']}bpw"
        table[o["variant"]] = {"repo": repo, "kl": o.get("kl_div"),
                               "kl_method": o.get("kl_method"),
                               "size_gb": o.get("_size_gb"), "codebook": codebook}

    try:
        found = list(api.list_models(author=owner, search=f"{model_name}-exl3"))
    except Exception as e:
        print(f"[backfill] WARN repo list failed: {e}", flush=True)
        found = []
    siblings = []
    for m in found:
        mm = rx.match(m.id.split("/")[-1])
        if mm and mm.group("variant") not in new_variants:
            siblings.append((mm.group("variant"), m.id))

    if not siblings:
        print("[backfill] no existing siblings to fill", flush=True)
        return
    print(f"[backfill] siblings: {', '.join(v for v, _ in siblings)}", flush=True)

    evaled = 0
    for v, repo in sorted(siblings, key=lambda x: float(x[0])):
        size_gb = _repo_size_gb(repo)
        # A sibling may well have been quantized with a different codebook than
        # this run, so read each one's own rather than assuming ours.
        qcfg_v = _repo_quant_config(repo, hf_token)
        cb_v = str(qcfg_v.get("codebook") or "mcg")
        hb_v = qcfg_v.get("head_bits", head_bits)
        vb_v = qcfg_v.get("vision_bits")
        # Already measured? read it back and skip the eval.
        existing = None
        try:
            qp = hf_hub_download(repo, "bq_quality.json", token=hf_token)
            existing = json.loads(Path(qp).read_text()).get("kl_div")
        except EntryNotFoundError:
            existing = None
        except Exception:
            existing = None
        if existing is not None:
            table[v] = {"repo": repo, "kl": float(existing), "size_gb": size_gb,
                        "codebook": cb_v}
            print(f"[backfill] {v} already has KL={float(existing):.6f}", flush=True)
            continue
        if evaled >= max_eval:
            print(f"[backfill] eval cap {max_eval} hit, leaving {v} for later",
                  flush=True)
            table[v] = {"repo": repo, "kl": None, "size_gb": size_gb, "codebook": cb_v}
            continue

        bdir = scratch_dir / f"backfill-{v}bpw"
        kl = None
        try:
            print(f"[backfill] {v} downloading quant ...", flush=True)
            snapshot_download(
                repo, local_dir=str(bdir), token=hf_token,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                                "tokenizer*"],
            )
            print(f"[backfill] {v} measuring KL vs fp16 ...", flush=True)
            _stats, _kl_method = _kl_div_eval(bdir, model_dir, rows=kl_rows)
            kl = _stats["kld_median"] if _stats else None
        except Exception as e:
            print(f"[backfill] {v} eval failed: {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(bdir, ignore_errors=True)
        if kl is not None:
            evaled += 1
            try:
                api.upload_file(
                    path_or_fileobj=json.dumps(
                        {"kl_div": kl, "kl_rows": kl_rows, "kl_method": _kl_method,
                         "kl_stats": _stats,
                         "metric": "KL(fp16||quant) median"}).encode(),
                    path_in_repo="bq_quality.json", repo_id=repo,
                )
                print(f"[backfill] {v} KL={kl:.6f} -> bq_quality.json", flush=True)
            except Exception as e:
                print(f"[backfill] {v} quality upload failed: {e}", flush=True)
        table[v] = {"repo": repo, "kl": kl, "size_gb": size_gb, "codebook": cb_v,
                    "head_bits": hb_v, "vision_bits": vb_v}

    # Re-render every card so the Quants table shows KL for all bpws.
    try:
        model_config = json.loads((model_dir / "config.json").read_text())
    except Exception:
        model_config = {}
    rows_cal = int(cal_rows) if cal_rows else 250
    license_id = cards.fetch_license(model_id, hf_token or None)
    collection_url = cards.collection_url_for(owner=owner, base_name=model_name,
                                              token=hf_token)
    quant_rows = [{
        "variant": v, "head_bits": d.get("head_bits", head_bits), "cal_rows": rows_cal,
        "vision_bits": d.get("vision_bits"), "repo_id": d["repo"],
        "size_gb": d["size_gb"], "url": f"https://huggingface.co/{d['repo']}",
        "kl_div": d["kl"], "kl_method": d.get("kl_method"),
    } for v, d in table.items()]
    for v, d in sorted(table.items(), key=lambda x: float(x[0])):
        try:
            card = cards.render_exl3_card(
                base_repo=model_id, repo_id=d["repo"], variant=v,
                head_bits=d.get("head_bits", head_bits),
                vision_bits=d.get("vision_bits"), cal_rows=rows_cal,
                size_gb=d["size_gb"],
                model_config=model_config, quant_rows=quant_rows,
                collection_url=collection_url, license_id=license_id,
                quantized_by=owner, codebook=d["codebook"],
            )
            api.upload_file(path_or_fileobj=card.encode(),
                            path_in_repo="README.md", repo_id=d["repo"])
            print(f"[backfill] re-rendered card for {v}", flush=True)
        except Exception as e:
            print(f"[backfill] {v} card re-render failed: {e}", flush=True)


def main() -> int:
    try:
        cfg = json.loads(Path(CONFIG_PATH).read_text())
        # cfg holds the HF token + RunPod key -- shred it now that it's in memory
        # so it isn't sitting world-readable on disk for the pod's whole life.
        try:
            os.unlink(CONFIG_PATH)
        except OSError:
            pass
        model_id: str = cfg["model_id"]
        variants: list[str] = cfg["variants"]
        hf_token: str = cfg.get("hf_token", "")
        hf_org: str = cfg.get("hf_org", "")
        _hb = cfg.get("head_bits")
        # None means the request did not pin it, so exllamav3 picks its own
        # default. The cards read the real number back off the written config
        # rather than guessing, which is what _written_head_bits is for.
        head_bits: int | None = int(_hb) if _hb is not None else None
        vision_bits = cfg.get("vision_bits")
        # How many cards the launcher rented. convert's -d defaults to "0", so
        # without this an N-GPU pod quantizes on one card and bills for N.
        gpu_count: int = max(1, int(cfg.get("gpu_count", 1) or 1))
        # Repos that ship several formats keep each in its own directory. Only
        # that subtree is fetched: the one that prompted this had 433 GB across
        # BF16/FP8/GGUF/NVFP4 and we pulled all of it to quantize one.
        subfolder: str = (cfg.get("subfolder") or "").strip().strip("/")
        # Self-calibration. donor_repo is a quant OF THIS MODEL at >= the
        # bitrate being built -- sc_rfn_probe and sc_measure both fail on a
        # foreign one, in two different ways. The bot resolves it before the
        # job starts; nothing here goes looking.
        sc: bool = bool(cfg.get("sc", False))
        donor_repo: str = (cfg.get("donor_repo") or "").strip()
        # One pair drives sc_trace and the conversion both, because convert
        # crops the calibration file to its own and refuses a smaller one.
        # What the published name carries, for anything that needs it before a
        # conversion exists. Once one does, _name_parts_for reads it back off
        # the written config instead -- see there.
        name_parts = {"sc": sc, "head_bits": head_bits, "vision_bits": vision_bits} if sc \
            else ({"vision_bits": vision_bits} if vision_bits else {})
        sc_cal_rows: int = int(cfg.get("cal_rows") or 250)
        sc_cal_cols: int = int(cfg.get("cal_cols") or 2048)
        # Calibration tunables — fewer rows trades quality for speed.
        # ExLlamaV3 defaults are 250 rows × 2048 cols when unset.
        cal_rows: int | None = cfg.get("cal_rows")
        cal_cols: int | None = cfg.get("cal_cols")
        # Trellis codebook. ExLlamaV3's own default is mcg; ours is mul1. The
        # converter rejects anything outside mcg/mul1/3inst, and the choice is
        # baked into the quant as a tensor, so the loader needs no config.
        codebook: str = (cfg.get("codebook") or "mul1").strip().lower()
        # Post-quant KL-divergence of the quant against the fp16, measured on
        # the pod where both still live. On by default (best-effort: skips if the
        # fp16 won't fit the GPU). kl_rows trades accuracy for pod time.
        # backfill_kl (re-eval existing siblings) is opt-in; off unless asked.
        kl_eval: bool = bool(cfg.get("kl_eval", True))
        kl_rows: int = int(cfg.get("kl_rows", 40))
        backfill_kl: bool = bool(cfg.get("backfill_kl", False))
        # Optional smoke-test prompt: run on each finished quant so the requester
        # sees a real reply. The fp16 is already gone by here; we load the quant.
        test_prompt: str = (cfg.get("test_prompt") or "").strip()

        t0 = time.time()

        import torch
        # Every card, not just device 0: a multi-GPU pod that came up with
        # fewer cards than were rented is worth seeing in the log rather than
        # discovering from the bill.
        _n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        _cards = " | ".join(
            f"{torch.cuda.get_device_name(i)} "
            f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.0f}GB"
            for i in range(_n)
        )
        print(f"[gpu] CUDA: {torch.cuda.is_available()} | {_n} device(s) | {_cards}",
              flush=True)
        if _n < gpu_count:
            print(f"[gpu] WARN rented {gpu_count} GPUs but torch sees {_n}", flush=True)

        from huggingface_hub import HfApi, snapshot_download, login as hf_login

        # Everything on the LOCAL container disk (/quant): model, HF cache, work
        # dir, and outputs. RunPod's /workspace volume is network-backed (mfs) in
        # some data centers and throws IO errors under big-model load -- Xet
        # reconstruction failures on download and stalled layer reads during
        # convert. Local NVMe is reliable; the container is sized for all of it.
        quant_root = Path("/quant")
        quant_root.mkdir(parents=True, exist_ok=True)
        workspace = quant_root / "blockquant"        # model + HF cache (local now)
        workspace.mkdir(parents=True, exist_ok=True)
        model_dir = workspace / "model"

        hf_cache = workspace / ".hf-cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
        # Disable Xet. Its chunk-reconstruction step hits "IO Error (os error 5)"
        # on big repos written to the RunPod network volume; the plain HTTP path
        # (accelerated by hf_transfer below) is reliable.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print(f"[disk] all on local container disk: model+cache -> {workspace} | "
              f"outputs+work -> {quant_root}", flush=True)

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            hf_login(token=hf_token)

        # hf_transfer does parallel, Rust-backed chunked downloads, typically a
        # few times faster than the default single-stream path on big repos
        # (the 70GB+ weights are the longest part of startup). Only enable it if
        # the package actually imports, so a base image without it just falls
        # back to the normal downloader instead of erroring.
        try:
            import hf_transfer  # noqa: F401
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("[download] hf_transfer enabled (parallel download)", flush=True)
        except Exception:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

        # Download on a worker thread and emit a size heartbeat every 20s. The
        # download is otherwise silent for many minutes (the full weights are
        # tens of GB), and a silent stretch longer than the controller's stall
        # window gets the pod killed mid-download. Regular "[download] N GB"
        # lines keep the stall watchdog happy and give the dashboard real
        # download progress. Errors from the thread are re-raised on the main
        # thread so the normal failure path still runs.
        import threading
        # Total repo size up front so the heartbeat can report a percent.
        _want = f"{subfolder}/" if subfolder else ""
        try:
            _info = HfApi(token=hf_token or None).model_info(model_id, files_metadata=True)
            _total_gb = sum((s.size or 0) for s in (_info.siblings or [])
                            if s.rfilename.startswith(_want)) / 1e9
        except Exception:
            _total_gb = 0.0
        _what = f"{model_id}/{subfolder}" if subfolder else model_id
        print(f"[download] {_what} ({_total_gb:.1f} GB) ...", flush=True)
        _dl_done = threading.Event()
        _dl_err: dict = {}

        def _do_download() -> None:
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(model_dir),
                    token=hf_token or None,
                    allow_patterns=[f"{subfolder}/*"] if subfolder else None,
                )
            except Exception as exc:  # surfaced after join()
                _dl_err["exc"] = exc
            finally:
                _dl_done.set()

        _dl_thread = threading.Thread(target=_do_download, daemon=True)
        _dl_thread.start()
        while not _dl_done.wait(20):
            gb = _dir_size_gb(model_dir) or 0.0
            if _total_gb > 0:
                pct = min(99, int(gb / _total_gb * 100))
                print(f"[download] {pct}% ({gb:.1f}/{_total_gb:.1f} GB)", flush=True)
            else:
                print(f"[download] {gb:.1f} GB downloaded...", flush=True)
        _dl_thread.join()
        if "exc" in _dl_err:
            raise _dl_err["exc"]
        print("[download] complete", flush=True)

        # snapshot_download preserves repo paths, so a subfolder fetch lands at
        # model_dir/<subfolder>. Move the root here rather than threading the
        # subfolder through _sanitize_config, the converter, the KL eval and the
        # smoke test -- they all just want the directory the weights are in.
        if subfolder:
            model_dir = model_dir / subfolder
            if not (model_dir / "config.json").exists():
                raise FileNotFoundError(
                    f"no config.json under {subfolder}/ after download; "
                    f"got {sorted(p.name for p in model_dir.parent.iterdir())[:10]}"
                )
            print(f"[download] model root -> {model_dir}", flush=True)

        # The self-calibration donor: a quant of this model that generates the
        # in-domain trace everything downstream is built on. Small next to the
        # fp16, and fetched here so a failure lands before any GPU time.
        donor_dir = None
        if sc:
            if not donor_repo:
                raise ValueError("self-calibration needs a donor quant; none was given")
            donor_dir = workspace / "donor"
            print(f"[sc] donor {donor_repo} ...", flush=True)
            snapshot_download(repo_id=donor_repo, local_dir=str(donor_dir),
                              token=hf_token or None)
            print(f"[sc] donor ready ({_dir_size_gb(donor_dir):.1f} GB)", flush=True)

        _sanitize_config(model_dir)
        _disable_missing_mtp(model_dir)
        _vision_preprocessor_config(model_dir)
        _ensure_fast_tokenizer(model_dir)

        from exllamav3.conversion.convert_model import parser, main as exl_main, prepare

        api = owner = model_name = None
        repo_ids = []
        if hf_token:
            api = HfApi(token=hf_token)
            model_name = model_id.split("/")[-1]
            # Resolve the user portion when no org was supplied -- HF rejects
            # bare slugs without a namespace.
            owner = hf_org or api.whoami()["name"]

        def _name_parts_for(rec: dict) -> dict:
            import cards
            """Name the artifact from what the converter wrote, not what we asked for.

            The request is an intent and can be vague -- head bits unset means
            "whatever exllamav3 picks", vision bits unset means "whatever the
            arch does", and 16 means "copy the tower", which is not a tower
            bitrate at all. The written quantization_config is the fact: it
            carries head_bits always and vision_bits only when the tower was
            quantized. Naming off the request published SC quants as -V16 for
            towers that were copied, and dropped the -V6 off ones that were.

            A plain quant still states nothing it did not choose, so it takes
            the suffix only when the requester pinned the tower themselves.
            """
            if sc:
                return {"sc": True,
                        "head_bits": rec.get("_head_bits", head_bits),
                        "vision_bits": rec.get("_vision_bits")}
            vb = cards.quantized_vision_bits(vision_bits)
            return {"vision_bits": vb} if vb else {}

        def _publish(variant, out_dir, work_dir, rec):
            # Own import, like _finalize_cards and _backfill_sibling_kl. main()
            # imports cards further down, which makes the name a local of main
            # -- so a nested function reading it gets an unassigned free
            # variable and NameErrors at the upload, after the whole quant is
            # built. quant.py cannot import it at module scope: in the repo it
            # is blockquant.cards, on the pod it is flat beside this file.
            import cards
            # Serial: upload one variant and free its disk before the next, so
            # peak = model + one output + one work dir + one kl-stage, not the
            # sum over all variants. rmtree only AFTER a confirmed upload -- on
            # failure the dirs stay for rescue_upload.py.
            rec["_size_gb"] = _dir_size_gb(out_dir)
            _qc = _written_quant_config(out_dir)
            rec["_head_bits"] = _qc.get("head_bits", head_bits)
            rec["_codebook"] = _qc.get("codebook", codebook)
            rec["_cal_rows"] = (_qc.get("calibration") or {}).get("rows", cal_rows)
            # Present only when the tower was quantized; that is the signal.
            rec["_vision_bits"] = _qc.get("vision_bits")
            # Settle the name here, where the conversion is on disk to read, so
            # the card pass and the manifest cannot derive a different one.
            rec["_name_parts"] = _name_parts_for(rec)
            if not hf_token:
                return
            # Build the name, never format it here: an SC quant under the
            # plain name would collide with the plain quant of the same
            # bitrate, which is the one thing the suffixes exist to stop.
            repo_id = cards.exl3_repo_id(owner, model_name, variant, **rec["_name_parts"])
            print(f"[upload] {variant} -> {repo_id} ...", flush=True)
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
            _upload_folder_hb(api, str(out_dir), repo_id, variant)
            rec["hf_repo_id"] = repo_id
            rec["hf_revision"] = "main"
            rec["hf_url"] = f"https://huggingface.co/{repo_id}"
            repo_ids.append(repo_id)
            # Echo the URL so the dashboard's HF_URL parser rule can populate
            # state["hf_url"].
            print(f"[upload] {variant} done -> {rec['hf_url']}", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(work_dir, ignore_errors=True)

        outputs = []
        for variant in variants:
            bpw = float(variant)
            out_dir = quant_root / f"output-{bpw}bpw"
            work_dir = quant_root / f"work-{bpw}"
            if (out_dir / "config.json").exists():
                print(f"[skip] {variant} already quantized at {out_dir}", flush=True)
                rec = {"variant": variant, "path": str(out_dir)}
                try:
                    q = json.loads((out_dir / "bq_quality.json").read_text())
                    if q.get("kl_div") is not None:
                        rec["kl_div"] = float(q["kl_div"])
                        rec["kl_method"] = q.get("kl_method")
                except Exception:
                    pass
                _publish(variant, out_dir, work_dir, rec)
                outputs.append(rec)
                continue
            # Self-calibration runs before the conversion and hands it a
            # per-tensor recipe plus the model's own calibration rows. The
            # stages are shared across every bitrate in the job -- only
            # sc_optimize is per-bitrate -- so the expensive part (trace,
            # probe, measure) is paid once.
            sc_recipe = sc_cal = None
            sc_timings: dict = {}
            if sc:
                _sc_work = workspace / "selfcal"
                _cache_repo = _cache_key = None
                if api and owner:
                    _cache_repo = _sc_cache_repo(owner, model_name)
                    _cache_key = _sc_cache_key(model_id, cfg.get("model_revision", ""),
                                               donor_repo, sc_cal_rows, sc_cal_cols)
                    # Only worth asking once: after the first variant the
                    # artifacts are on disk and the stages skip regardless.
                    if not (_sc_work / "measure.json").exists():
                        _sc_cache_restore(api, _cache_repo, _cache_key, _sc_work)
                sc_recipe, sc_cal = _run_sc_stages(
                    model_dir, donor_dir, work_root=_sc_work,
                    bpw=bpw, head_bits=head_bits, cal_rows=sc_cal_rows,
                    cal_cols=sc_cal_cols, timings=sc_timings)
                # A stage that was skipped is recorded as 0.0, so this is
                # "did this job build anything new". A run that restored
                # everything has nothing to add and re-uploading would just
                # churn the repo.
                _built = any(sc_timings.get(n, 0) > 0
                             for n in ("sc_trace", "sc_rfn_probe", "sc_measure"))
                if _cache_repo and _built:
                    _sc_cache_save(api, _cache_repo, _cache_key, _sc_work)

            print(f"[quantize] {variant} bpw ...", flush=True)
            _t_quant = time.time()
            old_argv = sys.argv
            argv = [
                "convert",
                "-i", str(model_dir),
                "-o", str(out_dir),
                "-w", str(work_dir),
                "-b", str(bpw),
                "--codebook", codebook,
                # No-op on exllamav3 >= 1.4 (parallel mode became the default and
                # the flag was kept as an accepted no-op), still meaningful on
                # 0.0.38. Passing it keeps this script correct on either, which
                # matters because the image can be rolled back under it.
                "--parallel_mode",
            ]
            # Vision tower. Omitted means exllamav3 decides: >=1.4.9 quantizes a
            # tower the arch declares validated to 6 bpw and copies the rest at
            # fp16, where <=1.4.2 copied every tower. Passing nothing therefore
            # tracks the image, and an explicit value is how a request pins it.
            if sc_recipe is not None:
                # -rcp replaces the budgeted allocation from --bits/--head_bits
                # (the recipe carries head_bits), and -cd replaces the bundled
                # corpus. cal_rows/cal_cols must match what sc_trace generated:
                # convert crops to its own and refuses a smaller file.
                argv += ["-rcp", str(sc_recipe), "-cd", str(sc_cal),
                         "--cal_rows", str(sc_cal_rows),
                         "--cal_cols", str(sc_cal_cols)]
            if gpu_count > 1:
                argv += ["-d", ",".join(str(i) for i in range(gpu_count))]
            if head_bits is not None:
                argv += ["--head_bits", str(int(head_bits))]
            if vision_bits is not None:
                argv += ["-vb", str(int(vision_bits))]
            # SC already set these to match its trace; a second pair would be
            # argparse's last-wins and could quietly disagree with the file.
            if not sc:
                if cal_rows is not None:
                    argv += ["--cal_rows", str(int(cal_rows))]
                if cal_cols is not None:
                    argv += ["--cal_cols", str(int(cal_cols))]
            sys.argv = argv
            try:
                args = parser.parse_args()
            finally:
                sys.argv = old_argv
            in_args, job_state, ok, err = prepare(args)
            if not ok:
                emit_result({"status": "failed", "error": f"prepare failed: {err}"})
                return 1
            # Stream layer-based quantize progress. The HONEST progress signal is
            # the layer index (0..num_hidden_layers): exllamav3's internal
            # curr/max counter resets per group and does NOT track overall
            # progress (it read 99% while only on layer 7 of 40). Tap stdout to
            # follow the highest "Quantized: ...layers.N" the converter prints,
            # and emit a parseable percent from layer/total.
            import re as _re
            import threading

            def _model_total_layers() -> int:
                try:
                    with open(model_dir / "config.json") as _f:
                        _cfg = json.load(_f)
                except Exception:
                    return 0

                def _find(o):
                    if isinstance(o, dict):
                        for k, v in o.items():
                            if k in ("num_hidden_layers", "num_layers") and isinstance(v, int):
                                return v
                            r = _find(v)
                            if r:
                                return r
                    return None

                return _find(_cfg) or 0

            _total_layers = _model_total_layers()
            _lstate = {"layer": 0}
            _layer_re = _re.compile(r"layers\.(\d+)")

            class _LayerTap:
                """Pass stdout through unchanged, but track the highest layer the
                converter reports so the monitor can read it."""
                def __init__(self, real):
                    self._real = real

                def write(self, s):
                    self._real.write(s)
                    if "Quantized" in s and "layers." in s:
                        for mm in _layer_re.finditer(s):
                            n = int(mm.group(1))
                            if n > _lstate["layer"]:
                                _lstate["layer"] = n

                def flush(self):
                    self._real.flush()

                def __getattr__(self, a):
                    return getattr(self._real, a)

            _q_done = threading.Event()

            def _quant_progress():
                last = -1
                t_first = None
                while not _q_done.wait(15):
                    n = _lstate["layer"]
                    if not _total_layers or n == 0:
                        # Measure/prep phase: no layer reported yet. Heartbeat so
                        # the embed leaves "Downloading" the moment conversion
                        # starts and the controller's tail always has a marker.
                        print(f"[progress] quantize {variant} 0% (preparing)", flush=True)
                        continue
                    if t_first is None:
                        t_first = time.time()
                    if n == last:
                        continue
                    pct = min(99, int(n / _total_layers * 100))
                    eta = ""
                    el = time.time() - t_first
                    if el > 0:
                        rem = (_total_layers - n) * (el / n)
                        eta = f" eta {int(rem // 60)}m" if rem >= 60 else f" eta {int(rem)}s"
                    print(f"[progress] quantize {variant} {pct}% (layer {n}/{_total_layers}){eta}", flush=True)
                    last = n

            _qt = threading.Thread(target=_quant_progress, daemon=True)
            _qt.start()
            _old_stdout = sys.stdout
            sys.stdout = _LayerTap(_old_stdout)
            try:
                exl_main(in_args, job_state)
            finally:
                sys.stdout = _old_stdout
                _q_done.set()
                _qt.join(timeout=2)
            _quant_secs = round(time.time() - _t_quant, 1)
            print(f"[quantize] {variant} complete ({_quant_secs / 60:.1f}m)", flush=True)
            rec = {"variant": variant, "path": str(out_dir)}
            # Shipped in the result so the cost estimate has something to fit.
            # A flat per-variant band cannot describe a job whose two longest
            # stages are bound by different things -- sc_trace on the GPU,
            # sc_measure on the CPU.
            if sc:
                rec["_sc_timings"] = dict(sc_timings)
                rec["_sc_cal"] = {"rows": sc_cal_rows, "cols": sc_cal_cols}
            rec["_quantize_secs"] = _quant_secs
            if kl_eval:
                print(f"[kl] {variant} measuring KL vs fp16 ...", flush=True)
                stats, kl_method = _kl_div_eval(out_dir, model_dir, rows=kl_rows)
                if stats is not None:
                    # The median is what the card quotes. The mean is kept
                    # because turboderp's charts plot it, but his own note is
                    # that it is dominated by tokens the reference is undecided
                    # on, so it is the worse of the two to lead with.
                    kl = stats["kld_median"]
                    rec["kl_div"] = kl
                    rec["kl_stats"] = stats
                    rec["kl_method"] = kl_method
                    print(f"[kl] {variant} KL(fp16||quant) median {kl:.6f} "
                          f"mean {stats['kld']:.6f}", flush=True)
                    # Persist next to the weights so a later card re-render
                    # (publish_quant) and retroactive backfill can read it back.
                    # The method goes with it: a number measured on held-out
                    # text and one measured on the calibration set are not the
                    # same metric and must not share a field unlabelled.
                    payload = json.dumps({"kl_div": kl, "kl_rows": kl_rows,
                                          "kl_method": kl_method,
                                          "kl_stats": stats,
                                          "metric": "KL(fp16||quant) median"})
                    try:
                        (out_dir / "bq_quality.json").write_text(payload, encoding="utf-8")
                    except Exception as e:
                        print(f"[kl] {variant} could not write bq_quality.json: {e}", flush=True)
                    # Written into out_dir BEFORE _publish uploads the folder,
                    # so it ships with the weights and needs no upload of its
                    # own. There used to be one here, guarded on
                    # rec["hf_repo_id"] -- which _publish does not set until
                    # after this block, so it never once ran, under a comment
                    # claiming KL was measured after the upload.
            if test_prompt:
                print(f"[sample] {variant} generating reply ...", flush=True)
                resp = _sample_generate(out_dir, test_prompt)
                if resp:
                    import base64
                    b64 = base64.b64encode(resp.encode("utf-8")).decode("ascii")
                    # Single-line base64 marker (the `b64` sentinel keeps the
                    # status line above from being mis-parsed) so newlines/quotes
                    # in the reply survive the log relay; the bot decodes + previews.
                    print(f"[sample] {variant} b64 {b64}", flush=True)
            _publish(variant, out_dir, work_dir, rec)
            outputs.append(rec)

        if hf_token:
            # Re-render each card with the full cross-variant table now that every
            # bpw is known and push README.md to each repo (out_dirs are gone).
            try:
                _finalize_cards(outputs, model_id, model_name, owner, hf_token,
                                head_bits, cal_rows, codebook, model_dir,
                                name_parts=name_parts)
            except Exception:
                # traceback is imported at module scope; a local re-import here
                # would make the name function-local and trip an
                # UnboundLocalError in the outer handler's traceback.print_exc().
                print("[card] WARN skipped:\n" + traceback.format_exc(), flush=True)

            try:
                import cards
                cards.ensure_collection(owner=owner, base_name=model_name,
                                        token=hf_token, item_repo_ids=repo_ids)
            except Exception as exc:
                print(f"[collection] WARN skipped ({type(exc).__name__}: {exc})", flush=True)
            print("[upload] complete", flush=True)

            # Retroactive KL for existing siblings, reusing the fp16 on disk.
            # Best-effort and runs after the quant has uploaded, so a failure
            # here can never cost the finished variant.
            if backfill_kl:
                try:
                    _backfill_sibling_kl(
                        outputs=outputs, model_id=model_id, model_name=model_name,
                        owner=owner, hf_token=hf_token, head_bits=head_bits,
                        cal_rows=cal_rows, codebook=codebook, kl_rows=kl_rows,
                        model_dir=model_dir, scratch_dir=workspace,
                    )
                except Exception:
                    print("[backfill] WARN skipped:\n" + traceback.format_exc(),
                          flush=True)

        emit_result({
            "status": "complete",
            "outputs": outputs,
            "total_time": time.time() - t0,
        })
        print("[done]", flush=True)

        # Self-terminate so we don't burn credit waiting for the local
        # poll loop to clean up. Skip on --keep-pod (forensics path).
        # Only fires on the success path; failure path below leaves the
        # pod alive for rescue_upload.py / debugging.
        if not cfg.get("keep_pod"):
            pid = cfg.get("pod_id", "")
            key = cfg.get("runpod_api_key", "")
            if pid and key:
                # Arm a detached backstop and exit immediately. A live
                # controller sees this process finish, drains the log, fetches
                # bq-result.json, and terminates the pod itself, all well
                # before the backstop fires. The backstop only matters if the
                # controller has died, so it can't orphan the pod.
                try:
                    grace = float(cfg.get("self_terminate_grace_seconds", 300))
                except (TypeError, ValueError):
                    grace = 300.0
                if grace < 0:
                    grace = 300.0
                _arm_self_terminate_backstop(pid, key, grace)
                print(f"[self-terminate] backstop armed ({grace:g}s)", flush=True)
            else:
                print("[self-terminate] skipped (missing pod_id or api_key)",
                      flush=True)
        else:
            print("[self-terminate] skipped (--keep-pod)", flush=True)
        return 0

    except Exception as e:
        traceback.print_exc()
        emit_result({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
