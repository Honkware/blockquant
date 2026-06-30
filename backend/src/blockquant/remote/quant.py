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
import shutil
import sys
import time
import traceback
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
    code = (
        "import time, urllib.request as u;"
        f"time.sleep({float(grace_seconds)});"
        f"u.urlopen(u.Request('https://rest.runpod.io/v1/pods/{pod_id}', "
        f"method='DELETE', headers={{'Authorization': 'Bearer {api_key}'}}), timeout=20)"
    )
    subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _qwen2vl_preprocessor_shim(model_dir: Path) -> None:
    """Drop a Qwen2VL preprocessor stub if missing — required by some VL
    builds even when we're only using the LM, otherwise convert.py barfs
    when it tries to read the processor config.
    """
    prep = model_dir / "preprocessor_config.json"
    if prep.exists():
        return
    prep.write_text(json.dumps({
        "size": {"shortest_edge": 56, "longest_edge": 56},
        "patch_size": 14,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }))


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


def _eval_text() -> str:
    """Eval text for the KL metric: exllamav3's bundled calibration corpus.

    Read straight off disk (the image bakes conversion/standard_cal_data/*.utf8),
    so the eval never depends on a live dataset download.
    """
    import glob
    import os
    import exllamav3
    base = os.path.join(os.path.dirname(exllamav3.__file__),
                        "conversion", "standard_cal_data")
    parts = []
    for fn in sorted(glob.glob(os.path.join(base, "*.utf8"))):
        try:
            with open(fn, encoding="utf-8") as f:
                parts.append(f.read())
        except Exception:
            pass
    return "\n\n".join(parts)


def _kl_div_eval(quant_dir: Path, fp16_dir: Path, rows: int = 32,
                 seq_len: int = 2048) -> float | None:
    """Mean KL(fp16 || quant) over a few rows of calibration text, for the card.

    Loads ONE model at a time (fp16, then quant) via the high-level Model API
    that the smoke-test path already proved works, forwards each row to get
    logits, and compares the distributions with compute_kl_div. The fp16 logits
    are staged to disk between passes so only one model is ever resident -- this
    sidesteps the segfault in exllamav3's low-level model_diff forward and fits
    any model whose fp16 loads alone (up to ~35B on an 80GB card; a bigger fp16
    like Mixtral will OOM and skip). Best-effort: returns None on any failure so
    a finished quant still uploads without a number.
    """
    import shutil
    import tempfile
    try:
        import torch
        from exllamav3 import Config, Model, Cache, Tokenizer
    except Exception as e:
        print(f"[kl] WARN import failed: {type(e).__name__}: {e}", flush=True)
        return None

    # Tokenize the eval text once, with the quant's tokenizer, into fixed rows.
    try:
        text = _eval_text()
        if not text:
            print("[kl] WARN no bundled eval text found", flush=True)
            return None
        tcfg = Config.from_directory(str(quant_dir))
        tokenizer = Tokenizer.from_config(tcfg)
        all_ids = tokenizer.encode(text)
        vocab = tokenizer.actual_vocab_size
        n = all_ids.shape[-1]
        seqs = [all_ids[:, a:a + seq_len]
                for a in range(0, n - seq_len, seq_len)][:rows]
        if not seqs:
            print("[kl] WARN not enough eval tokens", flush=True)
            return None
    except Exception as e:
        print(f"[kl] WARN tokenize failed: {type(e).__name__}: {e}", flush=True)
        return None

    def _forward_rows(model_dir, on_row) -> None:
        """Load model_dir, forward each seq, call on_row(i, logits_2d), unload."""
        config = Config.from_directory(str(model_dir))
        config.override_dynamic_seq_len(seq_len)
        model = Model.from_config(config)
        cache = Cache(model, max_num_tokens=seq_len)
        model.load()
        try:
            for i, seq in enumerate(seqs):
                params = {"attn_mode": "flash_attn", "cache": cache,
                          "past_len": 0, "batch_shape": (1, seq_len)}
                logits = model.forward(seq, params=params)  # (1, L, vocab)
                on_row(i, logits[0])
        finally:
            try:
                model.unload()
            except Exception:
                pass
            torch.cuda.empty_cache()

    stage = Path(tempfile.mkdtemp(prefix="klstage-", dir=str(quant_dir.parent)))
    try:
        # Pass 1: fp16 -> stage each row's logits (fp16 on disk to halve size).
        _forward_rows(fp16_dir, lambda i, lg: torch.save(lg.half().cpu(), stage / f"f{i}.pt"))

        # Pass 2: quant -> KL vs the staged fp16 logits.
        kls: list[float] = []

        def _cmp(i, q_logits):
            f_logits = torch.load(stage / f"f{i}.pt").to(q_logits.device).float()
            kv = min(vocab, q_logits.shape[-1], f_logits.shape[-1])
            qi = q_logits[..., :kv].float()
            fi = f_logits[..., :kv]
            # KL(P_fp16 || P_quant), pure torch. exllamav3's compute_kl_div uses
            # a custom kernel that segfaults here, so do the math directly.
            kl = (torch.softmax(fi, dim=-1)
                  * (torch.log_softmax(fi, dim=-1) - torch.log_softmax(qi, dim=-1))
                  ).sum(-1).mean().item()
            kls.append(kl)

        _forward_rows(quant_dir, _cmp)
        return (sum(kls) / len(kls)) if kls else None
    except Exception as e:
        print(f"[kl] WARN eval failed: {type(e).__name__}: {e}", flush=True)
        return None
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


def _upload_folder_hb(api, path, repo_id, variant) -> None:
    """upload_folder on a worker thread with a 20s heartbeat -- pushing tens of
    GB is silent for minutes, the last quiet phase that could trip the
    controller's stall watchdog. Raises on failure (after join)."""
    import threading
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
    if "exc" in err:
        raise err["exc"]


def _finalize_cards(outputs, model_id, model_name, owner, hf_token,
                    head_bits, cal_rows, model_dir) -> None:
    """Render each variant's card with the full cross-variant table and push
    README.md to its repo. Runs after the serial upload+delete, so the out_dir
    is gone -- sizes/KL come from the recs and the card goes up via the API."""
    import cards
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    rows_cal = int(cal_rows) if cal_rows else 250
    try:
        model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except Exception:
        model_config = {}

    quant_rows = [{
        "variant": o["variant"], "head_bits": head_bits, "cal_rows": rows_cal,
        "size_gb": o.get("_size_gb"),
        "url": o.get("hf_url") or f"https://huggingface.co/{cards.exl3_repo_id(owner, model_name, o['variant'])}",
        "kl_div": o.get("kl_div"),
    } for o in outputs]

    license_id = cards.fetch_license(model_id, hf_token or None)
    collection_url = cards.ensure_collection(owner=owner, base_name=model_name, token=hf_token)

    for o in outputs:
        repo_id = o.get("hf_repo_id") or cards.exl3_repo_id(owner, model_name, o["variant"])
        card = cards.render_exl3_card(
            base_repo=model_id, repo_id=repo_id, variant=o["variant"],
            head_bits=head_bits, cal_rows=rows_cal, size_gb=o.get("_size_gb"),
            model_config=model_config, quant_rows=quant_rows,
            collection_url=collection_url, license_id=license_id,
            quantized_by=owner,
        )
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model")
        print(f"[card] {o['variant']} written", flush=True)


def _backfill_sibling_kl(*, outputs, model_id, model_name, owner, hf_token,
                         head_bits, cal_rows, kl_rows, model_dir, scratch_dir,
                         max_eval=8) -> None:
    """Retroactively fill KL for existing sibling quants of the same base.

    The fp16 source is already on the pod, so any {owner}/{model_name}-exl3-Xbpw
    repo missing a KL number only needs its (small) quant downloaded to measure.
    Writes bq_quality.json into each, then re-renders every card in the
    collection so the Quants table carries KL for all bpws. Fully best-effort:
    the finished quant has already uploaded by the time this runs.
    """
    import re as _re
    import shutil
    import cards
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import EntryNotFoundError

    api = HfApi(token=hf_token)
    rx = _re.compile(rf"^{_re.escape(model_name)}-exl3-([0-9.]+)bpw$")
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
                               "size_gb": o.get("_size_gb")}

    try:
        found = list(api.list_models(author=owner, search=f"{model_name}-exl3"))
    except Exception as e:
        print(f"[backfill] WARN repo list failed: {e}", flush=True)
        found = []
    siblings = []
    for m in found:
        mm = rx.match(m.id.split("/")[-1])
        if mm and mm.group(1) not in new_variants:
            siblings.append((mm.group(1), m.id))

    if not siblings:
        print("[backfill] no existing siblings to fill", flush=True)
        return
    print(f"[backfill] siblings: {', '.join(v for v, _ in siblings)}", flush=True)

    evaled = 0
    for v, repo in sorted(siblings, key=lambda x: float(x[0])):
        size_gb = _repo_size_gb(repo)
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
            table[v] = {"repo": repo, "kl": float(existing), "size_gb": size_gb}
            print(f"[backfill] {v} already has KL={float(existing):.6f}", flush=True)
            continue
        if evaled >= max_eval:
            print(f"[backfill] eval cap {max_eval} hit, leaving {v} for later",
                  flush=True)
            table[v] = {"repo": repo, "kl": None, "size_gb": size_gb}
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
            kl = _kl_div_eval(bdir, model_dir, rows=kl_rows)
        except Exception as e:
            print(f"[backfill] {v} eval failed: {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(bdir, ignore_errors=True)
        if kl is not None:
            evaled += 1
            try:
                api.upload_file(
                    path_or_fileobj=json.dumps(
                        {"kl_div": kl, "kl_rows": kl_rows,
                         "metric": "KL(fp16||quant)"}).encode(),
                    path_in_repo="bq_quality.json", repo_id=repo,
                )
                print(f"[backfill] {v} KL={kl:.6f} -> bq_quality.json", flush=True)
            except Exception as e:
                print(f"[backfill] {v} quality upload failed: {e}", flush=True)
        table[v] = {"repo": repo, "kl": kl, "size_gb": size_gb}

    # Re-render every card so the Quants table shows KL for all bpws.
    try:
        model_config = json.loads((model_dir / "config.json").read_text())
    except Exception:
        model_config = {}
    rows_cal = int(cal_rows) if cal_rows else 250
    license_id = cards.fetch_license(model_id, hf_token or None)
    collection_url = cards.ensure_collection(owner=owner, base_name=model_name,
                                             token=hf_token)
    quant_rows = [{
        "variant": v, "head_bits": head_bits, "cal_rows": rows_cal,
        "size_gb": d["size_gb"], "url": f"https://huggingface.co/{d['repo']}",
        "kl_div": d["kl"],
    } for v, d in table.items()]
    for v, d in sorted(table.items(), key=lambda x: float(x[0])):
        try:
            card = cards.render_exl3_card(
                base_repo=model_id, repo_id=d["repo"], variant=v,
                head_bits=head_bits, cal_rows=rows_cal, size_gb=d["size_gb"],
                model_config=model_config, quant_rows=quant_rows,
                collection_url=collection_url, license_id=license_id,
                quantized_by=owner,
            )
            api.upload_file(path_or_fileobj=card.encode(),
                            path_in_repo="README.md", repo_id=d["repo"])
            print(f"[backfill] re-rendered card for {v}", flush=True)
        except Exception as e:
            print(f"[backfill] {v} card re-render failed: {e}", flush=True)


def main() -> int:
    try:
        cfg = json.loads(Path(CONFIG_PATH).read_text())
        model_id: str = cfg["model_id"]
        variants: list[str] = cfg["variants"]
        hf_token: str = cfg.get("hf_token", "")
        hf_org: str = cfg.get("hf_org", "")
        head_bits: int = int(cfg.get("head_bits", 8))
        # Calibration tunables — fewer rows trades quality for speed.
        # ExLlamaV3 defaults are 250 rows × 2048 cols when unset.
        cal_rows: int | None = cfg.get("cal_rows")
        cal_cols: int | None = cfg.get("cal_cols")
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
        print(
            f"[gpu] CUDA: {torch.cuda.is_available()} | "
            f"{torch.cuda.get_device_name(0)} | "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

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
        try:
            _info = HfApi(token=hf_token or None).model_info(model_id, files_metadata=True)
            _total_gb = sum((s.size or 0) for s in (_info.siblings or [])) / 1e9
        except Exception:
            _total_gb = 0.0
        print(f"[download] {model_id} ({_total_gb:.1f} GB) ...", flush=True)
        _dl_done = threading.Event()
        _dl_err: dict = {}

        def _do_download() -> None:
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(model_dir),
                    token=hf_token or None,
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

        _sanitize_config(model_dir)
        _qwen2vl_preprocessor_shim(model_dir)
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

        def _publish(variant, out_dir, work_dir, rec):
            # Serial: upload one variant and free its disk before the next, so
            # peak = model + one output + one work dir + one kl-stage, not the
            # sum over all variants. rmtree only AFTER a confirmed upload -- on
            # failure the dirs stay for rescue_upload.py.
            rec["_size_gb"] = _dir_size_gb(out_dir)
            if not hf_token:
                return
            repo_id = f"{owner}/{model_name}-exl3-{variant}bpw"
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
                except Exception:
                    pass
                _publish(variant, out_dir, work_dir, rec)
                outputs.append(rec)
                continue
            print(f"[quantize] {variant} bpw ...", flush=True)
            old_argv = sys.argv
            argv = [
                "convert",
                "-i", str(model_dir),
                "-o", str(out_dir),
                "-w", str(work_dir),
                "-b", str(bpw),
                "--head_bits", str(head_bits),
                "--parallel_mode",
            ]
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
            print(f"[quantize] {variant} complete", flush=True)
            rec = {"variant": variant, "path": str(out_dir)}
            if kl_eval:
                print(f"[kl] {variant} measuring KL vs fp16 ...", flush=True)
                kl = _kl_div_eval(out_dir, model_dir, rows=kl_rows)
                if kl is not None:
                    rec["kl_div"] = kl
                    print(f"[kl] {variant} KL(fp16||quant) = {kl:.6f}", flush=True)
                    # Persist next to the weights so a later card re-render
                    # (publish_quant) and retroactive backfill can read it back.
                    try:
                        (out_dir / "bq_quality.json").write_text(
                            json.dumps({"kl_div": kl, "kl_rows": kl_rows,
                                        "metric": "KL(fp16||quant)"}),
                            encoding="utf-8")
                    except Exception:
                        pass
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
                                head_bits, cal_rows, model_dir)
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
                        cal_rows=cal_rows, kl_rows=kl_rows, model_dir=model_dir,
                        scratch_dir=workspace,
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
