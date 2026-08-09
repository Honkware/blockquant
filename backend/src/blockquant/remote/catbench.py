#!/usr/bin/env python3
"""CatBench entrypoint — runs inside the RunPod pod.

Reads ``/root/bq-catbench-config.json``, downloads the weights from
HuggingFace, asks the model the two CatBench prompts, renders the matplotlib
answer to a PNG, and writes ``/root/bq-result.json`` for the controller to
pick up.

Two loaders: transformers for fp16/bf16 source weights, exllamav3 for an EXL3
quant. Which one is decided by the downloaded config.json, not the repo name.

The second prompt makes the model WRITE PYTHON THAT WE THEN RUN, so the
runner below treats that code as hostile: separate network namespace, an
unprivileged uid, address-space/CPU/file-size/proc rlimits, a scrubbed env, a
throwaway cwd and a hard wall-clock kill. See ``run_untrusted_python``.

SFTP'd to /root/catbench.py by backend/scripts/run_catbench_job.py. Nothing
here is shared with remote/quant.py, so /quant is untouched.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

CONFIG_PATH = "/root/bq-catbench-config.json"
RESULT_PATH = "/root/bq-result.json"
MODEL_DIR = Path("/quant/cb-model")

# Exact, not paraphrased. These are the benchmark.
PROMPT_SVG = "Create a detailed SVG image of a cute kitten."
PROMPT_PY = "Write a Python script that draws a cute kitten using matplotlib."

# SVG needs room to finish the drawing; a truncated one renders as junk.
MAX_NEW_SVG = 4096
MAX_NEW_PY = 2048
# exllamav3 allocates its cache up front, so it has to cover the longest turn
# (prompt + template + MAX_NEW_SVG) with room to spare.
CACHE_TOKENS = 8192

END_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>")

SVG_RE = re.compile(r"<svg[\s\S]*?</svg>", re.I)
# Fences must start a line, or the CLOSING fence of one block reads as the
# OPENING of the next and we hand back the prose in between.
FENCE_RE = re.compile(r"^```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n([\s\S]*?)^```", re.M)

# Sandbox bounds for the model-written script.
SBX_TIMEOUT_S = 90       # hard wall clock, process group killed
SBX_CPU_S = 60           # RLIMIT_CPU, SIGXCPU before the wall clock
SBX_MEM_BYTES = 4 << 30  # RLIMIT_AS
SBX_FSIZE_BYTES = 32 << 20
SBX_NPROC = 64


def emit_result(payload: dict) -> None:
    try:
        with open(RESULT_PATH, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"[fatal] could not write result: {e}", flush=True)


def _arm_self_terminate_backstop(pod_id: str, api_key: str, grace_seconds: float) -> None:
    """Detached process that kills the pod after `grace_seconds`.

    Only fires if the controller died before terminating us, so it can never
    orphan a pod or race a live controller. Same idea as remote/quant.py's.
    """
    if not pod_id or not api_key:
        return
    code = (
        "import time,sys\n"
        f"time.sleep({float(grace_seconds)})\n"
        "try:\n"
        "    import runpod\n"
        f"    runpod.api_key = {api_key!r}\n"
        f"    runpod.terminate_pod({pod_id!r})\n"
        "except Exception as e:\n"
        "    print('backstop failed', e, file=sys.stderr)\n"
    )
    try:
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"[progress] self-terminate backstop armed (+{grace_seconds:.0f}s)", flush=True)
    except Exception as e:
        print(f"[progress] WARN backstop not armed: {e}", flush=True)


# ── The sandbox ──────────────────────────────────────────────────────────────

# Runs as root under `unshare -n` (no network interfaces at all), then drops to
# an unprivileged uid, clamps rlimits, blinds the socket module, and only then
# execs the model's code. Written to disk rather than passed with -c so a
# traceback has real line numbers to report.
_RUNNER = r'''
import builtins, os, resource, socket, sys, traceback

WORK, CODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(WORK)

# Drop root. unshare gave us the empty netns; we do not need privileges to draw
# a cat. Non-fatal on a pod image with no `nobody`: the netns + rlimits stand.
try:
    import pwd
    ent = pwd.getpwnam("nobody")
    os.setgroups([])
    os.setgid(ent.pw_gid)
    os.setuid(ent.pw_uid)
except Exception as e:
    print(f"[sbx] WARN could not drop privileges: {e}", file=sys.stderr)

for res, lim in (
    (resource.RLIMIT_AS, %(mem)d),
    (resource.RLIMIT_CPU, %(cpu)d),
    (resource.RLIMIT_FSIZE, %(fsize)d),
    (resource.RLIMIT_NPROC, %(nproc)d),
    (resource.RLIMIT_CORE, 0),
):
    try:
        resource.setrlimit(res, (lim, lim))
    except Exception as e:
        print(f"[sbx] WARN rlimit {res}: {e}", file=sys.stderr)

# Belt for the netns braces: even if unshare was unavailable, nothing here opens
# a socket. Also kill interactive reads so a stray input() burns the clock.
def _blocked(*a, **k):
    raise OSError("network disabled in the CatBench sandbox")

socket.socket = _blocked
socket.create_connection = _blocked
socket.socketpair = _blocked
socket.create_server = _blocked
builtins.input = _blocked

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Most answers end in plt.show(). Swallow it and save at the end instead, so we
# get the figure whether the model shows it, saves it, or just builds it.
plt.show = lambda *a, **k: None

src = open(CODE, encoding="utf-8", errors="replace").read()
g = {"__name__": "__main__", "__file__": CODE, "__builtins__": builtins}
failed = None
try:
    exec(compile(src, "kitten.py", "exec"), g)
except SystemExit:
    pass
except BaseException:
    failed = traceback.format_exc(limit=6)

saved = False
try:
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if not fig.get_axes():
            continue
        fig.savefig(OUT, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        saved = True
        break
except Exception as e:
    print(f"[sbx] WARN savefig: {e}", file=sys.stderr)

# Fallback: the script wrote its own image file somewhere in the work dir.
if not saved:
    cands = [p for p in os.scandir(".")
             if p.is_file() and p.name != os.path.basename(OUT)
             and p.name.lower().endswith((".png", ".jpg", ".jpeg"))]
    if cands:
        newest = max(cands, key=lambda p: p.stat().st_mtime)
        os.replace(newest.path, OUT)
        saved = True

if not saved:
    print(f"[sbx] no figure produced{': ' + failed if failed else ''}", file=sys.stderr)
    sys.exit(4)
if failed:
    # Drew something and then blew up. Keep the picture, report the crash: exit
    # 5 so the parent knows stderr is worth passing on.
    print(f"[sbx] partial: {failed}", file=sys.stderr)
    sys.exit(5)
sys.exit(0)
'''


def _looks_like_python(s: str) -> bool:
    return "import " in s or "matplotlib" in s or "plt." in s


def extract_python(text: str) -> str | None:
    """Pull the model's script out of a reply.

    A python-tagged fence wins. Failing that, the biggest untagged fence that
    reads like python (replies often open with a ```bash pip install). Failing
    that, the whole reply, when the model skipped the fences entirely.
    """
    if not text:
        return None
    blocks = [(lang.lower(), body.strip()) for lang, body in FENCE_RE.findall(text)]
    tagged = [b for lang, b in blocks if lang in ("python", "py", "python3")]
    if tagged:
        return max(tagged, key=len) or None
    untagged = [b for lang, b in blocks if not lang and _looks_like_python(b)]
    if untagged:
        return max(untagged, key=len) or None
    stripped = text.strip()
    return stripped if _looks_like_python(stripped) else None


def extract_svg(text: str) -> str | None:
    m = SVG_RE.search(text or "")
    return m.group(0) if m else None


_unshare_ok: bool | None = None


def _unshare_works() -> bool:
    """True if we can actually enter an empty network namespace. Cached."""
    global _unshare_ok
    if _unshare_ok is None:
        _unshare_ok = False
        if shutil.which("unshare"):
            try:
                _unshare_ok = subprocess.run(
                    ["unshare", "-n", "--", "true"], timeout=15,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ).returncode == 0
            except Exception:
                _unshare_ok = False
    return _unshare_ok


def run_untrusted_python(code: str) -> tuple[bytes | None, str | None]:
    """Execute model-written matplotlib code and return (png_bytes, error).

    Every layer of confinement is applied here; see the module docstring. The
    pod is throwaway, but the pod also holds the HF token, so this does not
    lean on the pod being disposable.
    """
    work = Path(tempfile.mkdtemp(prefix="catbench-", dir="/tmp"))
    try:
        code_path = work / "kitten.py"
        runner_path = work / "_runner.py"
        out_path = work / "figure.png"
        code_path.write_text(code, encoding="utf-8")
        runner_path.write_text(
            _RUNNER % {"mem": SBX_MEM_BYTES, "cpu": SBX_CPU_S,
                       "fsize": SBX_FSIZE_BYTES, "nproc": SBX_NPROC},
            encoding="utf-8",
        )
        # The dropped-privilege uid has to be able to read both and write here.
        os.chmod(work, 0o777)
        os.chmod(code_path, 0o644)
        os.chmod(runner_path, 0o644)

        mpl_dir = work / "mplconfig"
        mpl_dir.mkdir()
        os.chmod(mpl_dir, 0o777)
        # Nothing inherited: no HF_TOKEN, no RUNPOD_API_KEY, no PYTHONPATH.
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(work),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(mpl_dir),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C.UTF-8",
        }
        argv = [sys.executable, str(runner_path), str(work), str(code_path), str(out_path)]
        # `unshare -n` gives the child an empty network namespace: no loopback,
        # no route, nothing to exfiltrate to. Needs the root we still have here,
        # so probe it rather than assuming (a locked-down host refuses it).
        if _unshare_works():
            argv = ["unshare", "-n", "--"] + argv
        else:
            print("[progress] WARN no netns; sandbox is rlimits + socket block only", flush=True)

        try:
            proc = subprocess.run(
                argv, env=env, cwd=str(work), timeout=SBX_TIMEOUT_S,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return None, f"script exceeded the {SBX_TIMEOUT_S}s sandbox timeout"

        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        if out_path.exists() and out_path.stat().st_size:
            return out_path.read_bytes(), (err[-600:] if proc.returncode else None)
        return None, (err[-600:] or f"script exited {proc.returncode} with no figure")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── Generation ───────────────────────────────────────────────────────────────

def _ensure_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
        return
    except ImportError:
        pass
    print("[progress] installing matplotlib", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", "matplotlib"],
        check=False,
    )


def download(model_id: str, token: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"[download] {model_id}", flush=True)
    snapshot_download(
        model_id, local_dir=str(MODEL_DIR), token=token or None, max_workers=8,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json",
                        "*.txt", "*.model", "tokenizer*", "*.jinja"],
    )
    print("[download] 100% complete", flush=True)


def quant_method() -> str:
    """quant_method from the downloaded config.json, "" for plain weights.

    Read from the file, never the repo name: `-exl3` in a repo id proves
    nothing, and plenty of exl3 repos are not named for it.
    """
    try:
        cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str((cfg.get("quantization_config") or {}).get("quant_method", "")).lower()


def _pkg_version(name: str, mod=None) -> str:
    """Installed version of a loader package.

    Package metadata first: a git/source install of exllamav3 sets no top-level
    __version__, which is why the image health check reads it this way too.
    """
    try:
        from importlib.metadata import version
        v = version(name)
        if v:
            return str(v)
    except Exception:
        pass
    v = (getattr(mod, "__version__", "")
         or getattr(getattr(mod, "version", None), "__version__", ""))
    return str(v or "")


def _hf_tokenizer():
    """AutoTokenizer from the download. Both loaders want it for the chat
    template; only the transformers one uses it to tokenize."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_DIR))


def _chat_wrap(tok, prompt: str) -> tuple[str, bool]:
    """(text, encode_special). Instruct models get their template so they reply
    in character; a base model gets the raw prompt."""
    if tok is not None and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        ), True
    return prompt, False


def _trim(resp: str) -> str:
    for marker in END_MARKERS:
        if marker in resp:
            resp = resp.split(marker, 1)[0]
    return resp.strip()


class HFRunner:
    """fp16/bf16 source weights through transformers."""

    loader = "transformers"

    def __init__(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
        self.engine = _pkg_version("transformers", transformers)
        # trust_remote_code stays OFF. /catbench has no approval gate, so an
        # arbitrary HF repo must not get to run its own python next to our HF
        # token. Models that need custom code fail here, loudly, by design.
        self.tok = _hf_tokenizer()
        kw = dict(device_map="cuda:0", low_cpu_mem_usage=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), dtype=torch.bfloat16, **kw)
        except TypeError:
            # transformers < 4.56 spells it torch_dtype.
            self.model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), torch_dtype=torch.bfloat16, **kw)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch
        tok = self.tok
        text, special = _chat_wrap(tok, prompt)
        ids = tok(text, return_tensors="pt", add_special_tokens=not special).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
        return _trim(tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True))

    def unload(self):
        del self.model


class Exl3Runner:
    """An already-quantized EXL3 repo through exllamav3.

    transformers cannot open one at all: its AutoQuantizer has no exl3 entry,
    so from_pretrained raises "Unknown quantization type, got exl3" before it
    touches a shard. Same load/generate shape as remote/quant.py's
    _sample_generate, which drives freshly converted quants on this image.
    """

    loader = "exl3"

    def __init__(self, max_tokens: int = CACHE_TOKENS):
        import exllamav3
        from exllamav3 import Config, Model, Cache, Tokenizer, Generator
        self.engine = _pkg_version("exllamav3", exllamav3)
        config = Config.from_directory(str(MODEL_DIR))
        self.model = Model.from_config(config)
        self.cache = Cache(self.model, max_num_tokens=max_tokens)
        self.model.load()
        self.gen = Generator(self.model, self.cache, Tokenizer.from_config(config))
        # Chat template and stop ids come from the HF tokenizer; exllamav3's own
        # has neither. Best-effort: a repo without one still runs raw, it just
        # rambles past the turn boundary until the token budget runs out.
        self.tok, self.stop = None, []
        try:
            self.tok = _hf_tokenizer()
            if self.tok.eos_token_id is not None:
                self.stop.append(self.tok.eos_token_id)
            for marker in END_MARKERS:
                tid = self.tok.convert_tokens_to_ids(marker)
                if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                    self.stop.append(tid)
        except Exception as e:
            print(f"[progress] WARN no HF tokenizer ({e}); raw prompt", flush=True)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        from exllamav3 import GreedySampler
        text, special = _chat_wrap(self.tok, prompt)
        out = self.gen.generate(
            prompt=text, max_new_tokens=max_new_tokens, sampler=GreedySampler(),
            completion_only=True, encode_special_tokens=special, add_bos=not special,
            stop_conditions=(list(dict.fromkeys(self.stop)) or None),
        )
        return _trim(out if isinstance(out, str) else (out[0] if out else ""))

    def unload(self):
        self.model.unload()


def load_model(fmt: str):
    """Pick a loader for what was actually downloaded.

    The controller rejects every other quant format at preflight, so anything
    reaching this branch is a bug there, not a user's problem. Say so plainly
    rather than letting transformers raise from three frames down.
    """
    print(f"[progress] loading weights ({fmt or 'bf16'})", flush=True)
    if fmt == "exl3":
        return Exl3Runner()
    if fmt:
        raise RuntimeError(
            f"quantization '{fmt}' has no loader on this pod; CatBench reads "
            "fp16/bf16 safetensors or an EXL3 quant")
    return HFRunner()


def main() -> None:
    t0 = time.time()
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except Exception as e:
        emit_result({"status": "error", "error": f"config unreadable: {e}"})
        print(f"[joberror] config unreadable: {e}", flush=True)
        return
    # Holds the HF token + RunPod key. Gone from disk before we touch weights.
    try:
        os.unlink(CONFIG_PATH)
    except OSError:
        pass

    model_id = cfg["model_id"]
    token = cfg.get("hf_token", "")
    _arm_self_terminate_backstop(
        cfg.get("pod_id", ""), cfg.get("runpod_api_key", ""),
        float(cfg.get("backstop_seconds", 1800)),
    )

    result: dict = {"status": "error", "model_id": model_id}
    runner = None
    try:
        _ensure_matplotlib()
        download(model_id, token)
        # Download is the only step that needs the token. Drop it before any
        # model code or model-written code gets a chance to read the env.
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
                  "RUNPOD_API_KEY"):
            os.environ.pop(k, None)
        token = ""
        cfg.pop("hf_token", None)
        cfg.pop("runpod_api_key", None)

        fmt = quant_method()
        result["format"] = fmt or "bf16"
        runner = load_model(fmt)
        # What read the weights, and at what version. The bot stores this with
        # the pictures: the same repo through a newer exllamav3 is arguably a
        # different result, and you cannot tell from a jpg.
        result["loader"] = runner.loader
        result["engine"] = runner.engine
        result["prompts"] = {"svg": PROMPT_SVG, "python": PROMPT_PY}

        print("[progress] prompt 1/2 (svg)", flush=True)
        svg_reply = runner.generate(PROMPT_SVG, MAX_NEW_SVG)
        print("[progress] prompt 2/2 (python)", flush=True)
        py_reply = runner.generate(PROMPT_PY, MAX_NEW_PY)

        svg = extract_svg(svg_reply)
        result["svg"] = svg
        if not svg:
            result["svg_error"] = "no <svg> element in the reply"

        code = extract_python(py_reply)
        result["python_source"] = code
        if not code:
            result["python_error"] = "no python in the reply"
        else:
            print("[progress] rendering the python answer (sandboxed)", flush=True)
            png, err = run_untrusted_python(code)
            if png:
                result["python_png_b64"] = base64.b64encode(png).decode("ascii")
            if err:
                result["python_error"] = err

        result["status"] = "complete"
        result["elapsed"] = round(time.time() - t0, 1)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        try:
            if runner is not None:
                runner.unload()
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        shutil.rmtree(MODEL_DIR, ignore_errors=True)

    emit_result(result)
    if result["status"] == "complete":
        print(f"[done] catbench {model_id} in {result.get('elapsed', 0):.0f}s", flush=True)
    else:
        print(f"[joberror] {result.get('error', 'unknown')}", flush=True)


if __name__ == "__main__":
    main()
