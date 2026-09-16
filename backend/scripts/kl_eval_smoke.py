"""Does _kl_div_eval actually produce a number?

It has never succeeded. Its first run died on "max_num_tokens must be a
multiple of 256" and the fix after that is verified by arithmetic, not by
executing. Meanwhile every job attempts it, so a wrong fix means every card
silently loses its KL column again.

Rents one small pod, pulls an fp16 model and the quant we already published
from it, and calls _kl_div_eval directly. Terminates the pod on every path.
"""
import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blockquant.providers.runpod.provider import RunPodProvider

# Stock moves constantly; one hardcoded card is a coin flip. Sweep like the
# launcher does rather than failing on the first QueryError.
CANDIDATES = ["NVIDIA RTX A5000", "NVIDIA A40", "NVIDIA RTX A6000",
              "NVIDIA GeForce RTX 3090", "NVIDIA L40S", "NVIDIA RTX 4000 Ada Generation"]


def rent(**kw):
    last = None
    for gpu in CANDIDATES:
        p = RunPodProvider(api_key=os.environ["RUNPOD_API_KEY"], gpu_type=gpu,
                           image=os.environ["RUNPOD_IMAGE"], **kw)
        try:
            pod = p.launch({"env": {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}})
            print(f"[pod] {gpu} -> {pod}", flush=True)
            return p, pod
        except Exception as e:
            print(f"[pod] {gpu} unavailable ({str(e)[:60]}), next", flush=True)
            last = e
    raise SystemExit(f"no GPU available from {CANDIDATES}: {last}")


FP16 = "Qwen/Qwen3.5-0.8B"
QUANT = "Honkware/Qwen3.5-0.8B-exl3-4.0bpw"

pod = None
try:
    p, pod = rent(container_disk_gb=60, volume_gb=0)
    print("[pod] ssh:", p._get_ssh_endpoint(pod, timeout=900), flush=True)

    # The baked quant.py predates the 256 fix, so ship the current one.
    local = Path(__file__).resolve().parents[1] / "src" / "blockquant" / "remote" / "quant.py"
    p._upload_file(pod, local, "/opt/blockquant/quant.py")
    print("[upload] current quant.py shipped", flush=True)

    def sh(cmd, label, tail=25):
        r = p.run(pod, cmd)
        print(f"[{label}] exit={r['code']}", flush=True)
        for stream in (r["stdout"], r["stderr"]):
            t = "\n".join((stream or "").splitlines()[-tail:])
            if t.strip():
                print(t, flush=True)
        return r

    sh(f"python -c \"from huggingface_hub import snapshot_download as d;"
       f"d('{FP16}', local_dir='/quant/fp16');d('{QUANT}', local_dir='/quant/q')\" 2>&1 | tail -2",
       "download")

    sh("cd /opt/blockquant && python -c \""
       "import importlib.util,json,sys;"
       "spec=importlib.util.spec_from_file_location('q','/opt/blockquant/quant.py');"
       "m=importlib.util.module_from_spec(spec);sys.modules['q']=m;spec.loader.exec_module(m);"
       "from pathlib import Path;"
       "stats,method=m._kl_div_eval(Path('/quant/q'),Path('/quant/fp16'),rows=4,seq_len=2048);"
       "print('METHOD:',method);"
       "print('STATS:',json.dumps({k:v for k,v in (stats or {}).items() if not isinstance(v,list)},indent=1) if stats else 'NONE')"
       "\" 2>&1 | tail -30", "kl_eval")
finally:
    if pod:
        try:
            p.terminate(pod)
            print(f"[pod] terminated {pod}", flush=True)
        except Exception as e:
            print(f"[pod] TERMINATE FAILED {pod}: {e} -- CHECK CONSOLE", flush=True)
