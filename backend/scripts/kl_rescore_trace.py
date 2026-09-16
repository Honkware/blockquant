"""Score a plain quant and a self-calibrated one on the same self-sampled trace.

On wikitext-2 the self-calibrated 3.0bpw measured 0.1200 median KL against the
plain 3.0bpw's 0.0782 -- worse, from a pipeline that ran clean. turboderp's
qbench_prompts.py says why that may be the wrong reading: "evaluating quants on
external corpora measures divergence on text the model may never produce
itself ... raw web text is so far out of distribution that the noise floor
inflates and KLD ordering degrades." Ordering is the entire question here.

So: same two quants, same reference, same rows -- scored on the model's own
sampled output instead. sc_trace already wrote that trace and the artifact
cache kept it, so nothing has to be regenerated.

Rents one pod, terminates it on every path.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blockquant.providers.runpod.provider import RunPodProvider

CANDIDATES = ["NVIDIA RTX A5000", "NVIDIA A40", "NVIDIA RTX A6000",
              "NVIDIA GeForce RTX 3090", "NVIDIA L40S", "NVIDIA RTX 4000 Ada Generation"]

FP16 = "Qwen/Qwen3.5-0.8B"
QUANTS = {
    "plain 3.0": "Honkware/Qwen3.5-0.8B-exl3-3.0bpw",
    "SC 3.0": "Honkware/Qwen3.5-0.8B-exl3-SC-3.0bpw-H6-V6",
}
TRACE_REPO = "Honkware/Qwen3.5-0.8B-exl3-selfcal"   # dataset repo, from the cache
ROWS = 40


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


pod = None
try:
    p, pod = rent(container_disk_gb=80, volume_gb=0)
    print("[pod] ssh:", p._get_ssh_endpoint(pod, timeout=900), flush=True)

    # The baked quant.py has no trace support; ship the current one.
    local = Path(__file__).resolve().parents[1] / "src" / "blockquant" / "remote" / "quant.py"
    p._upload_file(pod, local, "/opt/blockquant/quant.py")
    print("[upload] current quant.py shipped", flush=True)

    def sh(cmd, label, tail=30):
        r = p.run(pod, cmd)
        print(f"[{label}] exit={r['code']}", flush=True)
        for stream in (r["stdout"], r["stderr"]):
            t = "\n".join((stream or "").splitlines()[-tail:])
            if t.strip():
                print(t, flush=True)
        return r

    dl = ("from huggingface_hub import snapshot_download as d, hf_hub_download as f;"
          f"d('{FP16}', local_dir='/quant/fp16');")
    for i, repo in enumerate(QUANTS.values()):
        dl += f"d('{repo}', local_dir='/quant/q{i}');"
    dl += (f"print(f('{TRACE_REPO}', 'trace.json', repo_type='dataset',"
           f" local_dir='/quant/trace'))")
    sh(f'python -c "{dl}" 2>&1 | tail -3', "download")

    for i, (label, repo) in enumerate(QUANTS.items()):
        sh("cd /opt/blockquant && python -c \""
           "import importlib.util,json,sys;"
           "spec=importlib.util.spec_from_file_location('q','/opt/blockquant/quant.py');"
           "m=importlib.util.module_from_spec(spec);sys.modules['q']=m;spec.loader.exec_module(m);"
           "from pathlib import Path;"
           f"stats,method=m._kl_div_eval(Path('/quant/q{i}'),Path('/quant/fp16'),"
           f"rows={ROWS},seq_len=2048,trace_path=Path('/quant/trace/trace.json'));"
           "print('METHOD:',method);"
           "print('MEDIAN:',(stats or {}).get('kld_median'));"
           "print('MEAN:',(stats or {}).get('kld'))"
           "\" 2>&1 | tail -12", f"eval {label}")
finally:
    if pod:
        try:
            p.terminate(pod)
            print(f"[pod] terminated {pod}", flush=True)
        except Exception as e:
            print(f"[pod] TERMINATE FAILED {pod}: {e} -- CHECK CONSOLE", flush=True)
