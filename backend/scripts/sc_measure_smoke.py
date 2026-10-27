"""Does the vendored sc_measure actually run in our image?

Rents one small pod, downloads a 1.5 GB model, runs sc_measure --streaming
over a couple of rows, and reports whether it emitted a measurement. Not an SC
pipeline -- just proof the script executes here, which nobody upstream can
check for us: UnstableLlama validated their rework on their setup, we vendored
a branch of it into our own torch/triton/exllamav3 pins.

Re-run this whenever EXLLAMAV3_REF moves or the vendored scripts are bumped:
they reach into exllamav3.modules.* internals, so a version bump can break them
silently and nothing else here executes them.

Terminates the pod on every exit path. Reads RUNPOD_API_KEY, RUNPOD_IMAGE and
HF_TOKEN from the environment; run it with `set -a; . ./.env; set +a`.

Last green: exllamav3 1.5.0, 197 tensors measured on Qwen3-0.6B, scaling
exponent median 2.01 (the quadratic the error model predicts).
"""
import os, sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blockquant.providers.runpod.provider import RunPodProvider

MODEL = "Qwen/Qwen3-0.6B"
IMAGE = os.environ["RUNPOD_IMAGE"]
p = RunPodProvider(api_key=os.environ["RUNPOD_API_KEY"], gpu_type="NVIDIA RTX A5000",
                   image=IMAGE, container_disk_gb=60, volume_gb=0)
pod = None
try:
    pod = p.launch({"env": {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}})
    print(f"[pod] {pod}", flush=True)
    p.wait_for_active(pod, timeout=900)
    ep = p._get_ssh_endpoint(pod, timeout=900)
    print(f"[pod] ssh up: {ep}", flush=True)

    def sh(cmd, label):
        r = p.run(pod, cmd)
        print(f"[{label}] exit={r['code']}", flush=True)
        out = (r["stdout"] or "")[-1500:]
        if out.strip():
            print(out, flush=True)
        if r["code"] != 0:
            print((r["stderr"] or "")[-1500:], flush=True)
        return r

    sh("echo \"arch=[$TORCH_CUDA_ARCH_LIST] ds=[$HF_DATASETS_CACHE]\"; "
       "python -c 'import exllamav3,importlib.metadata as m;print(\"exllamav3\",m.version(\"exllamav3\"))'",
       "env")
    sh(f"HF_TOKEN=$HF_TOKEN python -c \"from huggingface_hub import snapshot_download;"
       f"snapshot_download('{MODEL}', local_dir='/quant/m')\" 2>&1 | tail -2", "download")
    # A couple of rows, short, so this costs minutes rather than an hour.
    sh("cd /opt/blockquant/selfcal && timeout 900 python sc_measure.py "
       "-m /quant/m -o /quant/measure.json --streaming -r 2 -l 512 2>&1 | tail -25",
       "sc_measure")
    r = sh("python -c \"import json;d=json.load(open('/quant/measure.json'));"
           "print('rows measured:', len(d.get('results',[])));"
           "print('first:', json.dumps(d['results'][0])[:200] if d.get('results') else 'NONE')\"",
           "result")
    print("VERDICT:", "sc_measure RAN" if r["code"] == 0 else "sc_measure DID NOT PRODUCE OUTPUT", flush=True)
finally:
    if pod:
        try:
            p.terminate(pod)
            print(f"[pod] terminated {pod}", flush=True)
        except Exception as e:
            print(f"[pod] TERMINATE FAILED {pod}: {e} -- CHECK RUNPOD CONSOLE", flush=True)
