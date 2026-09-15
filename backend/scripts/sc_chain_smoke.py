"""Run the whole self-calibration chain once, small, and see what breaks.

trace -> rfn probe -> measure -> optimize -> convert with the recipe. Only
sc_measure has ever been executed here; the other three and the -rcp/-cd
conversion have not, and the orchestration to come is shaped entirely by what
they need from each other. Cheaper to find that out on a 0.6B than to design
around a guess.

Deliberately tiny: 8 calibration rows of 512 tokens instead of 250x2048, so
this is minutes rather than hours. It proves the stages connect, not that the
output is good.

Terminates the pod on every exit path.
"""
import os, sys
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


FP16 = "Qwen/Qwen3-0.6B"
DONOR = "Honkware/Qwen3.5-0.8B-exl3-4.0bpw"   # any >=4bpw quant works as trace donor
SC = "/opt/blockquant/selfcal"

pod = None
results = {}
results = {}
try:
    p, pod = rent(container_disk_gb=80, volume_gb=0)
    print("[pod] ssh:", p._get_ssh_endpoint(pod, timeout=900), flush=True)

    def sh(cmd, label, tail=18):
        r = p.run(pod, cmd)
        results[label] = r["code"]
        print(f"[{label}] exit={r['code']}", flush=True)
        for stream in (r["stdout"], r["stderr"]):
            t = "\n".join((stream or "").splitlines()[-tail:])
            if t.strip():
                print(t, flush=True)
        return r

    sh(f"python -c \"from huggingface_hub import snapshot_download as d;"
       f"d('{FP16}',local_dir='/quant/fp16');d('{DONOR}',local_dir='/quant/donor')\" 2>&1|tail -2",
       "download")

    sh(f"cd {SC} && timeout 1800 python sc_trace.py -m /quant/donor "
       f"-o /quant/trace.json -co /quant/cal.safetensors -cr 8 -cc 512 "
       f"--max_new_tokens 256 2>&1 | tail -12", "1_trace")

    sh(f"cd {SC} && timeout 1800 python sc_rfn_probe.py -mq /quant/donor -mr /quant/fp16 "
       f"-o /quant/rfn.json 2>&1 | tail -10", "2_rfn_probe")

    sh(f"cd {SC} && timeout 2400 python sc_measure.py -m /quant/fp16 -o /quant/measure.json "
       f"--streaming -r 2 -l 512 -tr /quant/trace.json 2>&1 | tail -10", "3_measure")

    sh(f"cd {SC} && timeout 900 python sc_optimize.py -m /quant/measure.json -b 4.0 "
       f"-hb 6 -o /quant/recipe.yaml 2>&1 | tail -12", "4_optimize")

    sh("head -12 /quant/recipe.yaml 2>/dev/null || echo NO_RECIPE", "5_recipe")
finally:
    print("STAGES:", results, flush=True)
    if pod:
        try:
            p.terminate(pod)
            print(f"[pod] terminated {pod}", flush=True)
        except Exception as e:
            print(f"[pod] TERMINATE FAILED {pod}: {e} -- CHECK CONSOLE", flush=True)
