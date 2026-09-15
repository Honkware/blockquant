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


# The donor has to be a quant OF THIS MODEL, not any quant. sc_rfn_probe walks
# the two module trees together ("top-level module count mismatch" otherwise),
# and sc_measure feeds the donor's trace through the fp16's embedding, so a
# different tokenizer indexes out of range. Learned the hard way by pairing a
# 0.8B quant with a 0.6B fp16.
FP16 = "Qwen/Qwen3.5-0.8B"
DONOR = "Honkware/Qwen3.5-0.8B-exl3-4.0bpw"
SC = "/opt/blockquant/selfcal"

pod = None
results = {}
results = {}
try:
    p, pod = rent(container_disk_gb=80, volume_gb=0)
    print("[pod] ssh:", p._get_ssh_endpoint(pod, timeout=900), flush=True)

    def sh(cmd, label, tail=18, produces=None):
        # `| tail` masks the real status, so a stage is judged by whether it
        # left the file the next stage needs.
        r = p.run(pod, "set -o pipefail; " + cmd)
        ok = r["code"] == 0
        if produces:
            chk = p.run(pod, f"test -s {produces} && echo YES || echo NO")
            ok = ok and "YES" in (chk["stdout"] or "")
            print(f"[{label}] produced {produces}: {'yes' if 'YES' in (chk['stdout'] or '') else 'NO'}", flush=True)
        results[label] = "ok" if ok else "FAILED"
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
       f"--max_new_tokens 256 2>&1 | tail -12", "1_trace", produces="/quant/cal.safetensors")

    sh(f"cd {SC} && timeout 1800 python sc_rfn_probe.py -mq /quant/donor -mr /quant/fp16 "
       f"-o /quant/rfn.json 2>&1 | tail -10", "2_rfn_probe", produces="/quant/rfn.json")

    sh(f"cd {SC} && timeout 2400 python sc_measure.py -m /quant/fp16 -o /quant/measure.json "
       f"--streaming -r 2 -l 512 -tr /quant/cal.safetensors 2>&1 | tail -10", "3_measure", produces="/quant/measure.json")

    sh(f"cd {SC} && timeout 900 python sc_optimize.py -m /quant/measure.json -b 4.0 "
       f"-hb 6 -o /quant/recipe.yaml 2>&1 | tail -12", "4_optimize", produces="/quant/recipe.yaml")

    sh("head -12 /quant/recipe.yaml 2>/dev/null || echo NO_RECIPE", "5_recipe")

    # The payoff: a recipe is only worth producing if convert will take it.
    # This is create_q_strategy_from_recipe, which has no expert-group handling
    # -- fine for dense, and the reason SC stays dense-only for now.
    # convert crops the calibration file to --cal_rows x --cal_cols and refuses
    # a file smaller than that, and those default to 250x2048. So a real SC job
    # needs sc_trace to have generated at least what the conversion will ask
    # for -- the 500k-token trace is a requirement, not a nicety. Here the
    # convert is shrunk to match the 8x512 trace instead, to keep this cheap.
    sh("timeout 3600 python -m exllamav3.conversion.convert_model "
       "-i /quant/fp16 -o /quant/sc-out -w /quant/sc-work "
       "-rcp /quant/recipe.yaml -cd /quant/cal.safetensors "
       "-cr 8 -cc 512 2>&1 | tail -30",
       "6_convert", produces="/quant/sc-out/config.json")
    sh("python -c \"import json;q=json.load(open('/quant/sc-out/config.json'))"
       "['quantization_config'];print({k:v for k,v in q.items() if not isinstance(v,dict)})\" "
       "2>/dev/null || echo NO_CONFIG", "7_quant_config")
finally:
    print("STAGES:", results, flush=True)
    if pod:
        try:
            p.terminate(pod)
            print(f"[pod] terminated {pod}", flush=True)
        except Exception as e:
            print(f"[pod] TERMINATE FAILED {pod}: {e} -- CHECK CONSOLE", flush=True)
