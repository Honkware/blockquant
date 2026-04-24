"""Submit a Modal job and poll until completion."""
import os
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from blockquant.providers.modal_provider import ModalProvider

LOG_FILE = Path(__file__).parent / "modal_job.log"

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    model_id = "lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
    fmt = "exl3"
    variants = ["4.0"]
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_org = "blockblockblock"
    head_bits = 8

    if not hf_token:
        log("ERROR: HF_TOKEN not set")
        sys.exit(1)

    log(f"Starting Modal job for {model_id} ({fmt}, {variants})")
    log(f"Upload target: {hf_org}/{model_id.split('/')[-1]}-exl3")

    provider = ModalProvider()
    instance_id = provider.launch({})
    log(f"Instance ID: {instance_id}")

    result = provider.run_pipeline(
        instance_id=instance_id,
        model_id=model_id,
        format=fmt,
        variants=variants,
        hf_token=hf_token,
        hf_org=hf_org,
        head_bits=head_bits,
        use_imatrix=True,
    )
    call_id = result.get("call_id", "unknown")
    log(f"Spawned call: {call_id}")

    # Poll every 60 seconds
    last_progress = ""
    while True:
        time.sleep(60)
        progress = provider.get_progress(instance_id)
        if progress != last_progress:
            log(f"Progress: {progress}")
            last_progress = progress
        if "COMPLETE" in progress:
            break
        if "ERROR" in progress or "STATUS_ERROR" in progress:
            log("Job encountered an error.")
            break

    final = provider.get_result()
    log(f"Final result: {final}")

    if final and final.get("status") == "complete":
        outputs = final.get("outputs", [])
        log(f"Success! Outputs: {outputs}")
        for out in outputs:
            log(f"  - {out['variant']}: {out['path']}")
    else:
        error = final.get("error", "Unknown error") if final else "No result"
        log(f"Failed: {error}")

    log("Done.")

if __name__ == "__main__":
    main()
