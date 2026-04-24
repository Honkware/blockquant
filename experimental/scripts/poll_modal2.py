"""Poll an existing Modal call."""
import os
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from blockquant.providers.modal_provider import ModalProvider

CALL_ID = sys.argv[1] if len(sys.argv) > 1 else "fc-01KPVXZ6G20QDV814NERJAGRD1"
LOG_FILE = Path(__file__).parent / "modal_job.log"

def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()

def main():
    provider = ModalProvider()
    provider._call_id = CALL_ID
    log(f"Polling existing call: {CALL_ID}")

    last_progress = ""
    while True:
        time.sleep(60)
        progress = provider.get_progress("modal-serverless")
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
    else:
        error = final.get("error", "Unknown error") if final else "No result"
        log(f"Failed: {error}")

    log("Done.")

if __name__ == "__main__":
    main()
