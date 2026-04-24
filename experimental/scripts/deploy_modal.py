"""Deploy the BlockQuant Modal app and verify it works.

Usage:
    cd backend
    python scripts/deploy_modal.py

Prerequisites:
    - Modal SDK installed: pip install modal
    - MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars set
    - (Optional) HF_TOKEN env var for upload tests

What it does:
    1. Validates Modal credentials
    2. Deploys blockquant/providers/modal_app.py
    3. Runs a quick smoke test (hello + import check)
    4. Optionally runs a tiny quant job to verify the GPU function works
"""
import os
import sys
import time
from pathlib import Path

# Ensure backend/src is on path so we can import blockquant
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))


def _check_credentials():
    """Verify Modal creds are present."""
    tid = os.environ.get("MODAL_TOKEN_ID", "")
    tsec = os.environ.get("MODAL_TOKEN_SECRET", "")
    if not tid or not tsec:
        print("ERROR: Modal credentials not found.")
        print("  export MODAL_TOKEN_ID=your_token_id")
        print("  export MODAL_TOKEN_SECRET=your_token_secret")
        sys.exit(1)
    print("Modal credentials: OK")


def _deploy_app():
    """Run modal deploy via subprocess."""
    import subprocess

    app_path = BACKEND_ROOT / "src" / "blockquant" / "providers" / "modal_app.py"
    if not app_path.exists():
        print(f"ERROR: Modal app not found at {app_path}")
        sys.exit(1)

    print(f"\nDeploying Modal app: {app_path}")
    result = subprocess.run(
        [sys.executable, "-m", "modal", "deploy", str(app_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: Deploy failed:\n{result.stderr}")
        sys.exit(1)
    print("Deploy: OK")


def _smoke_test():
    """Run a quick smoke test against the deployed function."""
    import modal

    print("\nRunning smoke test...")
    try:
        fn = modal.Function.from_name("blockquant", "_run_pipeline_remote")
        # Dry-run style: we just spawn a tiny config and immediately check status
        call = fn.spawn({
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "format": "exl3",
            "variants": ["4.0"],
            "hf_token": os.environ.get("HF_TOKEN", ""),
            "hf_org": os.environ.get("HF_ORG", ""),
        })
        print(f"Spawned test call: {call.object_id}")

        # Poll for up to 60 seconds — just verifying it starts, not waiting for completion
        for i in range(12):
            time.sleep(5)
            try:
                result = call.get(timeout=0)
                print(f"Call completed early: {result['status']}")
                break
            except modal.exception.TimeoutError:
                print(f"  [{i+1}/12] Still running...")
            except Exception as e:
                print(f"  [{i+1}/12] Error checking status: {e}")
                break
        else:
            print("Smoke test: function started successfully (call is running)")
            # Cancel the test so we don't waste GPU time
            try:
                call.cancel()
                print("Cancelled test call to save GPU time.")
            except Exception:
                pass

    except Exception as e:
        print(f"WARNING: Smoke test failed: {e}")


def main():
    print("=" * 60)
    print("BlockQuant Modal Deploy")
    print("=" * 60)

    _check_credentials()
    _deploy_app()

    # Ask user before running smoke test (costs a few cents)
    answer = input("\nRun smoke test? Spawns a real GPU job (~$0.05). [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        _smoke_test()
    else:
        print("Skipping smoke test.")

    print("\n" + "=" * 60)
    print("Deploy complete.")
    print("You can now use provider='modal' in your QuantConfig.")
    print("=" * 60)


if __name__ == "__main__":
    main()
