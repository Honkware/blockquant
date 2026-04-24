# Experimental — Unconfirmed Providers & Scripts

Everything in this folder is **parked, not shipped**. Code here compiled at
some point and may have worked once, but nothing here is currently covered
by tests, docs, or an end-to-end validation run.

## What lives here

### `providers/`
- **`modal_provider.py`** + **`modal_app.py`** + **`patch_modal.py`** —
  Modal serverless provider. Reaches the stage of deploying a Modal app
  but the full quant → verify → upload path has never been observed to
  complete. Flash-attn install workaround in `modal_app.py` is a known
  sticking point.
- **`lambda_provider.py`** — Lambda Cloud provider. HTTP + SSH lifecycle
  similar to RunPod, but the GPU capacity in Lambda has been spotty and
  the provider hasn't been re-tested since the pipeline stabilised.

### `scripts/`
- **`run_modal_job.py`** — CLI analog of `run_runpod_job.py` for Modal.
- **`poll_modal.py`** / **`poll_modal2.py`** — one-off debug utilities.
- **`deploy_modal.py`** — deploys the Modal app.
- **`bootstrap_lambda.sh`** — shell script uploaded to Lambda pods during
  bootstrap; mirror of the Python bootstrap logic on the RunPod side.

## Un-shelving checklist (for whoever picks one of these up next)

1. Re-read the provider file top-to-bottom. Expect API drift in the
   upstream SDK and in the internal `runpod_provider.py` — the RunPod
   provider is the current reference implementation for the ABC shape,
   error handling, and bootstrap flow.
2. Add a unit test file under `backend/tests/providers/` mirroring the
   structure of `test_runpod_provider.py` (27 tests, full mock coverage).
3. Run an end-to-end job against a real account with a small, cheap
   model before wiring it back into the default path.
4. When all three are green, move the files back to
   `backend/src/blockquant/providers/` + `backend/scripts/`, re-register
   the provider in `providers/__init__.py`, and add the choice back to
   `src/commands/definitions.js`.

## Why this is better than deleting

Deleting would force the next person to rebuild from scratch. Keeping
them here preserves working-reference scaffolding while making it
impossible for an unsuspecting user to hit these paths through the
default pipeline.
