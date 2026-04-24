# Contributing to BlockQuant

Thanks for your interest. BlockQuant is small and opinionated; the
fastest path to a merged PR is to match the existing shapes.

## Setup

```bash
git clone https://github.com/Honkware/blockquant.git
cd blockquant

# Vendored upstream — kept out of this repo
git clone https://github.com/turboderp-org/exllamav3.git

# Python backend
cd backend
python -m venv venv
./venv/bin/pip install -r requirements.txt

# Node bot (optional — only if you're touching the Discord side)
cd .. && npm install

# Config
cp .env.example .env
# Fill in HF_TOKEN at minimum; RUNPOD_API_KEY if working on the runpod provider
```

## Running tests

```bash
cd backend
./venv/bin/pytest tests/providers -v
```

The provider tests are fully mocked — no GPU, no real RunPod / HF API
calls — so they run anywhere CI can install the deps. `pytest` against
`tests/providers/test_runpod_provider.py` should report 27 passing.

CI (`.github/workflows/test.yml`) runs the same tests on Python 3.10,
3.11, and 3.12 on every push and PR.

## Adding a new provider

The provider ABC lives at
[`backend/src/blockquant/providers/base.py`](backend/src/blockquant/providers/base.py).
Three methods are required (`launch`, `terminate`, `run`), seven more
are optional with safe defaults (`wait_for_active`, `bootstrap`,
`run_pipeline`, `get_progress`, `is_pipeline_running`, `sync_outputs`,
`get_result`, `get_cost_per_hour`).

Reference implementation: **[`runpod_provider.py`](backend/src/blockquant/providers/runpod_provider.py)** —
the canonical example of the SSH + retry-on-transient-network-error
pattern. Read it before writing a new provider.

Checklist for a new provider PR:

1. **Implement the ABC.** Subclass `Provider`, fill in the three required
   methods, override the optional hooks your backend actually needs.
2. **Wrap every network-touching call** in retry+reconnect for transient
   errors. The four resilience patterns are visible in
   `runpod_provider.py`: `_get_pod_resilient`, `run()`'s retry wrapper,
   `_sftp_put_with_retry`, and `_upload_directory`'s per-file retry.
3. **Add a test file** at `backend/tests/providers/test_{name}_provider.py`
   modelled on `test_runpod_provider.py` (~25 tests, full mock coverage).
4. **Register the provider** in
   [`backend/src/blockquant/providers/__init__.py`](backend/src/blockquant/providers/__init__.py).
5. **Add the CLI option** in
   [`src/commands/definitions.js`](src/commands/definitions.js)
   under the `provider` choices.
6. **Add config fields** to `QuantConfig` in
   [`backend/src/blockquant/models.py`](backend/src/blockquant/models.py)
   and the corresponding pass-through in
   [`backend/src/api/main.py`](backend/src/api/main.py).
7. **Document the env vars** in `.env.example`.

## Resurrecting a shelved provider

Modal and Lambda are parked under
[`experimental/`](experimental/). Before un-shelving:

1. Re-read the provider file top-to-bottom; expect API drift since it
   was last touched.
2. Run an end-to-end job against a real account with a small, cheap
   model.
3. Add tests (mirror `test_runpod_provider.py`).
4. Move files back to `backend/src/blockquant/providers/` and
   `backend/scripts/`, re-register, re-add the CLI option.

See [`experimental/README.md`](experimental/README.md) for the full
checklist.

## Code style

- Python 3.10+ type hints (`X | None`, not `Optional[X]`).
- No comments stating *what* the code does; only *why* when non-obvious.
- Tests are mocked-only — no real network calls in `tests/`.
- Provider error messages should tell the user *what to do*, not just
  *what failed*.

## What gets rejected

- New providers without tests.
- Code that hardcodes a model / org / token / pod ID.
- Anything that re-introduces a `hasattr(provider, ...)` check —
  extend the ABC instead.
- Adding back debug scripts with API keys baked in.
- A `CLAUDE.md` written specifically for this repo (out of scope).

## Filing issues

For bug reports: include the relevant log tail (use the dashboard's
"raw tail" view if it's a quant-pipeline issue), your provider name +
GPU choice, and the exact command you ran.

For feature requests: especially welcome are new provider implementations
(Vast.ai, Hyperstack, Crusoe), Hessian/calibration cache tooling, and
provenance-receipt schema work — see the project notes for more.
