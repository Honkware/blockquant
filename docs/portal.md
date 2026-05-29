# Quant Request Portal (design)

A minimal, polished "request a quant" experience in Discord. Requesters give
two things; admins approve; the job runs in a dedicated thread that links the
finished models. The expertise lives in the defaults, not in the UI.

## Principles

- **Two inputs, enforced:** a HuggingFace model (URL or `user/model`) and at
  least one BPW. No submission is possible without a BPW.
- **Minimal UI, good defaults:** architecture-aware settings are applied
  silently (see Defaults). Requesters never see a knob.
- **Bot = portal, backend = engine:** the Node bot owns submission, approval,
  threads, and progress. It never drives RunPod directly; on approve it calls
  the existing FastAPI -> Celery -> pipeline path, which owns the pod lifecycle
  (the verified terminate, watchdog, and self-terminate backstop already built).
- **No EXP.** Approval is the gate; the EXP charge/refund system is removed.

## Lifecycle

```
/quant  ->  PENDING  --approve-->  (preflight)  ->  RUNNING  ->  COMPLETED | PARTIAL | FAILED
                       --reject-->  REJECTED
   requester/admin may CANCEL while PENDING or RUNNING (cancel terminates pods)
```

Per request, a **batch** of one variant-job per selected BPW. One approval for
the whole batch; one thread; one living embed.

## Discord UX

### Submission
Discord modals cannot contain select menus, so the multi-select cannot live in
a modal. Pattern:

1. `/quant model:<url-or-repo-id>`: model is a required slash option,
   normalized + validated (reuse the existing model-id validators).
2. Ephemeral reply with a **BPW StringSelect (`min_values: 1`)** and a
   **Submit** button that stays disabled until at least one BPW is picked.
   This hard-enforces "no submission without a BPW."
3. Light preflight at submit: HF `model_info` to confirm the model exists and
   is accessible with our token. Fail fast with a clear reason.

### Approval (per-user admins)
Request posts a compact embed to `#quant-requests` with **Approve / Reject**
buttons. Handlers check `config.ADMIN_IDS.includes(interaction.user.id)`
(already the model today) and deny others ephemerally. The embed shows the
model, the BPWs, the requester, the **cost estimate + RunPod balance** and a
green/red sufficiency flag.

### Thread (source of truth)
On approve, create a thread named `quant · Model-Name · 4.0,5.0,6.0`. Set max
auto-archive (7 days) and post a lightweight keep-alive per stage, because
**editing an embed does not reset Discord's archive timer**. The thread holds
one living **batch embed** with a per-variant row, edited in place (throttled
to ~once / 8s). Buttons: **Cancel** (requester or admin), **Retry failed**
(admin, on partial/fail).

### Progress embed (mockup)
```
⚙️  Quantizing · Huihui-Qwen3.6-35B-A3B-abliterated     (title -> HF)
Requested by @user · RunPod A5000 · started <t:..:R>

4.0   ██████████  done        19.5 GB   ✅
5.0   ██████░░░░  62%  quantizing
6.0   ░░░░░░░░░░  queued
─────────────────────────────────
Elapsed 14m · est. $1.40 / $40.00 balance        BlockQuant · req #a1c3
```
Per-variant rows reuse `progressBar(pct, 10)` with status icons
(queued / running / uploading / done / failed). v1 is stage-level; rich
per-layer % + ETA is a follow-up that plumbs the pod telemetry (already parsed
by `log_dashboard.py`) up through `/jobs/{id}`.

### Terminal states
- **Complete:** green; per-bpw result lines with links + durations (the
  existing `jobComplete` layout) plus the collection link; pin it.
- **Partial:** yellow; links the variants that landed; **Retry failed**.
- **Failed:** red; redacted reason via the existing `sanitizeErrorText`.

## Execution

Default: **one pod, sequential variants** (the pipeline already loops variants,
so it downloads once). Admin **Expedite** toggle switches to the parallel
one-pod-per-BPW path (`run_parallel_quants.py` + finalize) when wall-clock
matters. `--gpu auto` lands the cheapest in-stock card that fits (the quant
peaks ~4GB VRAM, so the limiter is disk, not VRAM).

## Money safety

At approval, and re-checked at launch:
- Estimate cost per variant and sum it.
- Block when `balance < worst_case`, where `worst_case = max_runtime * rate *
  variants` (the watchdog cap bounds the real exposure). Show estimate + cap.
- **Committed-spend ledger:** subtract in-flight approved batches' worst-case
  so concurrent approvals cannot each see the full balance and overspend.
- **Daily spend cap** circuit breaker.
- New backend primitive: `RunPodProvider.get_balance()` (GraphQL
  `myself { clientBalance }`), exposed via a small FastAPI route.

## Defaults (invisible, architecture-aware)

Derived from the model `config.json`:
- `head_bits = 8` (output-layer precision matters, especially for small models).
- **MoE** detected -> MoE-appropriate calibration + `--hq` (raises attention /
  shared-expert precision cheaply; needs wiring into the remote path).
- Calibration rows bounded by the bundled corpus (250 default; do not exceed
  the corpus or convert raises an IndexError).
- One download, sequential variants.

## Backend / bot changes

- **Per-user admins:** reuse `config.ADMIN_IDS`; add `/admin add|remove|list`
  persisting to the bot store, env list as fallback. User IDs, never roles.
- **Request store** (`db.js`): `{requestId, userId, model, variants[], state,
  threadId, approvalMsgId, costEstimate, perVariant:{bpw:{jobId,status,hfUrl}},
  decidedBy, createdAt}`. Persisted; recovered on bot restart (reuse the
  `recoverPersistedJobs` pattern in `index.js`).
- **API:** request carries a `batch_id`; `/jobs/{id}` returns per-variant detail.
- **Remove EXP** from `db.js` and `quant.js`; replace its throttle with a
  per-user pending-request cap.

## Edge cases

failure / partial batch (mark + link successes + Retry failed; failure path
self-terminates the pod) · insufficient credits (blocked at approval) ·
duplicate request (warn if `{org}/{model}-exl3-{bpw}bpw` already exists or an
open request matches) · cancellation (verified terminate on any live pod) ·
concurrent jobs (`--gpu auto` fall-through, committed-spend ledger) ·
permission errors (ephemeral denials, HF write-token check) · network / SSH
blips (retry + watchdog + backstop) · bot restart (recovery).

## Roadmap

1. Request model + states in `db.js`; `/quant` creates a PENDING request; EXP removed.
2. Submission UX (model option + required BPW multi-select + Submit).
3. Approval surface (`#quant-requests` embed + Approve/Reject, per-user admin) + `/admin`.
4. Threads + per-variant batch embed wired to the poll loop.
5. Cost + balance preflight (`get_balance()` + estimate + route); block/warn.
6. Final success state (links + collection); PARTIAL + Retry failed.
7. Edge-case hardening (cancel->terminate, duplicate detect, failure self-terminate, launch-time balance recheck).

## Deferred (not MVP)

KL/PPL fidelity reports + the bpw size/quality sweep table · domain-matched
calibration · smoke-test gating · rich per-layer % + ETA · `/my-requests` ·
audit trail · `#releases` announce · daily-cap UI · expedite/parallel toggle.
All bolt on later without a rewrite.
