# RunPod notes

The RunPod provider is for running the same backend pipeline on an ephemeral GPU pod. It launches a pod, waits for SSH, uploads the remote runner, starts the quant job, polls progress, syncs outputs, and terminates the pod unless told not to.

Implementation lives in:

```text
backend/src/blockquant/providers/runpod/
```

The old `backend/src/blockquant/providers/runpod_provider.py` file is only a compatibility import.

## Required environment

```dotenv
HF_TOKEN=...
RUNPOD_API_KEY=...
EXLLAMAV3_DIR=./exllamav3
```

The provider also expects an SSH key, defaulting to:

```text
~/.ssh/id_rsa
~/.ssh/id_rsa.pub
```

The public key is injected into the pod on create. The private key is used by Paramiko for SSH/SFTP.

## CLI path

```bash
./backend/venv/bin/python backend/scripts/run_runpod_job.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --variants 4.5 \
  --profile balanced \
  --gpu "NVIDIA H100 NVL" \
  --gpu-fallback "NVIDIA H100 PCIe,NVIDIA A100-SXM4-80GB" \
  --hf-org ""
```

Dry-run the resolved config and estimated cost:

```bash
./backend/venv/bin/python backend/scripts/run_runpod_job.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --variants 4.5 \
  --profile balanced \
  --tune
```

Use the prebuilt image to avoid the normal bootstrap step:

```bash
./backend/venv/bin/python backend/scripts/run_runpod_job.py \
  --image ghcr.io/honkware/blockquant:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --variants 4.5
```

## Profiles

`--profile` is a convenience preset. Individual flags still win.

| Profile | Use when | Notes |
|---|---|---|
| `fast` | cheap test quant or smoke run | smaller calibration set |
| `balanced` | normal run | default |
| `quality` | final publish run | larger calibration set and more conservative cloud preference |

Useful overrides:

- `--cal-rows`
- `--cal-cols`
- `--cloud COMMUNITY|SECURE`
- `--gpu`
- `--gpu-fallback`
- `--network-volume-id`
- `--data-center-id`

## GPU fallback

The CLI tries `--gpu` first, then each value in `--gpu-fallback`. It only falls through on capacity/out-of-stock errors. Other errors abort because they usually mean bad credentials, bad image config, or a provider-side problem that retrying on another GPU will not fix.

## Pod lifecycle

Normal flow:

1. create pod with the requested GPU/image/disk settings
2. wait for SSH
3. bootstrap dependencies unless using the prebuilt image or an already-bootstrapped pod
4. upload the remote quant script
5. run the remote pipeline
6. poll logs/result metadata
7. sync output files and result JSON
8. terminate pod

Use `--keep-pod` when debugging. Remember to terminate the pod manually afterward.

## Remote files

The provider keeps pod-local files under `/root`:

| Path | Use |
|---|---|
| `/root/quant.py` | uploaded remote runner |
| `/root/bq.log` | remote job log |
| `/root/bq-result.json` | remote result payload |
| `/root/.bq-bootstrapped` | bootstrap marker |

These constants live in `backend/src/blockquant/providers/runpod/constants.py`.

## Pricing

`backend/src/blockquant/providers/runpod/pricing.py` first asks the RunPod SDK for live pricing. If that fails, it falls back to the small static table in the same file. Treat CLI estimates as planning numbers, not billing records.

## Failure handling

The provider has retries for transient RunPod API, SSH, and SFTP failures. Tests cover:

- transient `get_pod` retry
- non-transient `get_pod` no-retry
- SSH exec retry
- SFTP retry
- tokens not appearing on command lines

Run the provider tests without touching real cloud resources:

```bash
PYTHONPATH=backend/src .venv/bin/pytest backend/tests/providers -q
```

## Token handling

The RunPod API key and HF token are passed through controlled environment/config paths. They should not be placed on remote shell command lines or persisted into receipts/manifests. Regression tests check both paths.
