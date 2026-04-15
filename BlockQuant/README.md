# BlockQuant v2.0

Discord bot for quantizing HuggingFace models to **EXL3** format using [ExLlamaV3](https://github.com/turboderp-org/exllamav3), with automatic upload to HuggingFace Hub.
This branch runs in **HF-model-only** mode (GGUF conversion is not in the active pipeline).

## Features

- `/quant` — Submit a model for quantization with selectable BPW (bits-per-weight)
- Live progress updates in a dedicated Discord thread
- Pre-flight validation (HF token, model existence, write access) before any work starts
- Automatic HuggingFace repo creation & upload
- EXP system — users earn EXP by chatting, spend it on quantization jobs
- Admin commands for queue management and EXP grants
- Persistent jobs with restart recovery and idempotent EXP charge/refund ledger
- Local model cache reuse with prune controls
- Graceful shutdown with workspace cleanup
- Winston-based structured logging

## Prerequisites

- **Node.js** ≥ 20
- **Python** ≥ 3.10 with CUDA-capable PyTorch
- **ExLlamaV3** cloned separately and installed:
  ```bash
  git clone https://github.com/turboderp-org/exllamav3.git
  cd exllamav3
  python3 -m pip install --user -r requirements.txt
  python3 -m pip install --user -e .
  cd ..
  ```
- **Hugging Face helper script dependencies**:
  ```bash
  python3 -m pip install --user -r scripts/requirements.txt
  ```

## Setup

```bash
# 1. Clone and install Node deps
git clone <this-repo>
cd BlockQuant
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env with your Discord bot token, HF token, etc.
# Set EXLLAMAV3_DIR to your external exllamav3 clone path.

# 3. Install Python deps (required)
python3 -m pip install --user -r scripts/requirements.txt

# 4. Run
npm start        # production
npm run dev      # development (debug logging)
```

## Tooling

```bash
npm run lint         # lint JavaScript
npm run format:check # check formatting
npm run test         # run unit tests
npm run preflight    # run Python preflight script
```

## Reliability and Security

- **Persistent jobs:** queue state checkpoints are stored in `data/jobs.json`.
- **Startup recovery:** queued/running/interrupted jobs are recovered on boot.
- **Cache reuse:** downloaded source models are cached under `tmp/workdir/cache`.
- **Retries/timeouts:** HF and quantization subprocesses are governed by `.env` timeout/retry settings.
- **Secret hygiene:** logs and user-facing errors are redacted for token-like values.

## Project Structure

```
BlockQuant/
├── src/
│   ├── index.js              # Entry point — bot bootstrap
│   ├── config.js             # Validated env config
│   ├── logger.js             # Winston logger
│   ├── commands/
│   │   ├── definitions.js    # SlashCommandBuilder definitions
│   │   ├── router.js         # Command → handler dispatch
│   │   ├── quant.js          # /quant handler
│   │   ├── info.js           # /score, /leaderboard, /queue
│   │   └── admin.js          # /give, /pause, /resume
│   ├── services/
│   │   ├── db.js             # JSON file persistence
│   │   ├── workspace.js      # Filesystem workspace management
│   │   ├── huggingface.js    # HF download/upload/preflight
│   │   ├── quantizer.js      # ExLlamaV3 convert.py runner
│   │   └── queue.js          # PQueue job orchestration
│   └── utils/
│       ├── embeds.js         # Discord embed builders
│       └── format.js         # Progress bars, duration formatting
├── scripts/
│   ├── preflight.py          # HF token & model validation
│   ├── download_model.py     # Model downloader
│   ├── upload_model.py       # Model uploader
│   ├── generate_quant_readme.py # HF model card writer
│   ├── check_repo.py         # Post-upload verification
│   └── requirements.txt      # Python deps
├── tests/
│   └── format.test.js
├── data/                     # Auto-created JSON databases
├── logs/                     # Auto-created log files
├── .env.example
├── .gitignore
└── package.json
```

## Slash Commands

| Command        | Description               | Access |
| -------------- | ------------------------- | ------ |
| `/quant`       | Submit a quantization job | All    |
| `/queue`       | Check queue status        | All    |
| `/health`      | Check bot health summary  | All    |
| `/score`       | Check EXP balance         | All    |
| `/leaderboard` | Top 10 EXP earners        | All    |
| `/give`        | Grant/deduct EXP          | Admin  |
| `/pause`       | Pause the job queue       | Admin  |
| `/resume`      | Resume the job queue      | Admin  |
| `/diag`        | Detailed diagnostics      | Admin  |
| `/cache`       | Cache status/prune/clear  | Admin  |

## How Quantization Works

1. **Pre-flight** — Validates HF token has write access and the source model exists
2. **Download** — `snapshot_download()` pulls the model to a temp workspace
3. **Validate** — Checks for `config.json` and weight files
4. **Quantize** — Runs ExLlamaV3 `convert.py` (single-step EXL3 conversion, profile-aware)
5. **Upload** — `upload_folder()` pushes the quantized model to HuggingFace
6. **Manifest** — `blockquant-manifest.json` is written into output artifacts
7. **Cleanup** — Active workspace is wiped after each job (cache retained)

Each BPW in a job follows steps 4-5 sequentially. If any step fails, EXP is refunded and the user is notified.

## Differences from v1 (ExLlamaV2)

- **Single-step conversion** — No separate measurement pass; ExLlamaV3 computes Hessians on-the-fly
- **EXL3 format** — Based on QTIP, significantly better quality/size ratio
- **Simpler CLI** — `-i input -o output -w workdir -b bpw` is all you need
- **Pre-flight checks** — Token and model validation before any GPU work
- **Modular architecture** — Commands, services, and utilities are cleanly separated

## Quantization Guide

### Understanding BPW (Bits Per Weight)

BPW determines the compression level and quality of your quantized model:

| BPW | Quality | Use Case | Relative Size |
|-----|---------|----------|---------------|
| 3.0–3.5 | ⚠️ Low | VRAM-constrained only | ~40% of FP16 |
| **4.0–4.5** | ✅ Good | **Sweet spot** — most popular | ~50% of FP16 |
| **5.0–5.5** | ✅ Great | High quality | ~60% of FP16 |
| 6.0–6.5 | ⭐ Excellent | Near-lossless | ~70% of FP16 |
| 8.0 | ⭐ Premium | Minimal loss | ~90% of FP16 |

### Auto Mode (Recommended)

BlockQuant uses **Auto mode** by default, which intelligently adjusts parameters based on your chosen BPW:

```
/quant url: meta-llama/Llama-3.1-8B bpw: 4.0,5.0
```

Auto mode automatically sets:
- **head_bits** — Precision for the output layer (lm_head)
- **cal_rows/cal_cols** — Calibration data size for better quality

| BPW | head_bits | Calibration | Description |
|-----|-----------|-------------|-------------|
| ≤3.5 | 6 | 200×2048 | Fast, minimal VRAM |
| 4.0–4.5 | 6 | 250×2048 | **Balanced (default)** |
| 5.0–5.5 | 8 | 300×2560 | High quality |
| 6.0+ | 8 | 300×2560 | Higher lm_head precision |

### Manual Profiles

Override auto settings with profiles (optional):

```
/quant url: model bpw: 4.0 profile: quality   # Fixed head_bits=8
/quant url: model bpw: 4.0 profile: fast      # Fast conversion
```

| Profile | head_bits | Best For |
|---------|-----------|----------|
| `auto` | Scales with BPW | Most users (recommended) |
| `fast` | 6 | Quick conversion |
| `balanced` | 6 | Original behavior |
| `quality` | 8 | Maximum quality |

> **Note:** Admins can override head_bits directly with `head_bits: 4-16`.

### Technical Notes

**head_bits differences:**
Based on community testing (ExLlamaV2/V3), the difference between head_bits=6 and head_bits=8 is typically less than 1% in perplexity. The Auto mode uses head_bits=6 for most BPWs as the quality gain from head_bits=8 is minimal compared to the VRAM cost.

**Calibration data:**
The default calibration size (250 rows × 2048 columns) works well for most models. Increasing calibration data (e.g., to 300×2560) may marginally improve quality for high-BPW quants but significantly increases conversion time.

**Source format:**
EXL3 uses QTIP quantization. The calibration data is used to compute Hessians on-the-fly during conversion, unlike EXL2 which required a separate measurement pass.

### Recommended Workflows

**For VRAM-constrained (24GB GPU):**
```
/quant url: model bpw: 3.5        # Fits 70B models
```

**For balanced quality/size (most popular):**
```
/quant url: model bpw: 4.0,4.5    # ~90% quality, half the size
```

**For high-quality outputs:**
```
/quant url: model bpw: 5.0,6.0    # 95-98% quality
```

**Multiple variants at once:**
```
/quant url: model bpw: 3.5,4.0,4.5,5.0
```
