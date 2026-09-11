# BlockQuant Setup Guide (Windows + RTX 4090)

## Prerequisites

- Windows 10/11
- NVIDIA RTX 4090 (or similar CUDA-capable GPU)
- Python 3.10
- Node.js 20+
- Git

## 1. Clone and Enter the Repo

```powershell
cd path\to\blockquant
```

## 2. Python Backend

### 2.1 Create Virtual Environment

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2.2 Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2.3 Verify ExLlamaV3

ExLlamaV3 should be cloned at `exllamav3/` beside the repo root (set `EXLLAMAV3_DIR` if it lives elsewhere). The pipeline will auto-download calibration data on first run, but you can verify:

```powershell
python -c "import exllamav3; print('ExLlamaV3 OK')"
```

### 2.4 Start Redis

Redis is bundled in `backend/redis-win/`:

```powershell
cd redis-win\redis
.\redis-server
```

Leave this running in its own terminal.

### 2.5 Start FastAPI Server

In a new terminal (with venv activated):

```powershell
cd backend
$env:HF_TOKEN = "your_huggingface_token"
$env:EXLLAMAV3_DIR = "path\to\blockquant\exllamav3"
.\venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### 2.6 Start Celery Worker

In another new terminal (with venv activated):

```powershell
cd backend
$env:HF_TOKEN = "your_huggingface_token"
$env:EXLLAMAV3_DIR = "path\to\blockquant\exllamav3"
.\venv\Scripts\python -m celery -A scheduler.tasks worker --loglevel=info -n windows@%h -P solo
```

> **Windows Note:** Always use `-P solo` on Windows. The default prefork pool causes semaphore errors.

## 3. Discord Bot

### 3.1 Install Node Dependencies

```powershell
cd path\to\blockquant
npm install
```

### 3.2 Configure Environment

Create `.env` in the `BlockQuant/` directory:

```env
BOT_TOKEN=your_discord_bot_token
CLIENT_ID=your_discord_app_client_id
GUILD_ID=your_discord_server_id
BLOCKQUANT_API_URL=http://localhost:8000
HF_TOKEN=your_huggingface_token
```

### 3.3 Start the Bot

```powershell
cd path\to\blockquant
node src/index.js
```

The bot will register slash commands on startup and appear online in your Discord server.

## 4. Verify Everything Works

### 4.1 API Health Check

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
# Should return: @{status=ok; celery=true}
```

### 4.2 Dashboard

Open http://localhost:8000/dashboard/ in your browser.

### 4.3 Self-Test Harness

```powershell
cd path\to\blockquant
node self-test.mjs
```

## 5. Optional: RunPod Cloud Provider

1. Sign up at https://runpod.io and get an API key from your dashboard
2. Generate an SSH key pair (if you don't have one):
   ```powershell
   ssh-keygen -t rsa -b 4096 -f $HOME\.ssh\id_rsa -N ""
   ```
3. Add your public key to RunPod:
   ```powershell
   runpod ssh add-key --public-key $HOME\.ssh\id_rsa.pub
   ```
4. Set `RUNPOD_API_KEY` in your environment:
   ```powershell
   $env:RUNPOD_API_KEY = "your_runpod_api_key"
   ```
5. Use `/quant` with `provider: runpod` in Discord, or run the CLI:
   ```powershell
   cd backend\scripts
   python run_runpod_job.py --model <model_id> --variants 4.5 --gpu "NVIDIA H100 80GB"
   ```

## Deploying a change

A `/quant` run takes hours and a `/catbench` run takes 20 minutes, both on paid
GPUs, so check before you restart:

```bash
npm run --silent status && pm2 restart blockquant
```

`npm run status` lists every controller still running and exits non-zero if
there is one, so the `&&` holds the restart back. Restarting anyway is safe for
the work itself — controllers are reparented to init and pm2's tree kill cannot
reach them (`src/services/detached.js`) — but it is not safe for the Discord
side of it:

- the progress embed for a running job stops updating and never completes;
- for `/quant`, the repo cards and the collection are written by the bot after
  the upload, so they are skipped. Re-run them by hand once the quant lands:
  `backend/venv/bin/python backend/scripts/publish_quant.py --base <model_id>`;
- for `/catbench`, the result JSON is left in `backend/logs/` and nothing posts it.

The bot logs both cases on the next boot ("controller survived the restart",
"finished while the bot was down") with the log file to watch.

## Troubleshooting

### "ExLlamaV3 convert.py not found"

Set `EXLLAMAV3_DIR` environment variable to the absolute path of your `exllamav3/` directory.

### "Celery not configured" API error

The Celery worker isn't running or can't connect to Redis. Check:
- Redis is running on `localhost:6379`
- Celery worker started with `-P solo`
- No import errors in Celery logs

### Discord commands not appearing

- Make sure `CLIENT_ID` and `GUILD_ID` are correct in `.env`
- The bot needs `applications.commands` scope in your Discord app
- Restart the bot after adding new commands

### Download progress stuck

Large models (35B+) can take 20-40 minutes to download. The progress updates every 2 seconds. If it seems truly frozen, check the Celery worker logs for HuggingFace Hub errors.

### Quantize stage seems stuck

ExLlamaV3 quantize can take hours for large models. The streaming progress updates every 2 seconds. Check `backend/logs/celery.err` for actual errors.
