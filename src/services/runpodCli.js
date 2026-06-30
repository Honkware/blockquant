import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { getLogger } from '../logger.js';
import config from '../config.js';

const log = getLogger('runpod-cli');

const ROOT = config.ROOT_DIR;
const SCRIPT = path.join(ROOT, 'backend', 'scripts', 'run_runpod_job.py');
const PUBLISH = path.join(ROOT, 'backend', 'scripts', 'publish_quant.py');
const PYTHON = config.PYTHON_BIN || path.join(ROOT, 'backend', 'venv', 'bin', 'python');
const LOG_DIR = path.join(ROOT, 'backend', 'logs');

const RE = {
  pod: /Pod ID:\s*(\S+)/,
  bootstrap: /Bootstrap complete/,
  download: /\[download\]\s*(.+)/,
  quantizeStart: /\[quantize\]\s*([0-9.]+)\s*bpw/,
  // Real quantize progress emitted by remote/quant.py:
  // "[progress] quantize 5.0 42% (118/280) eta 12m"
  quantProgress: /\[progress\]\s*quantize\s+([0-9.]+)\s+(\d+)%(?:\s*\(([^)]*)\))?(?:\s*eta\s*(\S+))?/,
  layer: /Quantized:\s*\S*?layers\.(\d+)/,
  eta: /Estimated remaining time:\s*(.+)/,
  uploadDone: /\[upload\]\s*([0-9.]+)\s*(?:bpw\s*)?done\s*->\s*(https?:\/\/\S+)/,
  // Optional smoke-test reply, base64 so newlines/quotes survive the log relay.
  // The `b64` sentinel avoids matching the "[sample] N generating..." status line:
  // "[sample] 4.0 b64 <base64>"
  sample: /\[sample\]\s*([0-9.]+)\s+b64\s+([A-Za-z0-9+/=]+)/,
};

/**
 * Decide whether a failed controller run is worth re-spawning a pod for. ONLY a
 * genuine launch/stock failure is: a clean (non-signal) exit where no pod was
 * ever created. Two cases are terminal and must NOT respawn:
 *   - signal != null: the controller was killed (operator cancel of a broken
 *     model, or bot shutdown). A controller that hits a real error exits with a
 *     code; it never signals itself, so a signal is always a deliberate kill.
 *     Respawning here is the runaway: cancelling a job boots a fresh pod for it.
 *   - podId set: a pod booted and the run still failed, so the quant itself
 *     errored (e.g. an unsupported arch on this exllamav3) and another pod just
 *     fails the same way.
 * A transient network blip does NOT kill the controller (it surfaces as a
 * non-zero exit before any pod, handled inside run_runpod_job.py), so it stays
 * retryable.
 */
export function isControllerRetryable({ signal, podId }) {
  return signal == null && !podId;
}

/**
 * Run one variant on its own pod via `run`, retrying ONLY a transient launch
 * failure. A failure is retried when err.retryable !== false (set by runViaCli's
 * close handler / isControllerRetryable): a clean exit before any pod existed. A
 * signal-killed or pod-created failure is terminal, so a cancelled or broken job
 * never respawns a pod. `run(attempt)` is called once per attempt; `onRetry` (if
 * given) fires before each wait. Resolves run()'s result, or rethrows the last
 * error after the final attempt. `sleep` is injectable so tests don't wait.
 */
export async function runVariantWithRetry(
  variant,
  { run, maxAttempts = 3, retryDelayMs = 5000, onRetry, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) }
) {
  let lastErr;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await run(attempt);
    } catch (err) {
      lastErr = err;
      log.error(`variant ${variant} failed (attempt ${attempt}/${maxAttempts}): ${err.message}`);
      if (attempt < maxAttempts && err.retryable !== false) {
        onRetry?.(attempt + 1, maxAttempts);
        await sleep(retryDelayMs);
        continue;
      }
      break;
    }
  }
  throw lastErr;
}

/**
 * Run an EXL3 quant on RunPod by driving the hardened CLI
 * (backend/scripts/run_runpod_job.py), which auto-sizes disk, auto-selects a
 * cheap GPU, self-terminates the pod, and streams remote progress to stdout.
 *
 * Parses that stdout into aggregated progress across every requested variant
 * (so the embed shows the variant actually running, not just the first) and
 * resolves with one result row per variant.
 */
export function runViaCli({ modelId, variants, hfOrg, calRows = 250, testPrompt = null, onProgress }) {
  return new Promise((resolve, reject) => {
    const args = [
      SCRIPT,
      '--model', modelId,
      '--variants', variants.join(','),
      '--skip-local-exllama',
      '--cal-rows', String(calRows),
      // Cheapest card that fits, walking up on stock-outs. Without this the
      // CLI uses the profile's H100/A100 list and dies fast when those are
      // unavailable. Disk is auto-sized by the CLI (default).
      '--gpu', 'auto',
      '--min-vram', '24',
      // Cap $/hr scaled to model size: small models stay on cheap cards, a big
      // model reaches an A100/H100 (compute-bound, ~3x faster for ~same total
      // cost). The CLI also orders big-model candidates capable-first.
      '--max-price', 'auto',
      // Be patient: stock is tight and blips per-GPU, so keep re-sweeping for
      // ~20 min before giving up rather than failing after a few minutes.
      '--launch-retries', '24',
      '--launch-retry-delay', '25',
      // The default 60-min stall watchdog kills long quants whose log goes
      // quiet (SSH flakiness / a quiet phase) before they finish (~2-3h).
      // The pod self-terminates on success and max_runtime (8h) still backstops
      // a genuinely dead pod, so give the stall watchdog a wide 3h window.
      '--stall-timeout', '10800',
    ];
    if (hfOrg) args.push('--hf-org', hfOrg);
    // Optional pre-baked image (config.RUNPOD_IMAGE). Empty = bootstrap path.
    // The controller health-checks a baked image's ExLlamaV3 version and fails
    // fast if it is too old, so passing a stale image can't silently regress.
    if (config.RUNPOD_IMAGE) args.push('--image', config.RUNPOD_IMAGE);
    if (testPrompt) args.push('--test-prompt', testPrompt);

    // Detached, own process group, stdout+stderr to a log file. A bot restart
    // or crash then leaves the controller running: it finishes the quant,
    // uploads, and self-terminates its pod normally — instead of dying and
    // having its pod reaped (the pod name encodes this controller's pid). The
    // bot tails the log file for progress while it is alive. PYTHONUNBUFFERED
    // keeps the log line-buffered so the tail isn't ~8KB behind.
    fs.mkdirSync(LOG_DIR, { recursive: true });
    const slug = modelId.replace(/[^a-zA-Z0-9._-]/g, '_');
    const logPath = path.join(LOG_DIR, `ctrl-${slug}-${variants.join('-')}-${Date.now()}.log`);
    const logFd = fs.openSync(logPath, 'a');
    log.info(`spawn (detached): ${PYTHON} ${args.join(' ')} -> ${logPath}`);
    const child = spawn(PYTHON, args, {
      cwd: ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      detached: true,
      stdio: ['ignore', logFd, logFd],
    });
    child.unref();
    fs.closeSync(logFd);

    const total = variants.length;
    const results = new Map(); // bpw -> url
    const samples = new Map(); // bpw -> decoded smoke-test reply
    let curBpw = variants[0];
    let curDownloadPct = 0;   // from [download] N% (monotonic)
    let curQuantPct = 0;      // layer-based % from [progress] (monotonic)
    let lastOverall = 0;      // overall bar never moves backward
    let podId = '';
    let stage = 'Provisioning';

    // Each phase maps its REAL percent into a band of the overall bar, so the
    // bar climbs smoothly the whole run (download 4->25, quantize 25->90), and
    // a monotonic clamp keeps stale lines (the poll reprints its window) from
    // ever knocking the bar backward.
    const report = (message) => {
      let overall;
      switch (stage) {
        case 'Provisioning': overall = 4; break;
        case 'Downloading':  overall = 4 + Math.round((curDownloadPct / 100) * 21); break;
        case 'Quantizing':   overall = 25 + Math.round((curQuantPct / 100) * 65); break;
        case 'Uploading':    overall = 95; break;
        case 'Complete':     overall = 100; break;
        default:             overall = 0;
      }
      if (overall < lastOverall) overall = lastOverall;
      lastOverall = overall;
      onProgress?.({
        stage,
        message,
        overall,
        currentBPW: curBpw,
        bpwIndex: Math.max(0, variants.indexOf(curBpw)),
        totalBPWs: total,
        podId,
      });
    };

    function handleLine(line) {
      let m;
      if ((m = RE.uploadDone.exec(line))) {
        results.set(m[1], m[2]);
        stage = 'Uploading';
        return report(`Uploaded ${m[1]} bpw`);
      }
      if ((m = RE.sample.exec(line))) {
        try { samples.set(m[1], Buffer.from(m[2], 'base64').toString('utf8')); }
        catch { /* ignore a malformed marker */ }
        return;
      }
      if ((m = RE.quantProgress.exec(line))) {
        // pct is layer/total*100 (global, monotonic). "(preparing)" lines have
        // pct 0 during the measure phase -> bar holds at the band start.
        const pct = parseInt(m[2], 10);
        if (pct < curQuantPct) return;          // stale line the poll reprinted
        curBpw = m[1];
        curQuantPct = pct;
        stage = 'Quantizing';
        const eta = m[4] ? ` · eta ${m[4]}` : '';
        return report(`${pct}%${eta}`);
      }
      if ((m = RE.quantizeStart.exec(line))) {
        curBpw = m[1];
        stage = 'Quantizing';
        return report(`Quantizing ${m[1]} bpw`);
      }
      if (RE.bootstrap.test(line)) {
        stage = 'Downloading';
        return report('Bootstrap complete');
      }
      if ((m = RE.pod.exec(line))) {
        podId = m[1];
        return report(`Pod ${podId}`);
      }
      if ((m = RE.download.exec(line))) {
        stage = 'Downloading';
        const pm = m[1].match(/(\d+)\s*%/);
        if (pm) {
          const dp = parseInt(pm[1], 10);
          if (dp < curDownloadPct) return; // stale reprint
          curDownloadPct = dp;
        }
        return report(m[1].slice(0, 80));
      }
      if ((m = RE.eta.exec(line))) {
        // Strip exllamav3's "(avg over last 8 blocks)" suffix; it just gets
        // truncated in the embed. Leaves e.g. "ETA 1 hour, 14 minutes".
        const eta = m[1].replace(/\s*\(avg over[^)]*\)/i, '').trim();
        return report(`ETA ${eta.slice(0, 36)}`);
      }
    }

    // Tail the log file (the controller writes to it, not to a pipe) and feed
    // whole lines to the same parser. Polling is fine — progress lines arrive
    // every few seconds at most.
    let buf = '';
    let readOffset = 0;
    const drain = () => {
      let stat;
      try { stat = fs.statSync(logPath); } catch { return; }
      if (stat.size <= readOffset) return;
      const fd = fs.openSync(logPath, 'r');
      const b = Buffer.alloc(stat.size - readOffset);
      try { fs.readSync(fd, b, 0, b.length, readOffset); } finally { fs.closeSync(fd); }
      readOffset = stat.size;
      buf += b.toString('utf8');
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (process.env.BQ_DEBUG_RAW) log.info(`[raw] ${line.slice(0, 140)}`);
        try { handleLine(line); } catch (e) { log.debug(`parse: ${e.message}`); }
      }
    };
    const tailTimer = setInterval(drain, 1000);

    child.on('error', (err) => { clearInterval(tailTimer); reject(err); });
    // exit fires while the bot is alive (it still tracks the detached child). On
    // a deliberate kill, signal is set; a real error always exits with a code.
    child.on('exit', (code, signal) => {
      clearInterval(tailTimer);
      drain(); // flush any final lines (esp. the last [upload] done)
      const out = variants.map((v) => {
        const url = results.get(v) || null;
        return {
          bpw: v,
          variant: v,
          url,
          treeUrl: url,
          pushed: !!url,
          reused: false,
          duration: '',
          sample: samples.get(v) || null,
          error: url
            ? null
            : signal
              ? `cli killed (${signal})`
              : code === 0
                ? 'no upload URL seen'
                : `cli exit ${code}`,
        };
      });
      if (results.size > 0) {
        resolve(out); // full or partial success
      } else {
        const err = new Error(
          signal
            ? `run_runpod_job.py killed by ${signal} with no uploads`
            : `run_runpod_job.py exited ${code} with no uploads`
        );
        err.signal = signal || null;
        err.podCreated = !!podId;
        err.retryable = isControllerRetryable({ signal, podId });
        reject(err);
      }
    });
  });
}

/**
 * Re-sync every EXL3 card for a model and its collection. publish_quant.py
 * discovers ALL `-exl3-*bpw` repos for the base model on HF (no matter which
 * run/when they were made), rewrites each card with the full cross-linked
 * Quants table, and adds them all to the collection. Idempotent, so it's safe
 * to call after every job. Resolves { ok, collectionUrl }.
 */
export function finalizeCollection({ modelId, hfOrg }) {
  return new Promise((resolve) => {
    const args = [PUBLISH, '--base', modelId];
    if (hfOrg) args.push('--hf-org', hfOrg);
    log.info(`finalize: ${PYTHON} ${args.join(' ')}`);
    const child = spawn(PYTHON, args, { cwd: ROOT, env: { ...process.env, PYTHONUNBUFFERED: '1' } });
    let collectionUrl = '';
    let buf = '';
    child.stdout.on('data', (c) => {
      buf += c.toString();
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        const m = line.match(/collection:\s*(https?:\/\/\S+)/);
        if (m) collectionUrl = m[1];
      }
    });
    child.stderr.on('data', (c) => log.debug(`[publish] ${c.toString().trim().slice(0, 150)}`));
    child.on('error', (err) => {
      log.error(`finalize failed to spawn: ${err.message}`);
      resolve({ ok: false, collectionUrl: '' });
    });
    child.on('close', (code) => resolve({ ok: code === 0, collectionUrl }));
  });
}
