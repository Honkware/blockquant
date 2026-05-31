import { spawn } from 'node:child_process';
import path from 'node:path';
import { getLogger } from '../logger.js';
import config from '../config.js';

const log = getLogger('runpod-cli');

const ROOT = config.ROOT_DIR;
const SCRIPT = path.join(ROOT, 'backend', 'scripts', 'run_runpod_job.py');
const PUBLISH = path.join(ROOT, 'backend', 'scripts', 'publish_quant.py');
const PYTHON = config.PYTHON_BIN || path.join(ROOT, 'backend', 'venv', 'bin', 'python');

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
};

/**
 * Run an EXL3 quant on RunPod by driving the hardened CLI
 * (backend/scripts/run_runpod_job.py), which auto-sizes disk, auto-selects a
 * cheap GPU, self-terminates the pod, and streams remote progress to stdout.
 *
 * Parses that stdout into aggregated progress across every requested variant
 * (so the embed shows the variant actually running, not just the first) and
 * resolves with one result row per variant.
 */
export function runViaCli({ modelId, variants, hfOrg, calRows = 250, onProgress }) {
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

    log.info(`spawn: ${PYTHON} ${args.join(' ')}`);
    // PYTHONUNBUFFERED so the child's stdout (which drives the progress embed)
    // streams line-by-line instead of block-buffering ~8KB when piped, which
    // freezes the embed through the quiet bootstrap/download phases.
    const child = spawn(PYTHON, args, {
      cwd: ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    const total = variants.length;
    const results = new Map(); // bpw -> url
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

    let buf = '';
    child.stdout.on('data', (chunk) => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (process.env.BQ_DEBUG_RAW) log.info(`[raw] ${line.slice(0, 140)}`);
        try { handleLine(line); } catch (e) { log.debug(`parse: ${e.message}`); }
      }
    });
    child.stderr.on('data', (c) => log.debug(`[cli] ${c.toString().trim().slice(0, 200)}`));

    child.on('error', reject);
    child.on('close', (code) => {
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
          error: url ? null : code === 0 ? 'no upload URL seen' : `cli exit ${code}`,
        };
      });
      if (results.size > 0) {
        resolve(out); // full or partial success
      } else {
        const err = new Error(`run_runpod_job.py exited ${code} with no uploads`);
        // Only a launch/stock failure (we never got a pod) is worth retrying.
        // If a pod WAS created and the run still failed, the quant itself
        // errored (e.g. an unsupported arch on this exllamav3) — retrying just
        // boots another pod that fails the same way, which is the runaway we
        // saw. Mark those non-retryable so runVariant surfaces them instead.
        err.retryable = !podId;
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
