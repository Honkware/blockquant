import { spawn } from 'node:child_process';
import path from 'node:path';
import { getLogger } from '../logger.js';
import config from '../config.js';

const log = getLogger('runpod-cli');

const ROOT = config.ROOT_DIR;
const SCRIPT = path.join(ROOT, 'backend', 'scripts', 'run_runpod_job.py');
const PUBLISH = path.join(ROOT, 'backend', 'scripts', 'publish_quant.py');
const PYTHON = config.PYTHON_BIN || path.join(ROOT, 'backend', 'venv', 'bin', 'python');
// Most MoE/dense LLMs we target are <= 48 blocks; used only to turn the live
// layer index into a fraction for the aggregate bar, so a rough guess is fine.
const TOTAL_LAYERS_GUESS = 48;

const RE = {
  pod: /Pod ID:\s*(\S+)/,
  bootstrap: /Bootstrap complete/,
  download: /\[download\]\s*(.+)/,
  quantizeStart: /\[quantize\]\s*([0-9.]+)\s*bpw/,
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
      // Cap $/hr so a stock-out can't bump us onto an idle H100/A100. The quant
      // is layer-by-layer (~4GB peak) so a cheap card is plenty; extra VRAM
      // doesn't speed it up. $1/hr keeps it on 4090 / RTX 5000 Ada / A5000 tier.
      '--max-price', '1.0',
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
    let curLayer = 0;
    let podId = '';
    let stage = 'Provisioning';

    // Phase-based progress so the bar always moves through the pipeline even
    // when a phase emits no granular events (the download phase is silent for
    // ~10-15 min). Quantize interpolates by layer across its band.
    const PHASE_PCT = { Provisioning: 4, Downloading: 20, Uploading: 94, Complete: 100 };
    const report = (message) => {
      let overall;
      if (stage === 'Quantizing') {
        overall = 25 + Math.round(Math.min(curLayer / TOTAL_LAYERS_GUESS, 1) * 65);
      } else {
        overall = PHASE_PCT[stage] ?? 0;
      }
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
      if ((m = RE.quantizeStart.exec(line))) {
        curBpw = m[1];
        curLayer = 0;
        stage = 'Quantizing';
        return report(`Quantizing ${m[1]} bpw`);
      }
      if ((m = RE.layer.exec(line))) {
        const l = parseInt(m[1], 10);
        if (Number.isFinite(l) && l >= curLayer) curLayer = l;
        stage = 'Quantizing';
        return report(`layer ${l}`);
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
        reject(new Error(`run_runpod_job.py exited ${code} with no uploads`));
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
