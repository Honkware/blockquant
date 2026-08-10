import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { getLogger } from '../logger.js';
import config from '../config.js';
import { AppError, sanitizeErrorText } from '../errors/taxonomy.js';
import { spawnDetached, wait } from './detached.js';

const log = getLogger('catbench-cli');

const ROOT = config.ROOT_DIR;
const SCRIPT = path.join(ROOT, 'backend', 'scripts', 'run_catbench_job.py');
const STORE_SCRIPT = path.join(ROOT, 'backend', 'scripts', 'catbench_store.py');
const PYTHON = config.PYTHON_BIN || path.join(ROOT, 'backend', 'venv', 'bin', 'python');
const LOG_DIR = path.join(ROOT, 'backend', 'logs');
const MIRROR_DIR = path.join(ROOT, 'data', 'catbench');

// A result file is the only copy of what a pod produced, and a pod costs money.
// It used to be unlinked the instant it was parsed, which left a run whose
// delivery failed, or one that landed while the bot was down, with nothing to
// recover from -- and index.js already tells the operator to go look at
// meta.resultPath in exactly that case. So they stay, and a new run prunes the
// old ones rather than the reader deleting its own evidence.
const KEEP_RESULTS = 20;

const RE = {
  pod: /Pod ID:\s*(\S+)/,
  gpu: /Trying\s+(.+?)\s+\(~\$([0-9.]+)/,
  download: /\[download\]\s*(.+)/,
  progress: /\[progress\]\s*(.+)/,
  result: /\[catbench\]\s*result\s+(\S+)/,
  jobError: /\[joberror\]\s*(.+)/,
};

/**
 * Size gate, run before anything is provisioned. The controller's --preflight
 * does the HF lookup (the same one /quant's GPU sizing uses) and prints one
 * JSON line. Resolves { ok, gb, error }; a crashed preflight is a rejection,
 * because a size we can't read is a size we can't promise fits one H100.
 */
export function preflight(modelId) {
  return new Promise((resolve) => {
    const child = spawn(PYTHON, [SCRIPT, '--model', modelId, '--preflight'], {
      cwd: ROOT,
      env: { ...process.env, HF_TOKEN: config.HF_TOKEN, PYTHONUNBUFFERED: '1' },
    });
    let out = '';
    let errOut = '';
    const timer = setTimeout(() => child.kill('SIGKILL'), 45_000);
    child.stdout.on('data', (c) => (out += c.toString()));
    child.stderr.on('data', (c) => (errOut += c.toString()));
    child.on('error', (err) => {
      clearTimeout(timer);
      resolve({ ok: false, error: `size check could not run: ${err.message}` });
    });
    child.on('close', () => {
      clearTimeout(timer);
      const line = out.trim().split('\n').filter(Boolean).pop();
      try {
        resolve(JSON.parse(line));
      } catch {
        log.error(`preflight unparseable: ${(errOut || out).slice(-300)}`);
        resolve({ ok: false, error: 'the model size check failed; try again shortly' });
      }
    });
  });
}

/** Drop all but the newest KEEP_RESULTS result files. Name carries the stamp. */
function pruneResults() {
  let files;
  try {
    files = fs
      .readdirSync(LOG_DIR)
      .map((n) => ({ n, m: /^catbench-.+-(\d+)\.json$/.exec(n) }))
      .filter((f) => f.m)
      .sort((a, b) => Number(b.m[1]) - Number(a.m[1]));
  } catch {
    return;
  }
  for (const f of files.slice(KEEP_RESULTS)) {
    try {
      fs.unlinkSync(path.join(LOG_DIR, f.n));
    } catch {
      /* already gone */
    }
  }
}

/**
 * Run CatBench for one model on a throwaway pod. Mirrors runpodCli.runViaCli:
 * detached controller writing to a log file, tailed here for progress, so a bot
 * restart can't orphan the pod (the controller still terminates it, and the
 * bot's own `bq-<pid>` reaper backstops a dead controller).
 *
 * Resolves the controller's result JSON: { svg, python_source, python_png_b64,
 * python_error, svg_error, cost_usd, wall_seconds }.
 */
export function runCatbench({ modelId, onProgress }) {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    pruneResults();
    const slug = modelId.replace(/[^a-zA-Z0-9._-]/g, '_');
    const stamp = Date.now();
    const logPath = path.join(LOG_DIR, `catbench-${slug}-${stamp}.log`);
    const outPath = path.join(LOG_DIR, `catbench-${slug}-${stamp}.json`);

    const args = [SCRIPT, '--model', modelId, '--out', outPath];
    if (config.RUNPOD_IMAGE) args.push('--image', config.RUNPOD_IMAGE);
    if (config.CATBENCH_MAX_GB) args.push('--max-gb', String(config.CATBENCH_MAX_GB));

    log.info(`spawn (reparented): ${PYTHON} ${args.join(' ')} -> ${logPath}`);
    let handle;
    try {
      handle = spawnDetached({
        command: PYTHON,
        args,
        cwd: ROOT,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        logPath,
        kind: 'catbench',
        meta: { modelId, resultPath: outPath },
      });
    } catch (err) {
      return reject(err);
    }

    let stage = 'Provisioning';
    let message = 'looking for a GPU';
    let podId = '';
    let jobError = '';
    let resultPath = '';

    const report = () => onProgress?.({ stage, message, podId });

    function handleLine(line) {
      let m;
      if ((m = RE.jobError.exec(line))) {
        jobError = m[1].trim().slice(0, 300);
        return;
      }
      if ((m = RE.result.exec(line))) {
        resultPath = m[1];
        return;
      }
      if ((m = RE.download.exec(line))) {
        stage = 'Downloading';
        message = m[1].slice(0, 80);
        return report();
      }
      if ((m = RE.progress.exec(line))) {
        stage = /prompt|render/i.test(m[1]) ? 'Generating' : 'Loading';
        message = m[1].slice(0, 80);
        return report();
      }
      if ((m = RE.pod.exec(line))) {
        podId = m[1];
        message = `pod ${podId}`;
        return report();
      }
      if ((m = RE.gpu.exec(line))) {
        message = `${m[1]} ~$${m[2]}/hr`;
        return report();
      }
    }

    let buf = '';
    let readOffset = 0;
    const drain = () => {
      let stat;
      try {
        stat = fs.statSync(logPath);
      } catch {
        return;
      }
      if (stat.size <= readOffset) return;
      const fd = fs.openSync(logPath, 'r');
      const b = Buffer.alloc(stat.size - readOffset);
      try {
        fs.readSync(fd, b, 0, b.length, readOffset);
      } finally {
        fs.closeSync(fd);
      }
      readOffset = stat.size;
      buf += b.toString('utf8');
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        try {
          handleLine(line);
        } catch (e) {
          log.debug(`parse: ${e.message}`);
        }
      }
    };
    // No exit event: the controller is reparented to init, so poll its recorded
    // status and drain the log on the way (see services/detached.js).
    wait(handle, { onTick: drain }).then(({ code, signal }) => {
      const src = resultPath || outPath;
      let result = null;
      try {
        result = JSON.parse(fs.readFileSync(src, 'utf8'));
      } catch {
        /* no result file */
      }
      // One line per finished run, always. Without it a controller that ended
      // is indistinguishable in the log from one still going, which is how a
      // completed bench read as a run that vanished.
      log.info(
        `finished (${signal ? `killed by ${signal}` : `exit ${code}`}): ` +
          (result ? `result ${src}` : 'no result file')
      );
      if (result) return resolve(result);
      const why =
        jobError ||
        (signal
          ? `catbench controller killed by ${signal}`
          : `catbench controller exited ${code} with no result`);
      // publicMessage, not a bare Error: toUserMessage redacts anything it does
      // not recognise down to "Unexpected internal error", which is how a real
      // diagnosis ("Unknown quantization type, got exl3") never reached anyone.
      // The controller writes this line, and sanitizeErrorText scrubs tokens.
      reject(new AppError('QUANT_EXIT_FAILED', why, { publicMessage: sanitizeErrorText(why) }));
    });
  });
}

/**
 * Persist one graded run through backend/scripts/catbench_store.py: the local
 * mirror always, the HF dataset when CATBENCH_DATASET names one.
 *
 * The upload runs here, on the VPS, not on the pod. The pod already hands the
 * SVG, the script and the rendered PNG back in its result JSON, and it drops
 * HF_TOKEN from its environment before any model output executes. Uploading
 * from the pod would put a write token back next to model-written code for no
 * gain, so it stays here.
 *
 * Resolves the store's JSON receipt. Rejects if it refuses the payload.
 */
export function storeRun(payload) {
  return new Promise((resolve, reject) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bq-catbench-'));
    const write = (name, buf) => {
      const p = path.join(dir, name);
      fs.writeFileSync(p, buf);
      return p;
    };
    const body = {
      model_id: payload.model_id,
      display_name: payload.display_name || String(payload.model_id).split('/').pop(),
      loader: payload.loader || '',
      engine: payload.engine || '',
      format: payload.format || '',
      run_date: payload.run_date || new Date().toISOString(),
      prompts: payload.prompts || {},
      svg_source: payload.svgSource || '',
      python_source: payload.pythonSource || '',
      svg_png: payload.svgPng ? write('svg.png', payload.svgPng) : '',
      python_png: payload.pythonPng ? write('python.png', payload.pythonPng) : '',
    };
    const payloadPath = write('payload.json', Buffer.from(JSON.stringify(body)));

    const args = [STORE_SCRIPT, '--payload', payloadPath, '--mirror', MIRROR_DIR];
    if (config.CATBENCH_DATASET) args.push('--repo', config.CATBENCH_DATASET);

    const child = spawn(PYTHON, args, {
      cwd: ROOT,
      env: { ...process.env, HF_TOKEN: config.HF_TOKEN, PYTHONUNBUFFERED: '1' },
    });
    let out = '';
    let errOut = '';
    // Big enough for a slow LFS push of two jpgs, short enough that a hung
    // upload cannot pin the command's finally block open.
    const timer = setTimeout(() => child.kill('SIGKILL'), 180_000);
    child.stdout.on('data', (c) => (out += c.toString()));
    child.stderr.on('data', (c) => (errOut += c.toString()));
    const done = (err, val) => {
      clearTimeout(timer);
      fs.rmSync(dir, { recursive: true, force: true });
      err ? reject(err) : resolve(val);
    };
    child.on('error', (err) => done(err));
    child.on('close', (code) => {
      const line = out.trim().split('\n').filter(Boolean).pop() || '';
      let receipt = null;
      try {
        receipt = JSON.parse(line);
      } catch {
        /* not JSON */
      }
      if (receipt?.ok) return done(null, receipt);
      const why = receipt?.error || (errOut || out).trim().slice(-300) || `store exited ${code}`;
      done(new Error(sanitizeErrorText(why)));
    });
  });
}
