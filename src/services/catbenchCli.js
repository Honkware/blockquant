import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { getLogger } from '../logger.js';
import config from '../config.js';

const log = getLogger('catbench-cli');

const ROOT = config.ROOT_DIR;
const SCRIPT = path.join(ROOT, 'backend', 'scripts', 'run_catbench_job.py');
const PYTHON = config.PYTHON_BIN || path.join(ROOT, 'backend', 'venv', 'bin', 'python');
const LOG_DIR = path.join(ROOT, 'backend', 'logs');

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
    const slug = modelId.replace(/[^a-zA-Z0-9._-]/g, '_');
    const stamp = Date.now();
    const logPath = path.join(LOG_DIR, `catbench-${slug}-${stamp}.log`);
    const outPath = path.join(LOG_DIR, `catbench-${slug}-${stamp}.json`);

    const args = [SCRIPT, '--model', modelId, '--out', outPath];
    if (config.RUNPOD_IMAGE) args.push('--image', config.RUNPOD_IMAGE);
    if (config.CATBENCH_MAX_GB) args.push('--max-gb', String(config.CATBENCH_MAX_GB));

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
    const tailTimer = setInterval(drain, 1000);

    child.on('error', (err) => {
      clearInterval(tailTimer);
      reject(err);
    });
    child.on('exit', (code, signal) => {
      clearInterval(tailTimer);
      drain();
      const src = resultPath || outPath;
      let result = null;
      try {
        result = JSON.parse(fs.readFileSync(src, 'utf8'));
      } catch {
        /* no result file */
      }
      try {
        fs.unlinkSync(src);
      } catch {
        /* already gone */
      }
      if (result) return resolve(result);
      reject(
        new Error(
          jobError ||
            (signal
              ? `catbench controller killed by ${signal}`
              : `catbench controller exited ${code} with no result`)
        )
      );
    });
  });
}
