// Long-running controllers that a bot restart cannot touch.
//
// `spawn(..., { detached: true })` is not enough. detached gives the child its
// own session and process group; it does NOT change its parent, so the child is
// still a child. pm2 defaults to treekill:true, which walks descendants by PPID
// and signals each one, so `pm2 restart blockquant` killed every controller the
// bot had started. That is how a /catbench run died mid-bootstrap with a
// KeyboardInterrupt inside recv_exit_status, and it was aimed at /quant next.
//
// So the controller is reparented before we return. `sh` starts it in the
// background and exits immediately; the controller reparents to init and is
// nobody's descendant by the time any tree walk runs. spawnSync on that `sh` is
// deliberate: it means reparenting has already happened when spawn() returns,
// instead of a few milliseconds later. It costs one fork+exec on the event loop.
//
// The price is the Node `exit` event: spawn returns when the wrapper exits, not
// when the work finishes. So the wrapper records the run in files instead --
// a pid to poll and an exit status to read -- and wait() polls those. Those
// files outlive the bot too, which is what lets a fresh bot see that a run it
// never started is still going (list/sweep, `npm run status`).
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

// No config.js import on purpose: this module has to be loadable by
// scripts/status.mjs and by the tree-kill test without a bot environment.
export const RUN_DIR = process.env.BQ_RUN_DIR || path.join(ROOT, 'backend', 'logs', 'controllers');

// Run the command in the background, record its exit status, then exit. $! is
// the backgrounded subshell, which lives exactly as long as the command does.
// Written through a temp name so a reader never sees a half-written status.
const WRAPPER =
  '{ "$@"; printf %s "$?" > "$BQ_STATUS.part" && mv "$BQ_STATUS.part" "$BQ_STATUS"; } &\n' +
  'printf %s "$!" > "$BQ_PID"';

const SIGNALS = Object.entries(os.constants.signals);

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM'; // exists, owned by someone else
  }
}

/**
 * Decode the wrapper's `$?`. A shell reports a signalled child as 128+signum,
 * which is the only signal channel `$?` has; a controller's own exit codes are
 * small (0, 1, 2), so the overlap is theoretical.
 */
export function decodeStatus(raw) {
  const n = Number.parseInt(String(raw).trim(), 10);
  if (!Number.isInteger(n)) return null;
  const sig = SIGNALS.find(([, num]) => num === n - 128);
  return sig ? { code: null, signal: sig[0] } : { code: n, signal: null };
}

function readStatus(statusPath) {
  try {
    return decodeStatus(fs.readFileSync(statusPath, 'utf8'));
  } catch {
    return null;
  }
}

function paths(id) {
  return {
    pidPath: path.join(RUN_DIR, `${id}.pid`),
    statusPath: path.join(RUN_DIR, `${id}.status`),
    recordPath: path.join(RUN_DIR, `${id}.json`),
  };
}

function drop(handle) {
  for (const p of [handle.pidPath, handle.statusPath, handle.recordPath]) {
    try {
      fs.unlinkSync(p);
    } catch {
      /* already gone */
    }
  }
}

/**
 * Start `command` reparented to init, logging to `logPath`. The run is named
 * after the log file, so a record in RUN_DIR always points at readable output.
 *
 * Throws if the wrapper could not start; there is no 'error' event to wait for.
 * Returns a handle for wait().
 */
export function spawnDetached({ command, args, cwd, env, logPath, kind = 'job', meta = {} }) {
  if (command.includes('/') && !fs.existsSync(command)) {
    throw new Error(`controller interpreter not found: ${command}`);
  }
  fs.mkdirSync(RUN_DIR, { recursive: true });
  fs.mkdirSync(path.dirname(logPath), { recursive: true });

  const id = path.basename(logPath).replace(/\.log$/, '');
  const { pidPath, statusPath, recordPath } = paths(id);
  drop({ pidPath, statusPath, recordPath }); // a reused id would read as finished

  const logFd = fs.openSync(logPath, 'a');
  let res;
  try {
    res = spawnSync('/bin/sh', ['-c', WRAPPER, 'bq', command, ...args], {
      cwd,
      env: { ...env, BQ_PID: pidPath, BQ_STATUS: statusPath },
      detached: true,
      stdio: ['ignore', logFd, logFd],
    });
  } finally {
    fs.closeSync(logFd);
  }
  if (res.error) throw res.error;
  if (res.status !== 0) throw new Error(`controller wrapper exited ${res.status}`);

  const pid = Number.parseInt(fs.readFileSync(pidPath, 'utf8'), 10);
  if (!Number.isInteger(pid)) throw new Error('controller wrapper wrote no pid');

  const handle = { id, kind, pid, logPath, pidPath, statusPath, recordPath, startedAt: Date.now() };
  fs.writeFileSync(recordPath, JSON.stringify({ ...handle, meta, argv: [command, ...args] }, null, 1));
  return handle;
}

/**
 * Resolve { code, signal } when the controller finishes, polling every second
 * and calling `onTick` each time (the callers tail their log file there).
 *
 * A dead pid with no status file means something SIGKILLed the wrapper out from
 * under the job. Reporting that as a signal keeps it terminal: a killed run must
 * never look retryable, or cancelling one spawns another pod.
 */
export function wait(handle, { onTick, intervalMs = 1000 } = {}) {
  return new Promise((resolve) => {
    let misses = 0;
    const tick = () => {
      try {
        onTick?.();
      } catch {
        /* a parse blowup must not strand the run */
      }
      let status = readStatus(handle.statusPath);
      if (!status) {
        if (alive(handle.pid)) return;
        // Gone but nothing written yet: give the mv one more tick to land.
        if (++misses < 2) return;
        status = readStatus(handle.statusPath) || { code: null, signal: 'SIGKILL' };
      }
      clearInterval(timer);
      try {
        onTick?.();
      } catch {
        /* ditto */
      }
      drop(handle);
      resolve(status);
    };
    const timer = setInterval(tick, intervalMs);
  });
}

/** Every controller this box has a record of, running or not. */
export function list() {
  let names;
  try {
    names = fs.readdirSync(RUN_DIR).filter((n) => n.endsWith('.json'));
  } catch {
    return [];
  }
  const out = [];
  for (const name of names) {
    let rec;
    try {
      rec = JSON.parse(fs.readFileSync(path.join(RUN_DIR, name), 'utf8'));
    } catch {
      continue;
    }
    const status = readStatus(rec.statusPath);
    // pid reuse could in principle show a finished run as running. It would
    // cost a spurious warning, never a killed job, so it is not worth /proc.
    out.push({ ...rec, status, running: !status && alive(rec.pid) });
  }
  return out.sort((a, b) => a.startedAt - b.startedAt);
}

/**
 * Boot-time tidy: forget the runs that ended, keep the ones still going.
 * Returns both, because a run that finished while the bot was down wrote a
 * result nobody has looked at.
 */
export function sweep() {
  const live = [];
  const finished = [];
  for (const rec of list()) {
    if (rec.running) live.push(rec);
    else {
      finished.push(rec);
      drop(rec);
    }
  }
  return { live, finished };
}
