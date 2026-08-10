import { readFileSync } from 'node:fs';
import { getLogger } from '../logger.js';

const log = getLogger('reaper');

// Every controller names its pod `bq-<pid>-<ts%100000>-<ts>`, so the pod itself
// carries who launched it and when. See run_runpod_job.py + provider.py.
const RUN_TAG = /^bq-(\d+)-(\d+)-(\d+)$/;

// Kept in step with CONTROLLER_SCRIPTS in backend/scripts/pod_watchdog.py.
const CONTROLLER_SCRIPTS = [
  'run_runpod_job.py',
  'run_catbench_job.py',
  'run_parallel_quants.py',
];

// A pod younger than this is never reaped, whatever the pid says. Provisioning
// and bootstrap take minutes, and a controller that dies and is retried inside
// that window is a job in progress, not a leak.
const MIN_AGE_MS = Number(process.env.REAP_MIN_AGE_MS || 15 * 60_000);

// Clock skew between the controller's ts and /proc's start time. Same 60s the
// python watchdog allows.
const START_SKEW_S = 60;

/** (pid, created ts in epoch seconds) for one of our pod names, else null. */
export function parseRunTag(name) {
  const m = RUN_TAG.exec(name || '');
  if (!m) return null;
  const pid = Number(m[1]);
  const ts = Number(m[3]);
  // The third field is NOT a checksum of the fourth. `run_tag` is built as
  // bq-<pid>-<time()%100000> when the controller starts, and the provider
  // appends the pod's own creation time when it rents one, seconds to minutes
  // later. Requiring them to agree rejected every real pod.
  if (!Number.isFinite(pid) || pid <= 0 || !Number.isFinite(ts) || ts < 1e9) return null;
  return { pid, ts };
}

function bootTime(readFile) {
  try {
    const line = readFile('/proc/stat').split('\n').find((l) => l.startsWith('btime '));
    return line ? Number(line.trim().split(/\s+/)[1]) : null;
  } catch {
    return null;
  }
}

/** cmdline + start time for a pid, or null when /proc says nothing. */
function procInfo(pid, readFile) {
  let cmdline;
  try {
    cmdline = readFile(`/proc/${pid}/cmdline`).replace(/\0/g, ' ');
  } catch {
    return null;
  }
  let started = null;
  const bt = bootTime(readFile);
  if (bt) {
    try {
      const stat = readFile(`/proc/${pid}/stat`);
      // Fields after the comm field's closing paren; starttime is field 22,
      // which is index 19 once state is index 0.
      const after = stat.slice(stat.lastIndexOf(')') + 2).trim().split(/\s+/);
      const ticks = Number(after[19]);
      if (Number.isFinite(ticks)) started = bt + ticks / 100; // SC_CLK_TCK is 100
    } catch {
      /* start time unknown, handled by the caller as "unsure" */
    }
  }
  return { cmdline, started };
}

/**
 * True unless the pid is provably not the controller that launched this pod.
 *
 * `process.kill(pid, 0)` only answers "does some process hold this number".
 * This box has been up 100+ days, so a recycled pid wearing a dead controller's
 * number is realistic, and the old check read that as alive forever. Two extra
 * tests close it: the cmdline must be a known controller, and the process must
 * predate the pod, because a controller cannot have launched a pod before it
 * existed.
 *
 * Every uncertainty resolves to alive. Killing a working job costs hours of GPU
 * and a user's work; leaving a leak costs the next sweep.
 */
export function controllerAlive(pid, createdTs, deps = {}) {
  const kill = deps.kill || ((p, s) => process.kill(p, s));
  const readFile = deps.readFile || ((p) => readFileSync(p, 'utf8'));

  try {
    kill(pid, 0);
  } catch (err) {
    // EPERM is a live process owned by someone else. Only ESRCH is gone.
    if (err && err.code === 'EPERM') return true;
    return false;
  }

  const proc = procInfo(pid, readFile);
  if (!proc) return true; // no /proc (or unreadable): cannot disprove it
  if (!CONTROLLER_SCRIPTS.some((s) => proc.cmdline.includes(s))) return false;
  if (proc.started != null && createdTs && proc.started > createdTs + START_SKEW_S) return false;
  return true;
}

/**
 * Terminate pods whose controller is gone. Only ever touches `bq-` pods it can
 * parse; anything else is somebody's, including the ad-hoc research pods the
 * python watchdog covers.
 */
export async function reapOrphans({ listPods, terminatePod, now = Date.now, dryRun } = {}) {
  const dry = dryRun ?? process.env.REAP_DRY_RUN === '1';
  let pods;
  try {
    pods = await listPods();
  } catch {
    return [];
  }

  const reaped = [];
  for (const p of pods) {
    const tag = parseRunTag(p.name);
    if (!tag) continue;
    if (controllerAlive(tag.pid, tag.ts)) continue;

    const ageMs = now() - tag.ts * 1000;
    if (ageMs < MIN_AGE_MS) {
      log.info(`Orphan-looking pod ${p.id} is only ${Math.round(ageMs / 1000)}s old; leaving it`);
      continue;
    }
    if (dry) {
      log.warn(`WOULD reap ${p.id} (controller pid ${tag.pid} gone, age ${Math.round(ageMs / 60000)}m)`);
      continue;
    }
    const ok = await terminatePod(p.id);
    reaped.push(p.id);
    log.warn(`Reaped orphan pod ${p.id} (dead controller pid ${tag.pid}): ${ok ? 'terminated' : 'FAILED'}`);
  }
  return reaped;
}
