import { readFile, writeFile, mkdir, rename } from 'fs/promises';
import path from 'path';
import config from '../config.js';
import { getLogger } from '../logger.js';

const log = getLogger('db');
const dataDir = path.join(config.ROOT_DIR, 'data');

let writeQueue = Promise.resolve();

/** Serialize writes so concurrent saves don't corrupt files. */
function enqueue(fn) {
  writeQueue = writeQueue.catch(() => {}).then(fn);
  return writeQueue;
}

async function ensureDir() {
  await mkdir(dataDir, { recursive: true });
}

async function load(filename) {
  let raw;
  try {
    raw = await readFile(path.join(dataDir, filename), 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') return {};
    log.error(`Failed to read ${filename}, treating as empty`, { error: err.message });
    return {};
  }
  // An empty or truncated file must never crash startup; treat it as empty.
  if (!raw.trim()) return {};
  try {
    return JSON.parse(raw);
  } catch (err) {
    log.error(`Corrupt ${filename}, treating as empty`, { error: err.message });
    return {};
  }
}

async function saveRaw(filename, data) {
  await ensureDir();
  // Atomic write: serialize to a temp file then rename, so a reader (or a
  // concurrent writer) can never observe a half-written / truncated file.
  const target = path.join(dataDir, filename);
  const tmp = `${target}.tmp`;
  await writeFile(tmp, JSON.stringify(data, null, 2));
  await rename(tmp, target);
}

async function save(filename, data) {
  return enqueue(async () => {
    await saveRaw(filename, data);
  });
}

// ── Public API ──────────────────────────────────────────────────────────────

const FILES = { users: 'users.json', models: 'models.json' };
const JOB_STATUS = Object.freeze({
  pending_approval: 'pending_approval',
  queued: 'queued',
  running: 'running',
  interrupted: 'interrupted',
  failed: 'failed',
  completed: 'completed',
  rejected: 'rejected',
});

export async function loadModels() {
  return load(FILES.models);
}
export async function saveModels(d) {
  return save(FILES.models, d);
}

export async function loadJobs() {
  return load('jobs.json');
}

export async function saveJobs(d) {
  return save('jobs.json', d);
}

export async function upsertJob(job) {
  return enqueue(async () => {
    const jobs = await loadJobs();
    jobs[job.id] = {
      ...jobs[job.id],
      ...job,
      updatedAt: Date.now(),
    };
    await saveRaw('jobs.json', jobs);
    return jobs[job.id];
  });
}

export async function patchJob(jobId, patch) {
  return enqueue(async () => {
    const jobs = await loadJobs();
    const current = jobs[jobId] ?? { id: jobId };
    const next = typeof patch === 'function' ? patch(current) : { ...current, ...(patch ?? {}) };
    jobs[jobId] = {
      ...next,
      id: jobId,
      updatedAt: Date.now(),
    };
    await saveRaw('jobs.json', jobs);
    return jobs[jobId];
  });
}

export async function listRecoverableJobs() {
  const jobs = await loadJobs();
  return Object.values(jobs)
    .filter((job) =>
      [JOB_STATUS.queued, JOB_STATUS.running, JOB_STATUS.interrupted].includes(job.status)
    )
    .sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
}

export { JOB_STATUS };
