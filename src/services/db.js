import { readFile, writeFile, mkdir } from 'fs/promises';
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
  try {
    const raw = await readFile(path.join(dataDir, filename), 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    if (err.code === 'ENOENT') return {};
    log.error(`Failed to load ${filename}`, { error: err.message });
    throw err;
  }
}

async function saveRaw(filename, data) {
  await ensureDir();
  await writeFile(path.join(dataDir, filename), JSON.stringify(data, null, 2));
}

async function save(filename, data) {
  return enqueue(async () => {
    await saveRaw(filename, data);
  });
}

// ── Public API ──────────────────────────────────────────────────────────────

const FILES = { users: 'users.json', models: 'models.json' };
const JOB_STATUS = Object.freeze({
  queued: 'queued',
  running: 'running',
  interrupted: 'interrupted',
  failed: 'failed',
  completed: 'completed',
});

export async function loadUsers() {
  return load(FILES.users);
}
export async function loadModels() {
  return load(FILES.models);
}
export async function saveUsers(d) {
  return save(FILES.users, d);
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

export async function chargeForJob({ jobId, userId, cost }) {
  return enqueue(async () => {
    const jobs = await loadJobs();
    const users = await loadUsers();
    const job = jobs[jobId];
    if (!job) throw new Error(`Unknown job: ${jobId}`);

    const user = users[userId] ?? { exp: 0, lastQuant: 0 };
    if (job.chargedAt) {
      return { charged: false, balance: user.exp };
    }
    if (user.exp < cost) {
      return { charged: false, balance: user.exp, insufficient: true };
    }

    user.exp -= cost;
    user.lastQuant = Date.now();
    users[userId] = user;

    jobs[jobId] = {
      ...job,
      chargedAt: Date.now(),
      cost,
      updatedAt: Date.now(),
    };

    await saveRaw(FILES.users, users);
    await saveRaw('jobs.json', jobs);
    return { charged: true, balance: user.exp };
  });
}

export async function refundForJob({ jobId, userId, cost }) {
  return enqueue(async () => {
    const jobs = await loadJobs();
    const users = await loadUsers();
    const job = jobs[jobId];
    if (!job) throw new Error(`Unknown job: ${jobId}`);
    if (!job.chargedAt || job.refundedAt) {
      return { refunded: false, balance: users[userId]?.exp ?? 0 };
    }

    const user = users[userId] ?? { exp: 0, lastQuant: 0 };
    user.exp += cost;
    users[userId] = user;

    jobs[jobId] = {
      ...job,
      refundedAt: Date.now(),
      updatedAt: Date.now(),
    };

    await saveRaw(FILES.users, users);
    await saveRaw('jobs.json', jobs);
    return { refunded: true, balance: user.exp };
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
