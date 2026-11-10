import { describe, it, expect, vi, beforeEach } from 'vitest';

// What a bot restart does with the jobs it cut off: each one goes back to its
// controllers, running or finished, and a job with nothing left is closed.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const h = vi.hoisted(() => ({ jobs: {}, controllers: [], dropped: [] }));

vi.mock('../src/services/db.js', () => {
  const JOB_STATUS = { queued: 'queued', running: 'running', interrupted: 'interrupted',
                       failed: 'failed', completed: 'completed' };
  return {
    JOB_STATUS,
    listRecoverableJobs: async () =>
      Object.values(h.jobs).filter((j) => ['queued', 'running', 'interrupted'].includes(j.status)),
    patchJob: async (id, patch) => {
      h.jobs[id] = typeof patch === 'function' ? patch(h.jobs[id]) : { ...h.jobs[id], ...patch };
      return h.jobs[id];
    },
  };
});

vi.mock('../src/services/detached.js', () => ({
  list: () => h.controllers,
  drop: (c) => h.dropped.push(c.id),
}));

const { recoverJobs } = await import('../src/services/recover.js');

const job = (over = {}) => ({
  id: 'job-1', modelId: 'org/model', variants: ['4.0'], status: 'running',
  threadId: 'thread-1', progressMessageId: 'msg-1', ...over,
});

const controller = (over = {}) => ({
  id: 'ctrl-1', kind: 'quant', pid: 4242, logPath: '/tmp/ctrl-1.log', running: true, status: null,
  meta: { jobId: 'job-1', modelId: 'org/model', variants: ['4.0'] }, ...over,
});

/** Run a recovery and return what each resumed job was handed. */
async function recover(resume = async () => {}) {
  const calls = [];
  await recoverJobs({ client: {}, resume: (a) => (calls.push(a), resume(a)) });
  await new Promise((r) => setTimeout(r, 0));
  return calls;
}

beforeEach(() => {
  h.jobs = {};
  h.controllers = [];
  h.dropped = [];
});

describe('recoverJobs', () => {
  it('hands a running job its live controller', async () => {
    h.jobs = { 'job-1': job() };
    h.controllers = [controller()];
    const [call] = await recover();
    expect(call.resumeFrom.get('4.0').pid).toBe(4242);
    expect(call.lost).toBeNull();
  });

  it('hands over a controller that finished while the bot was down', async () => {
    h.jobs = { 'job-1': job() };
    h.controllers = [controller({ running: false, status: { code: 0, signal: null } })];
    const [call] = await recover();
    expect(call.resumeFrom.get('4.0').id).toBe('ctrl-1');
    expect(h.dropped).toEqual([]);
  });

  it('forgets a finished controller no job is waiting on', async () => {
    h.jobs = { 'job-1': job({ status: 'completed' }) };
    h.controllers = [controller({ running: false, status: { code: 0, signal: null } })];
    expect(await recover()).toEqual([]);
    expect(h.dropped).toEqual(['ctrl-1']);
  });

  it('keeps the record of a controller that is still running', async () => {
    h.controllers = [controller({ kind: 'catbench', meta: { modelId: 'org/model' } })];
    await recover();
    expect(h.dropped).toEqual([]);
  });

  it('closes a carded job whose controllers are gone', async () => {
    h.jobs = { 'job-1': job() };
    const [call] = await recover();
    expect(call.lost).toBe('Interrupted by a bot restart');
    expect(call.resumeFrom.size).toBe(0);
  });

  it('fails a job that never got a card', async () => {
    h.jobs = { 'job-1': job({ threadId: null, progressMessageId: null }) };
    expect(await recover()).toEqual([]);
    expect(h.jobs['job-1']).toMatchObject({ status: 'failed', error: 'Interrupted by a bot restart' });
  });

  it('fails a job whose resume rejects', async () => {
    h.jobs = { 'job-1': job() };
    h.controllers = [controller()];
    await recover(async () => { throw new Error('Unknown Message'); });
    expect(h.jobs['job-1']).toMatchObject({ status: 'failed', error: 'Unknown Message' });
  });

  it('leaves a job that ended on its own when the resume rejects afterwards', async () => {
    h.jobs = { 'job-1': job() };
    h.controllers = [controller()];
    await recover(async () => {
      h.jobs['job-1'].status = 'completed';
      throw new Error('Missing Access');
    });
    expect(h.jobs['job-1'].status).toBe('completed');
  });
});
