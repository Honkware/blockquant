import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// stop.js reaches config.js, which hard-exits on missing required env vars.
// ADMIN_IDS has to be set before the import too: config freezes it at load.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';
process.env.ADMIN_IDS = 'admin-1';

// Mock the two stores and the RunPod client stop.js acts through. The log file
// is real: reading the pod id out of a controller log is the part that decides
// whether a pod gets terminated or bills for 8 hours.
const h = vi.hoisted(() => ({ jobs: {}, settings: {}, controllers: [], calls: [] }));

vi.mock('../src/services/db.js', () => ({
  JOB_STATUS: {
    pending_approval: 'pending_approval',
    queued: 'queued',
    running: 'running',
    stopped: 'stopped',
  },
  loadJobs: async () => h.jobs,
  loadSettings: async () => h.settings,
  patchJob: async (id, patch) => {
    h.jobs[id] = { ...h.jobs[id], ...patch };
    return h.jobs[id];
  },
}));

vi.mock('../src/services/runpod.js', () => ({
  terminatePod: async (id) => {
    h.calls.push(`terminate:${id}`);
    return id !== 'pod-refused';
  },
}));

vi.mock('../src/services/detached.js', () => ({ list: () => h.controllers }));

const { canStop, stopJob, controllersForJob } = await import('../src/commands/stop.js');

const ROLE = 'role-quanter';
const STARTED = 1_800_000_000_000;

/** A GuildMember-shaped member; discord.js also hands over a raw roles array. */
const member = (roleIds) => ({ roles: { cache: new Map(roleIds.map((r) => [r, {}])) } });

const job = (over = {}) => ({
  id: 'job-1',
  userId: 'quanter-1',
  modelId: 'org/model',
  variants: ['4.0'],
  status: 'running',
  startedAt: STARTED,
  ...over,
});

function controllerLog(text) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bq-stop-'));
  const logPath = path.join(dir, 'ctrl.log');
  fs.writeFileSync(logPath, text);
  return logPath;
}

beforeEach(() => {
  h.jobs = {};
  h.settings = { quanterRoleId: ROLE };
  h.controllers = [];
  h.calls = [];
});

describe('canStop', () => {
  it('lets an admin stop someone else’s job', async () => {
    expect(await canStop({ userId: 'admin-1', member: member([]), job: job() })).toBe(true);
  });

  it('lets a quanter stop their own job', async () => {
    expect(
      await canStop({ userId: 'quanter-1', member: member([ROLE]), job: job() })
    ).toBe(true);
  });

  it('refuses a quanter stopping somebody else’s job', async () => {
    expect(
      await canStop({ userId: 'quanter-2', member: member([ROLE]), job: job() })
    ).toBe(false);
  });

  it('refuses a plain member, even on their own job', async () => {
    expect(
      await canStop({ userId: 'quanter-1', member: member(['role-other']), job: job() })
    ).toBe(false);
  });

  it('refuses everyone but admins while no quanter role is set', async () => {
    h.settings = {};
    expect(await canStop({ userId: 'quanter-1', member: member([ROLE]), job: job() })).toBe(false);
    expect(await canStop({ userId: 'admin-1', member: member([ROLE]), job: job() })).toBe(true);
  });

  it('reads roles off the raw interaction member too', async () => {
    expect(await canStop({ userId: 'quanter-1', member: { roles: [ROLE] }, job: job() })).toBe(true);
  });
});

describe('controllersForJob', () => {
  const rec = (over = {}) => ({
    running: true,
    pid: 4242,
    startedAt: STARTED + 1000,
    meta: { modelId: 'org/model' },
    logPath: '/dev/null',
    ...over,
  });

  it('skips another model and anything that started before the job did', () => {
    h.controllers = [
      rec(),
      rec({ pid: 5, meta: { modelId: 'org/other' } }),
      rec({ pid: 6, startedAt: STARTED - 10 * 60_000 }),
      rec({ pid: 7, running: false }),
    ];
    expect(controllersForJob(job()).map((c) => c.pid)).toEqual([4242]);
  });
});

describe('stopJob', () => {
  it('terminates the pod before killing the controller', async () => {
    const logPath = controllerLog('booting\nPod ID: pod-abc\n[download] 12%\n');
    h.controllers = [
      { running: true, pid: 4242, startedAt: STARTED + 1000, meta: { modelId: 'org/model' }, logPath },
    ];
    const kill = (pid, sig) => h.calls.push(`kill:${pid}:${sig}`);

    const res = await stopJob(job(), { by: 'admin-1', kill, sleep: async () => {} });

    // Order is the point: a controller killed first leaves its pod renting a
    // GPU until the reaper's 15-minute floor, or RunPod's 8h max_runtime.
    expect(h.calls[0]).toBe('terminate:pod-abc');
    expect(h.calls[1]).toBe('kill:4242:SIGTERM');
    expect(res.pods).toEqual(['pod-abc']);
    expect(h.jobs['job-1'].status).toBe('stopped');
    expect(h.jobs['job-1'].stoppedBy).toBe('admin-1');
  });

  it('escalates to SIGKILL when the controller is still up after the grace', async () => {
    const logPath = controllerLog('Pod ID: pod-abc\n');
    h.controllers = [
      { running: true, pid: 4242, startedAt: STARTED, meta: { modelId: 'org/model' }, logPath },
    ];
    const kill = (pid, sig) => h.calls.push(`kill:${pid}:${sig}`);

    await stopJob(job(), { by: 'admin-1', kill, sleep: async () => {} });

    expect(h.calls).toContain('kill:4242:SIGKILL');
  });

  it('leaves the controller alone once it has taken the SIGTERM', async () => {
    const logPath = controllerLog('Pod ID: pod-abc\n');
    h.controllers = [
      { running: true, pid: 4242, startedAt: STARTED, meta: { modelId: 'org/model' }, logPath },
    ];
    const kill = (pid, sig) => {
      h.calls.push(`kill:${pid}:${sig}`);
      if (sig === 0) throw Object.assign(new Error('ESRCH'), { code: 'ESRCH' });
    };

    await stopJob(job(), { by: 'admin-1', kill, sleep: async () => {} });

    expect(h.calls).not.toContain('kill:4242:SIGKILL');
  });

  it('still terminates pods recorded on the job when no controller matches', async () => {
    const res = await stopJob(job({ podIds: ['pod-xyz'] }), {
      by: 'admin-1',
      kill: () => {},
      sleep: async () => {},
    });

    expect(h.calls).toEqual(['terminate:pod-xyz']);
    expect(res.killed).toEqual([]);
    expect(h.jobs['job-1'].status).toBe('stopped');
  });

  it('reports a pod RunPod would not terminate instead of swallowing it', async () => {
    const res = await stopJob(job({ podIds: ['pod-refused'] }), {
      by: 'admin-1',
      kill: () => {},
      sleep: async () => {},
    });

    expect(res.pods).toEqual([]);
    expect(res.failedPods).toEqual(['pod-refused']);
  });

  it('terminates every pod a retried controller rented', async () => {
    const logPath = controllerLog('Pod ID: pod-1\nlaunch failed, retrying\nPod ID: pod-2\n');
    h.controllers = [
      { running: true, pid: 4242, startedAt: STARTED, meta: { modelId: 'org/model' }, logPath },
    ];

    const res = await stopJob(job(), { by: 'admin-1', kill: () => {}, sleep: async () => {} });

    expect(res.pods).toEqual(['pod-1', 'pod-2']);
  });
});
