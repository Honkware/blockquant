import { describe, it, expect, vi, beforeEach } from 'vitest';

// access.js reaches config.js, which hard-exits on missing required env vars,
// and freezes ADMIN_IDS at load.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';
process.env.ADMIN_IDS = 'admin-1';

const h = vi.hoisted(() => ({ jobs: {}, settings: {} }));

vi.mock('../src/services/db.js', () => ({
  JOB_STATUS: {
    pending_approval: 'pending_approval',
    queued: 'queued',
    running: 'running',
    stopped: 'stopped',
    completed: 'completed',
  },
  loadJobs: async () => h.jobs,
  loadSettings: async () => h.settings,
}));

const { claimRunSlot, releaseRunSlot, activeJobsFor, isQuanter } = await import(
  '../src/services/access.js'
);

const ROLE = 'role-quanter';
const member = (roleIds) => ({ roles: { cache: new Map(roleIds.map((r) => [r, {}])) } });

const QUANTER = { userId: 'quanter-1', member: member([ROLE]) };

beforeEach(() => {
  h.jobs = {};
  h.settings = { quanterRoleId: ROLE };
  releaseRunSlot('quanter-1');
  releaseRunSlot('admin-1');
});

describe('isQuanter', () => {
  it('is false while no role is configured, whatever the member holds', async () => {
    h.settings = {};
    expect(await isQuanter(member([ROLE]))).toBe(false);
  });

  it('is false for a member the interaction carried no roles for', async () => {
    expect(await isQuanter(undefined)).toBe(false);
  });
});

describe('activeJobsFor', () => {
  it('counts queued and running, and nothing else', () => {
    const jobs = {
      a: { id: 'a', userId: 'u', status: 'queued', createdAt: 2 },
      b: { id: 'b', userId: 'u', status: 'running', createdAt: 1 },
      c: { id: 'c', userId: 'u', status: 'completed', createdAt: 3 },
      d: { id: 'd', userId: 'u', status: 'pending_approval', createdAt: 4 },
      e: { id: 'e', userId: 'u', status: 'stopped', createdAt: 5 },
      f: { id: 'f', userId: 'other', status: 'running', createdAt: 6 },
    };
    expect(activeJobsFor(jobs, 'u').map((j) => j.id)).toEqual(['b', 'a']);
  });
});

describe('claimRunSlot', () => {
  it('refuses a quanter who already has one running', async () => {
    h.jobs = { a: { id: 'a', userId: 'quanter-1', status: 'running', modelId: 'org/busy' } };

    const slot = await claimRunSlot(QUANTER);

    expect(slot).toMatchObject({ quanter: true, ok: false });
    expect(slot.running.modelId).toBe('org/busy');
  });

  it('lets a quanter through once nothing of theirs is live', async () => {
    h.jobs = {
      a: { id: 'a', userId: 'quanter-1', status: 'completed' },
      b: { id: 'b', userId: 'someone-else', status: 'running' },
    };

    expect(await claimRunSlot(QUANTER)).toMatchObject({ quanter: true, ok: true, running: null });
  });

  it('does not cap an admin', async () => {
    h.jobs = { a: { id: 'a', userId: 'admin-1', status: 'running', modelId: 'org/busy' } };

    expect(await claimRunSlot({ userId: 'admin-1', member: member([ROLE]) })).toMatchObject({
      admin: true,
      quanter: false,
      ok: true,
    });
  });

  it('does not cap a plain member, whose request waits on approval anyway', async () => {
    h.jobs = { a: { id: 'a', userId: 'plain-1', status: 'running' } };

    expect(await claimRunSlot({ userId: 'plain-1', member: member([]) })).toMatchObject({
      quanter: false,
      ok: true,
    });
  });

  it('refuses a second claim before the first job is written', async () => {
    // The job file cannot show a request that is still being validated, so two
    // /quant's typed back to back would otherwise both pass the scan.
    const [first, second] = await Promise.all([claimRunSlot(QUANTER), claimRunSlot(QUANTER)]);

    expect([first.ok, second.ok].sort()).toEqual([false, true]);

    releaseRunSlot('quanter-1');
    expect(await claimRunSlot(QUANTER)).toMatchObject({ ok: true });
  });
});
