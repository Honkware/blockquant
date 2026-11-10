import { describe, it, expect, vi, beforeEach } from 'vitest';

// The Retry button: it has to answer Discord before the slow quote, and a
// retry that cannot be posted must leave the job retryable.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const h = vi.hoisted(() => ({ jobs: {}, calls: [] }));

vi.mock('../src/services/db.js', () => ({
  JOB_STATUS: { pending_approval: 'pending_approval', failed: 'failed' },
  loadJobs: async () => h.jobs,
  upsertJob: async (j) => (h.jobs[j.id] = j),
  patchJob: async (id, patch) => (h.jobs[id] = { ...h.jobs[id], ...patch }),
}));
vi.mock('../src/services/gpuQuote.js', () => ({
  quoteCost: async () => (h.calls.push('quote'), null),
}));
vi.mock('../src/services/runpod.js', () => ({
  getBalance: async () => null, convertHours: () => ({}), calibrationHours: () => ({}),
}));
vi.mock('../src/services/approvalNotify.js', () => ({ notifyAdmins: async () => {}, cardUrl: () => '' }));

const { handleRetry } = await import('../src/commands/quant.js');

const failed = { id: 'job-1', userId: 'u1', modelId: 'org/model', variants: ['4.0'], channelId: 'c1',
                 status: 'failed', partialResults: [{ bpw: '4.0', variant: '4.0', pushed: false, error: 'x' }] };

function button(channel) {
  return {
    customId: 'bq:retry:job-1',
    user: { id: 'u1' },
    client: { channels: { fetch: async () => channel } },
    deferUpdate: async () => h.calls.push('defer'),
    editReply: async () => h.calls.push('edit'),
    followUp: async (m) => h.calls.push(`followUp:${m.content}`),
    reply: async (m) => h.calls.push(`reply:${m.content}`),
  };
}

beforeEach(() => {
  h.jobs = { 'job-1': { ...failed } };
  h.calls = [];
});

describe('the Retry button', () => {
  it('answers Discord before the quote', async () => {
    await handleRetry(button({ send: async () => ({ id: 'm1' }) }));
    expect(h.calls.indexOf('defer')).toBeLessThan(h.calls.indexOf('quote'));
    expect(h.calls.at(-1)).toBe('edit');
  });

  it('leaves the job retryable when its channel is gone', async () => {
    await handleRetry(button(null));
    expect(h.jobs['job-1'].retriedAs).toBeUndefined();
    expect(Object.keys(h.jobs)).toEqual(['job-1']);
    expect(h.calls.at(-1)).toBe('followUp:The channel this job came from is gone.');
  });
});
