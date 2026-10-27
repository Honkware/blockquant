import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { activeLines, historyLines, liveControllers, counts } = await import('../src/services/jobView.js');

const H = 3_600_000;
const JOBS = {
  a: { id: 'a', modelId: 'o/A', variants: ['4.0'], status: 'running', userId: '1', createdAt: 1 * H, approvedAt: 2 * H },
  b: { id: 'b', modelId: 'o/B', variants: ['3.6'], sc: true, status: 'pending_approval', userId: '2', createdAt: 3 * H },
  c: { id: 'c', modelId: 'o/C', variants: ['4.0', '5.0'], status: 'completed', userId: '1', completedAt: 5 * H,
       actualCost: { cost: 0.92 } },
  d: { id: 'd', modelId: 'o/D', variants: ['3.0'], sc: true, scLite: true, status: 'failed', userId: '3', failedAt: 6 * H },
  e: { id: 'e', modelId: 'o/E', variants: ['2.0'], status: 'rejected', userId: '3', rejectedAt: 4 * H },
};

describe('what /queue shows', () => {
  it('lists running and waiting jobs, oldest first, and nothing finished', () => {
    const lines = activeLines(JOBS);
    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain('`o/A` 4.0');
    expect(lines[1]).toContain('`o/B` 3.6 · SC');
    expect(lines[1]).toContain('asked');
  });

  it('says nothing about an empty bot', () => {
    expect(activeLines({})).toEqual([]);
  });
});

describe('what /history shows', () => {
  it('is the jobs that ended, newest first, with what they cost', () => {
    const lines = historyLines(JOBS);
    expect(lines.map((l) => l.match(/`(o\/\w)`/)[1])).toEqual(['o/D', 'o/C', 'o/E']);
    expect(lines[0]).toContain('SCT');
    expect(lines[1]).toContain('$0.92');
  });

  it('stops at the limit', () => {
    expect(historyLines(JOBS, 1)).toHaveLength(1);
  });
});

it('counts controllers only while they run', () => {
  expect(liveControllers([{ running: true }, { running: false }])).toHaveLength(1);
  expect(counts(JOBS)).toMatchObject({ running: 1, completed: 1, failed: 1 });
});
