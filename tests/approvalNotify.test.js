import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'g';
process.env.HF_TOKEN ??= 'test';

const { stalePending, cardUrl } = await import('../src/services/approvalNotify.js');

const H = 3_600_000;
const NOW = 100 * H;
const job = (id, over) => ({ id, status: 'pending_approval', createdAt: NOW - 8 * H, ...over });

describe('who gets a reminder', () => {
  it('reminds about a request that has waited past the threshold', () => {
    expect(stalePending({ a: job('a') }, NOW).map((j) => j.id)).toEqual(['a']);
  });

  it('leaves a fresh request alone', () => {
    expect(stalePending({ a: job('a', { createdAt: NOW - H }) }, NOW)).toEqual([]);
  });

  it('does not remind about the same request every hour', () => {
    expect(stalePending({ a: job('a', { remindedAt: NOW - 2 * H }) }, NOW)).toEqual([]);
    expect(stalePending({ a: job('a', { remindedAt: NOW - 13 * H }) }, NOW)).toHaveLength(1);
  });

  it('ignores anything already decided', () => {
    for (const status of ['running', 'completed', 'rejected', 'failed']) {
      expect(stalePending({ a: job('a', { status }) }, NOW)).toEqual([]);
    }
  });
});

describe('card links', () => {
  it('links the card when both ids are known', () => {
    expect(cardUrl({ channelId: 'c', cardMessageId: 'm' })).toMatch(/\/channels\/[^/]+\/c\/m$/);
    expect(cardUrl({ channelId: 'c' })).toBe(null);
  });
});
