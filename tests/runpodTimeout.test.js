import { describe, it, expect, vi, afterEach } from 'vitest';

// The RunPod calls /quant and /stop wait on have to give up on their own.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';
process.env.RUNPOD_API_KEY ??= 'test';

const { listPods, terminatePod, getBalance } = await import('../src/services/runpod.js');

afterEach(() => vi.unstubAllGlobals());

describe('RunPod requests', () => {
  it('each carry an abort signal', async () => {
    const seen = [];
    vi.stubGlobal('fetch', async (url, opts = {}) => {
      seen.push(opts.signal);
      return { ok: false, status: 503 };
    });
    await listPods();
    await terminatePod('abc');
    await getBalance();
    expect(seen).toHaveLength(3);
    for (const s of seen) expect(s).toBeInstanceOf(AbortSignal);
  });
});
