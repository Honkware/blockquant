import { describe, it, expect, vi } from 'vitest';

// The result card waits on the collection finalizer, so it must not run forever.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const h = vi.hoisted(() => ({ opts: null }));
vi.mock('node:child_process', async () => {
  const { EventEmitter } = await import('node:events');
  return {
    spawn: (cmd, args, opts) => {
      h.opts = opts;
      const child = new EventEmitter();
      child.stdout = new EventEmitter();
      child.stderr = new EventEmitter();
      setTimeout(() => child.emit('close', null, 'SIGTERM'), 0);
      return child;
    },
  };
});

const { finalizeCollection } = await import('../src/services/runpodCli.js');

describe('finalizeCollection', () => {
  it('runs the publisher with a time limit and reports a kill as a failure', async () => {
    const out = await finalizeCollection({ modelId: 'org/model', hfOrg: 'o' });
    expect(h.opts.timeout).toBeGreaterThan(0);
    expect(out).toEqual({ ok: false, collectionUrl: '' });
  });
});
