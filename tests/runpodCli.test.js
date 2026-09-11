import fs from 'node:fs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// runpodCli pulls in config.js, which hard-exits on missing required env vars.
// Provide throwaway values so the module loads under the test runner.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

// Mock the launcher, not child_process: the controller is reparented to init
// and reports through files, so there is no child object to emit events on.
// Real survival is proved in backend/tests/test_detached.py; this drives the
// finish handler. `finish` resolves wait() the way a recorded exit status does,
// after a line has been written to the log the real code tails.
const h = vi.hoisted(() => ({ logPath: '', finish: null }));
vi.mock('../src/services/detached.js', () => ({
  spawnDetached: (opts) => {
    h.logPath = opts.logPath;
    return { id: 'test', pid: process.pid, logPath: opts.logPath };
  },
  wait: (handle, { onTick } = {}) =>
    new Promise((resolve) => {
      h.finish = (code, signal) => {
        onTick?.();
        resolve({ code, signal });
      };
    }),
}));

const { runViaCli, isControllerRetryable, runVariantWithRetry } = await import(
  '../src/services/runpodCli.js'
);

describe('isControllerRetryable', () => {
  it('treats a signal kill as terminal even before a pod exists', () => {
    expect(isControllerRetryable({ signal: 'SIGKILL', podId: '' })).toBe(false);
  });

  it('treats a pod-created failure as terminal', () => {
    expect(isControllerRetryable({ signal: null, podId: 'abc123' })).toBe(false);
  });

  it('retries only a clean exit that never created a pod', () => {
    expect(isControllerRetryable({ signal: null, podId: '' })).toBe(true);
  });
});

describe('runViaCli finish handler', () => {
  beforeEach(() => {
    h.logPath = '';
    h.finish = null;
  });

  it('rejects non-retryable when SIGKILLed after a pod was created', async () => {
    const p = runViaCli({ modelId: 'org/model', variants: ['5.0'] });
    // Controller announced its pod, then an operator SIGKILLed it to cancel a
    // broken model (code null, signal set).
    fs.writeFileSync(h.logPath, 'Pod ID: abc123\n');
    h.finish(null, 'SIGKILL');

    await expect(p).rejects.toMatchObject({
      retryable: false,
      signal: 'SIGKILL',
      podCreated: true,
    });
  });

  it('rejects non-retryable when SIGKILLed during provisioning (no pod id seen yet)', async () => {
    const p = runViaCli({ modelId: 'org/model', variants: ['5.0'] });
    // This is the runaway repro: killed before the "Pod ID:" line. The old
    // guard (retryable = !podId) made this retryable and respawned a pod.
    h.finish(null, 'SIGKILL');

    await expect(p).rejects.toMatchObject({ retryable: false, signal: 'SIGKILL' });
  });

  it('keeps a clean launch failure (no pod, no signal) retryable', async () => {
    const p = runViaCli({ modelId: 'org/model', variants: ['5.0'] });
    h.finish(1, null); // a genuine stock/launch failure before any pod

    await expect(p).rejects.toMatchObject({ retryable: true });
  });
});

describe('runVariantWithRetry', () => {
  const noSleep = () => Promise.resolve();

  it('does not respawn a signal-killed (non-retryable) failure', async () => {
    const err = Object.assign(new Error('cli killed (SIGKILL)'), { retryable: false });
    const run = vi.fn().mockRejectedValue(err);
    const onRetry = vi.fn();

    await expect(
      runVariantWithRetry('5.0', { run, maxAttempts: 3, onRetry, sleep: noSleep })
    ).rejects.toBe(err);

    expect(run).toHaveBeenCalledTimes(1); // no second controller spawned
    expect(onRetry).not.toHaveBeenCalled();
  });

  it('retries a transient (retryable) launch failure up to maxAttempts', async () => {
    const err = Object.assign(new Error('stock blip'), { retryable: true });
    const run = vi.fn().mockRejectedValue(err);
    const onRetry = vi.fn();

    await expect(
      runVariantWithRetry('5.0', { run, maxAttempts: 3, onRetry, sleep: noSleep })
    ).rejects.toBe(err);

    expect(run).toHaveBeenCalledTimes(3);
    expect(onRetry).toHaveBeenCalledTimes(2);
  });

  it('returns the result on first success without retrying', async () => {
    const run = vi.fn().mockResolvedValue([{ bpw: '5.0', url: 'https://hf/x' }]);
    const out = await runVariantWithRetry('5.0', { run, sleep: noSleep });
    expect(out).toEqual([{ bpw: '5.0', url: 'https://hf/x' }]);
    expect(run).toHaveBeenCalledTimes(1);
  });
});
