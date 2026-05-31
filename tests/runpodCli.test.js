import { EventEmitter } from 'node:events';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// runpodCli pulls in config.js, which hard-exits on missing required env vars.
// Provide throwaway values so the module loads under the test runner.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

// Mock the child process so we can drive the close handler directly: emit a
// "Pod ID:" line, then close with (code, signal) the way Node does on a kill.
const h = vi.hoisted(() => ({ spawn: vi.fn() }));
vi.mock('node:child_process', () => ({ spawn: h.spawn }));

const { runViaCli, isControllerRetryable, runVariantWithRetry } = await import(
  '../src/services/runpodCli.js'
);

function fakeChild() {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  return child;
}

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

describe('runViaCli close handler', () => {
  beforeEach(() => h.spawn.mockReset());

  it('rejects non-retryable when SIGKILLed after a pod was created', async () => {
    const child = fakeChild();
    h.spawn.mockReturnValue(child);

    const p = runViaCli({ modelId: 'org/model', variants: ['5.0'] });
    // Controller announced its pod, then an operator SIGKILLed it to cancel a
    // broken model (code null, signal set).
    child.stdout.emit('data', Buffer.from('Pod ID: abc123\n'));
    child.emit('close', null, 'SIGKILL');

    await expect(p).rejects.toMatchObject({
      retryable: false,
      signal: 'SIGKILL',
      podCreated: true,
    });
  });

  it('rejects non-retryable when SIGKILLed during provisioning (no pod id seen yet)', async () => {
    const child = fakeChild();
    h.spawn.mockReturnValue(child);

    const p = runViaCli({ modelId: 'org/model', variants: ['5.0'] });
    // This is the runaway repro: killed before the "Pod ID:" line. The old
    // guard (retryable = !podId) made this retryable and respawned a pod.
    child.emit('close', null, 'SIGKILL');

    await expect(p).rejects.toMatchObject({ retryable: false, signal: 'SIGKILL' });
  });

  it('keeps a clean launch failure (no pod, no signal) retryable', async () => {
    const child = fakeChild();
    h.spawn.mockReturnValue(child);

    const p = runViaCli({ modelId: 'org/model', variants: ['5.0'] });
    child.emit('close', 1, null); // a genuine stock/launch failure before any pod

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
