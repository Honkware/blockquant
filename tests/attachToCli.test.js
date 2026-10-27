import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, it, expect, vi } from 'vitest';

process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

// wait() polls a status file in the real thing. Drive it by hand so the test
// controls when the "controller" exits.
const h = vi.hoisted(() => ({ finish: null, spawned: 0 }));
vi.mock('../src/services/detached.js', () => ({
  spawnDetached: () => {
    h.spawned += 1;
    return {};
  },
  wait: (handle, { onTick } = {}) =>
    new Promise((resolve) => {
      const timer = setInterval(() => onTick?.(), 5);
      h.finish = (code, signal) => {
        clearInterval(timer);
        onTick?.();
        resolve({ code, signal });
      };
    }),
}));

const { attachToCli } = await import('../src/services/runpodCli.js');

function fakeHandle() {
  const logPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'bq-')), 'ctrl.log');
  fs.writeFileSync(logPath, '');
  return { id: 'ctrl-test', pid: process.pid, logPath, statusPath: `${logPath}.status` };
}

describe('attachToCli', () => {
  it('drives progress from an existing log without spawning anything', async () => {
    const handle = fakeHandle();
    const seen = [];
    const p = attachToCli(handle, { variants: ['4.0'], onProgress: (d) => seen.push(d) });

    // A restarted bot re-reads a log that already has history in it.
    fs.appendFileSync(handle.logPath, 'Pod ID: abc123\n[download] 50%\n');
    await new Promise((r) => setTimeout(r, 30));
    fs.appendFileSync(handle.logPath, '[upload] 4.0 done -> https://huggingface.co/o/m-exl3-4.0bpw\n');
    h.finish(0, null);

    const out = await p;
    expect(h.spawned).toBe(0); // attaching must never start a second controller
    expect(out[0].url).toBe('https://huggingface.co/o/m-exl3-4.0bpw');
    expect(seen.some((d) => d.podId === 'abc123')).toBe(true);
  });

  it('returns a drafter as a flagged extra, never as a variant', async () => {
    const handle = fakeHandle();
    const p = attachToCli(handle, { variants: ['4.0'], onProgress: () => {} });
    fs.appendFileSync(handle.logPath,
      '[upload] 4.0 done -> https://huggingface.co/o/m-exl3-4.0bpw\n'
      + '[drafter] done -> https://huggingface.co/o/m-DFlash2-exl3-4.0bpw\n');
    await new Promise((r) => setTimeout(r, 30));
    h.finish(0, null);
    const out = await p;
    expect(out).toHaveLength(2);
    expect(out[0].variant).toBe('4.0');
    expect(out[1]).toMatchObject({ drafter: true, pushed: true,
                                   url: 'https://huggingface.co/o/m-DFlash2-exl3-4.0bpw' });
  });

  it('reads what the pod cost, on success and on failure', async () => {
    let handle = fakeHandle();
    let p = attachToCli(handle, { variants: ['4.0'], onProgress: () => {} });
    fs.appendFileSync(handle.logPath, '[upload] 4.0 done -> https://huggingface.co/o/m-exl3-4.0bpw\n'
      + '[cost] 1905s on NVIDIA RTX PRO 6000 Blackwell Workstation Edition (COMMUNITY) at $1.69/h = $0.89\n');
    await new Promise((r) => setTimeout(r, 30));
    h.finish(0, null);
    expect((await p)[0].podCost).toMatchObject({ secs: 1905, cloud: 'COMMUNITY', cost: 0.89 });

    handle = fakeHandle();
    p = attachToCli(handle, { variants: ['3.6'], onProgress: () => {} });
    fs.appendFileSync(handle.logPath, '[joberror] pod gone\n'
      + '[cost] 11520s on NVIDIA A100-SXM4-80GB x2 (SECURE) at $3.18/h = $10.18\n');
    await new Promise((r) => setTimeout(r, 30));
    h.finish(1, null);
    await expect(p).rejects.toMatchObject({ podCost: { cost: 10.18, gpu: 'NVIDIA A100-SXM4-80GB x2' } });
  });

  it('reports the controller error when it dies with no upload', async () => {
    const handle = fakeHandle();
    const p = attachToCli(handle, { variants: ['4.0'], onProgress: () => {} });
    fs.appendFileSync(handle.logPath, '[joberror] unsupported architecture\n');
    await new Promise((r) => setTimeout(r, 20));
    h.finish(2, null);

    await expect(p).rejects.toThrow(/unsupported architecture/);
  });
});
