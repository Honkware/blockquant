import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// catbenchCli pulls in config.js, which hard-exits on missing required env vars.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

// Same shape as tests/runpodCli.test.js: mock the launcher, not child_process.
// The controller is reparented to init and reports through files, so there is
// no child object to emit events on. `finish` resolves wait() the way a
// recorded exit status does, after the log the real code tails has a line in it.
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

const { runCatbench } = await import('../src/services/catbenchCli.js');
const config = (await import('../src/config.js')).default;

const LOG_DIR = path.join(config.ROOT_DIR, 'backend', 'logs');
const results = () => fs.readdirSync(LOG_DIR).filter((n) => /^catbench-.+-\d+\.json$/.test(n));
const seeded = [];

function seed(stamp) {
  const p = path.join(LOG_DIR, `catbench-seed_m-${stamp}.json`);
  fs.mkdirSync(LOG_DIR, { recursive: true });
  fs.writeFileSync(p, '{}');
  seeded.push(p);
  return p;
}

describe('runCatbench finish handler', () => {
  beforeEach(() => {
    h.logPath = '';
    h.finish = null;
    seeded.length = 0;
  });

  afterEach(() => {
    for (const p of [...seeded, h.logPath, h.logPath.replace(/\.log$/, '.json')]) {
      if (p) fs.rmSync(p, { force: true });
    }
  });

  it('leaves the result file on disk, so a pod is not paid for twice', async () => {
    const p = runCatbench({ modelId: 'org/model' });
    const out = h.logPath.replace(/\.log$/, '.json');
    fs.writeFileSync(out, JSON.stringify({ status: 'complete', svg: '<svg/>' }));
    fs.writeFileSync(h.logPath, `[catbench] result ${out}\n`);
    h.finish(0, null);

    await expect(p).resolves.toMatchObject({ status: 'complete' });
    // It used to unlink this the instant it parsed it, which left a delivery
    // that failed with nothing to recover from.
    expect(fs.existsSync(out)).toBe(true);
  });

  it('reports a killed controller instead of a missing result', async () => {
    const p = runCatbench({ modelId: 'org/model' });
    h.finish(null, 'SIGKILL');
    await expect(p).rejects.toThrow(/SIGKILL/);
  });

  it('trims old results on the next run, never on the way out', async () => {
    // Future stamps so these are the newest whatever else the box has.
    for (let i = 0; i < 25; i++) seed(9_000_000_000_000 + i);
    expect(results().length).toBeGreaterThanOrEqual(25);

    const p = runCatbench({ modelId: 'org/model' });
    h.finish(1, null);
    await expect(p).rejects.toThrow();

    // 20 kept by the sweep at the start, plus this run's own log-less entry.
    expect(fs.existsSync(seeded[0])).toBe(false); // oldest of the 25
    expect(fs.existsSync(seeded[24])).toBe(true); // newest of the 25
    expect(results().length).toBe(20);
  });
});
