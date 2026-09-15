import { describe, it, expect, vi } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const h = vi.hoisted(() => ({ args: null }));
vi.mock('../src/services/detached.js', () => ({
  spawnDetached: (o) => { h.args = o.args; return {}; },
  wait: () => new Promise(() => {}),
}));
const { runViaCli } = await import('../src/services/runpodCli.js');
const flag = (a, n) => (a.indexOf(n) === -1 ? undefined : a[a.indexOf(n) + 1]);

describe('self-calibration argv', () => {
  it('sends the flag and the donor together', () => {
    runViaCli({ modelId: 'o/m', variants: ['4.0'], sc: true,
                donorRepo: 'o/m-exl3-6.0bpw', headBits: 6 });
    expect(h.args).toContain('--sc');
    expect(flag(h.args, '--donor-repo')).toBe('o/m-exl3-6.0bpw');
    expect(flag(h.args, '--head-bits')).toBe('6');
  });

  // The launcher enforces this, but only once a controller is spawning. The
  // bot resolved head bits for the published name and left the flag unsent, so
  // every SC job with the option blank failed at launch.
  it('refuses to build an SC job with no head bits', async () => {
    await expect(runViaCli({ modelId: 'o/m', variants: ['4.0'], sc: true,
                             donorRepo: 'o/m-exl3-6.0bpw' }))
      .rejects.toThrow(/headBits/);
  });

  it('sends neither for a plain quant', () => {
    runViaCli({ modelId: 'o/m', variants: ['4.0'] });
    expect(h.args).not.toContain('--sc');
    expect(h.args).not.toContain('--donor-repo');
  });
});

describe("keep-pod passthrough", () => {
  it("sends the flag only when asked", () => {
    runViaCli({ modelId: "o/m", variants: ["4.0"] });
    expect(h.args).not.toContain("--keep-pod");
    runViaCli({ modelId: "o/m", variants: ["4.0"], keepPod: true });
    expect(h.args).toContain("--keep-pod");
  });
});
