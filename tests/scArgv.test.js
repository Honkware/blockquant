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

describe('convert settings argv', () => {
  it('sends sc-lite only with sc', () => {
    runViaCli({ modelId: 'o/m', variants: ['3.0'], sc: true, scLite: true,
                donorRepo: 'o/m-exl3-6.0bpw', headBits: 6, hq: 2 });
    expect(h.args).toContain('--sc-lite');
    expect(flag(h.args, '--hq')).toBe('2');
    runViaCli({ modelId: 'o/m', variants: ['3.0'], scLite: true });
    expect(h.args).not.toContain('--sc-lite');
  });

  it('passes pinned MTP and n-gram bits, and nothing when unset', () => {
    runViaCli({ modelId: 'o/m', variants: ['4.0'], mtpBits: 2.5, ngramBits: 6, hq: 1 });
    expect(flag(h.args, '--mtp-bits')).toBe('2.5');
    expect(flag(h.args, '--ngram-bits')).toBe('6');
    expect(flag(h.args, '--hq')).toBe('1');
    runViaCli({ modelId: 'o/m', variants: ['4.0'] });
    for (const f of ['--mtp-bits', '--ngram-bits', '--hq']) expect(h.args).not.toContain(f);
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

describe("gpu count passthrough", () => {
  it("sends the flag only when asked", () => {
    runViaCli({ modelId: "o/m", variants: ["4.0"] });
    expect(h.args).not.toContain("--gpu-count");
    runViaCli({ modelId: "o/m", variants: ["4.0"], gpuCount: 2 });
    expect(flag(h.args, "--gpu-count")).toBe("2");
  });
});

describe("max runtime passthrough", () => {
  it("is left at the launcher default unless asked", () => {
    runViaCli({ modelId: "o/m", variants: ["4.0"] });
    expect(h.args).not.toContain("--max-runtime");
  });

  it("goes over in seconds, because that is what the CLI takes", () => {
    // 8h is ample for a conversion and thin for a big SC job, where the trace
    // and the measurement both run before any weight is quantized.
    runViaCli({ modelId: "o/m", variants: ["4.0"], maxRuntimeH: 12 });
    expect(flag(h.args, "--max-runtime")).toBe(String(12 * 3600));
  });
});
