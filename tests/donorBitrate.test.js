import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { donorBitrateFor } = await import('../src/commands/quant.js');

describe('picking a donor to build', () => {
  it('is never below the target', () => {
    // sc_trace samples through it and sc_rfn_probe walks its module tree
    // against the target; a thinner donor is refused downstream.
    for (const t of [3.0, 3.6, 4.2, 5.0, 6.0, 7.5]) {
      expect(Number(donorBitrateFor(t))).toBeGreaterThanOrEqual(t);
    }
  });

  it('traces through 6bpw at the least', () => {
    // The calibration set is generated through the donor, so its errors are
    // what the quant is calibrated toward. turboderp and ezexl3 use 6bpw or
    // better; a 6bpw teacher measured no different from bf16 on Gemma 4.
    // The 27B's 3.6 was traced through a 4.0.
    expect(donorBitrateFor(3.6)).toBe('6.0');
    expect(donorBitrateFor(2.0)).toBe('6.0');
    expect(donorBitrateFor(4.5)).toBe('6.0');
    expect(donorBitrateFor(6.0)).toBe('6.0');
  });

  it('rounds a target above the floor up to a half bit', () => {
    expect(donorBitrateFor(6.2)).toBe('6.5');
    expect(donorBitrateFor(7.0)).toBe('7.0');
  });

  it('never exceeds what exllamav3 takes', () => {
    expect(Number(donorBitrateFor(8.0))).toBeLessThanOrEqual(8);
    expect(Number(donorBitrateFor(12))).toBeLessThanOrEqual(8);
  });

  it('emits the one-decimal form the repo names use', () => {
    // Anything else and findDonorQuant will not match the published repo.
    for (const t of [3.0, 3.6, 5.0]) {
      expect(donorBitrateFor(t)).toMatch(/^\d\.\d$/);
    }
  });

  it('refuses a target it cannot read', () => {
    expect(() => donorBitrateFor('nonsense')).toThrow(TypeError);
  });
});
