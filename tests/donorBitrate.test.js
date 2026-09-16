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

  it('does not overshoot the target it has to serve', () => {
    // The donor is generated through on every calibration row, so a fatter one
    // is a slower trace for nothing.
    expect(donorBitrateFor(4.5)).toBe('4.5');
    expect(donorBitrateFor(6.0)).toBe('6.0');
  });

  it('rounds a fractional target up to a sensible artifact', () => {
    // 3.6 would publish a 3.6bpw donor nobody asked for.
    expect(donorBitrateFor(3.6)).toBe('4.0');
    expect(donorBitrateFor(4.1)).toBe('4.5');
  });

  it('floors at 4.0 for low targets', () => {
    // The donor writes the calibration corpus; a 2bpw model generating it is
    // a worse starting point than the extra conversion time costs.
    expect(donorBitrateFor(2.0)).toBe('4.0');
    expect(donorBitrateFor(3.0)).toBe('4.0');
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
