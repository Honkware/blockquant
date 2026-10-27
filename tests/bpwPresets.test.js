import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { expandBpw, BPW_PRESETS, podRuntimeHours } = await import('../src/commands/quant.js');

describe('bpw presets', () => {
  it('expands a preset in place and keeps typed numbers', () => {
    expect(expandBpw('low')).toEqual(BPW_PRESETS.low);
    expect(expandBpw('low, 6.0')).toEqual([...BPW_PRESETS.low, '6.0']);
    expect(expandBpw('3.6')).toEqual(['3.6']);
  });

  it('is case-insensitive and drops repeats', () => {
    expect(expandBpw('LOW,low,3.0')).toEqual(BPW_PRESETS.low);
  });

  it('leaves a typo for the caller to refuse rather than dropping it', () => {
    // It used to be filtered out as NaN and the rest ran without it.
    expect(expandBpw('4.0,lwo')).toEqual(['4.0', 'lwo']);
  });
});

describe('pod runtime limit', () => {
  it('stays at the launcher default for an ordinary convert', () => {
    expect(podRuntimeHours({ sizeGb: 15 })).toBe(8);
    expect(podRuntimeHours({})).toBe(8);
  });

  it('gives an SC ladder room to finish on its one pod', () => {
    // A 27B low ladder: calibration then four conversions, serially.
    const h = podRuntimeHours({ sc: true, sizeGb: 51.75, nVariants: 4 });
    expect(h).toBeGreaterThan(8);
    expect(h).toBeLessThanOrEqual(24);
  });

  it('gives trace mode less than full SC', () => {
    const full = podRuntimeHours({ sc: true, sizeGb: 51.75, nVariants: 4 });
    const lite = podRuntimeHours({ sc: true, scLite: true, sizeGb: 51.75, nVariants: 4 });
    expect(lite).toBeLessThan(full);
  });
});
