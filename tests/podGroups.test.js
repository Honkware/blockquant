import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { podGroups } = await import('../src/commands/quant.js');

describe('how variants are spread over pods', () => {
  it('gives a plain quant one pod per bitrate', () => {
    // Independent work, so they run in parallel and finish sooner.
    expect(podGroups(['3.0', '4.0', '5.0'], false)).toEqual([['3.0'], ['4.0'], ['5.0']]);
  });

  it('keeps an SC job on one pod', () => {
    // The trace, probe and measure are per MODEL. Three pods would each
    // generate the same 512,000 calibration tokens -- ~40 min of it, measured.
    expect(podGroups(['3.0', '4.0', '5.0'], true)).toEqual([['3.0', '4.0', '5.0']]);
  });

  it('is the same either way for a single bitrate', () => {
    expect(podGroups(['3.0'], true)).toEqual([['3.0']]);
    expect(podGroups(['3.0'], false)).toEqual([['3.0']]);
  });

  it('does not hand out the caller\'s array', () => {
    // The group is mutated downstream (retry bookkeeping); aliasing `variants`
    // would edit the job record.
    const variants = ['3.0', '4.0'];
    const [g] = podGroups(variants, true);
    g.push('9.9');
    expect(variants).toEqual(['3.0', '4.0']);
  });

  it('covers every variant exactly once in both modes', () => {
    for (const sc of [true, false]) {
      const flat = podGroups(['3.0', '4.0', '5.0'], sc).flat();
      expect(flat.sort()).toEqual(['3.0', '4.0', '5.0']);
    }
  });
});
