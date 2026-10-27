import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { jobOutcome } = await import('../src/commands/quant.js');

const ok = (v, url) => ({ variant: v, bpw: v, pushed: true, url, error: null });
const bad = (v, error) => ({ variant: v, bpw: v, pushed: false, url: null, error });

describe('judging a finished job', () => {
  it('fails a self-calibrated job whose only success was its own donor', () => {
    // The first 27B run, exactly: the 3.6 SC variant died in sc_measure, the
    // 4.0 donor pushed, and the job was recorded as completed.
    const o = jobOutcome(['3.6'], [
      ok('4.0', 'https://huggingface.co/Honkware/Swift-Qwen3.8-27b-exl3-4.0bpw'),
      bad('3.6', 'pod exited without a result file and nothing reached HF'),
    ]);
    expect(o.status).toBe('failed');
    expect(o.allDone).toBe(false);
    expect(o.failed).toEqual(['3.6']);
    expect(o.reason).toMatch(/pod exited/);
  });

  it('still reports the donor that did land', () => {
    const o = jobOutcome(['3.6'], [ok('4.0', 'u/donor'), bad('3.6', 'x')]);
    expect(o.extras).toEqual(['u/donor']);
  });

  it('is done when everything asked for landed', () => {
    const o = jobOutcome(['3.6'], [ok('4.0', 'u/donor'), ok('3.6', 'u/sc')]);
    expect(o.status).toBe('completed');
    expect(o.allDone).toBe(true);
    expect(o.failed).toEqual([]);
  });

  it('calls a mixed multi-bitrate job completed but names what failed', () => {
    const o = jobOutcome(['3.0', '4.0'], [ok('3.0', 'u/3'), bad('4.0', 'oom')]);
    expect(o.status).toBe('completed');
    expect(o.allDone).toBe(false);
    expect(o.failed).toEqual(['4.0']);
  });

  it('fails a plain job where nothing landed', () => {
    const o = jobOutcome(['4.0'], [bad('4.0', 'no GPU')]);
    expect(o.status).toBe('failed');
  });

  it('matches variants whatever form the bitrate arrived in', () => {
    // results carry both variant and bpw, and either can be a string.
    const o = jobOutcome(['5.0'], [{ bpw: '5.0', pushed: true, url: 'u' }]);
    expect(o.allDone).toBe(true);
  });

  it('does not count a failed donor as an extra', () => {
    const o = jobOutcome(['3.6'], [bad('4.0', 'x'), bad('3.6', 'y')]);
    expect(o.extras).toEqual([]);
  });
});

describe('a donor at the same bitrate as its target', () => {
  // SC 4.0 builds a 4.0 donor; the two share a label.
  it('does not let the donor stand in for a failed SC variant', () => {
    const o = jobOutcome(['4.0'], [{ ...ok('4.0', 'u/donor'), donor: true }, bad('4.0', 'oom')]);
    expect(o.status).toBe('failed');
    expect(o.failed).toEqual(['4.0']);
    expect(o.extras).toEqual(['u/donor']);
  });

  it('is done when the SC variant landed', () => {
    const o = jobOutcome(['4.0'], [{ ...ok('4.0', 'u/donor'), donor: true }, ok('4.0', 'u/sc')]);
    expect(o.allDone).toBe(true);
    expect(o.extras).toEqual(['u/donor']);
  });
});
