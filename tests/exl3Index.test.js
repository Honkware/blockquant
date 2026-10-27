import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { isQuantOf, describeQuant } = await import('../src/services/exl3Index.js');

describe('which repos are quants of a model', () => {
  it('takes the exact model name before exl3, however it is spelled', () => {
    expect(isQuantOf('o/Qwen3.8-27B-exl3-4.0bpw', 'Qwen3.8-27B')).toBe(true);
    expect(isQuantOf('o/Qwen3.8-27B-EXL3-3.5bpw', 'Qwen3.8-27B')).toBe(true);
    expect(isQuantOf('o/Qwen3.8-27B_exl3-8bpwh8', 'Qwen3.8-27B')).toBe(true);
  });

  it('does not count a fine-tune as a quant of its base', () => {
    expect(isQuantOf('o/Swift-Qwen3.8-27b-exl3-4.0bpw', 'Qwen3.8-27B')).toBe(false);
    expect(isQuantOf('o/Qwen3.8-27B-antislop-exl3-4.0bpw', 'Qwen3.8-27B')).toBe(false);
  });
});

describe('what a quant name says', () => {
  it('reads the bitrate in the shapes people use', () => {
    expect(describeQuant('o/M-exl3-4.0bpw').bpw).toBe(4);
    expect(describeQuant('o/M-EXL3-EAQv1-4-bpw').bpw).toBe(4);
    expect(describeQuant('o/TandemLLM-M-EXL3-4.5').bpw).toBe(4.5);
    expect(describeQuant('o/M-exl3').bpw).toBe(null);
  });

  it('tells SC from SCT from plain', () => {
    expect(describeQuant('o/M-exl3-SC-3.0bpw-H6').kind).toBe('SC');
    expect(describeQuant('o/M-exl3-SCT-3.0bpw-H6-HQ2').kind).toBe('SCT');
    expect(describeQuant('o/M-exl3-2.65bpw-SC').kind).toBe('SC');
    expect(describeQuant('o/M-exl3-4.0bpw').kind).toBe(null);
  });
});
