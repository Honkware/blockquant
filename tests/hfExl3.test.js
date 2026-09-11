import { describe, it, expect } from 'vitest';
import { exl3RepoName } from '../src/utils/hfExl3.js';

// The pod uploads to `{model}-exl3-{variant}bpw` where `variant` is the string
// quant.js normalized the request to. Anything the bot looks up has to match
// that byte for byte, or the pre-flight checks a repo that does not exist.
// It was toFixed(2) for a while, and nothing here caught it.
describe('exl3RepoName', () => {
  it('matches the names actually published on HF', () => {
    expect(exl3RepoName('MiniCPM5-2B', 6)).toBe('MiniCPM5-2B-exl3-6.0bpw');
    expect(exl3RepoName('Qwen3.8-27B', 4.5)).toBe('Qwen3.8-27B-exl3-4.5bpw');
    expect(exl3RepoName('Qwen3.8-27B', 3.8)).toBe('Qwen3.8-27B-exl3-3.8bpw');
  });

  it('agrees with the variant strings quant.js produces', () => {
    // quant.js:127 -- Number.isInteger(b) ? b.toFixed(1) : String(b)
    const normalize = (b) => (Number.isInteger(b) ? b.toFixed(1) : String(b));
    for (const bpw of [2, 2.5, 3, 3.8, 4, 4.5, 5, 6, 8]) {
      expect(exl3RepoName('M', bpw)).toBe(`M-exl3-${normalize(bpw)}bpw`);
    }
  });

  it('rejects a bpw that is not a number', () => {
    expect(() => exl3RepoName('M', 'four')).toThrow(TypeError);
  });

  it('marks a quantized vision tower, and says nothing when there is none', () => {
    // Absent -V means the tower was copied at fp16, which is how exllamav3
    // records it too: quantization_config carries vision_bits only when the
    // tower was quantized.
    expect(exl3RepoName('Qwen3-VL-8B', 4)).toBe('Qwen3-VL-8B-exl3-4.0bpw');
    expect(exl3RepoName('Qwen3-VL-8B', 4, { visionBits: 6 })).toBe('Qwen3-VL-8B-exl3-4.0bpw-V6');
  });

  it('names head bits on a self-calibrated build', () => {
    expect(exl3RepoName('Qwen3.8-27B', 5, { sc: true, headBits: 6 })).toBe(
      'Qwen3.8-27B-exl3-SC-5.0bpw-H6'
    );
    expect(exl3RepoName('Qwen3.8-27B', 4, { sc: true, headBits: 5, visionBits: 6 })).toBe(
      'Qwen3.8-27B-exl3-SC-4.0bpw-H5-V6'
    );
  });

  it('keeps one decimal unless more of them say something', () => {
    // 4 and '4.00' are the same quant, so they must not produce two names.
    // A fractional SC target really can land on 4.07, so those digits stay.
    expect(exl3RepoName('M', '4.00')).toBe(exl3RepoName('M', 4));
    expect(exl3RepoName('M', '4.50')).toBe('M-exl3-4.5bpw');
    expect(exl3RepoName('M', '4.05')).toBe('M-exl3-4.05bpw');
    expect(exl3RepoName('M', '3.14')).toBe('M-exl3-3.14bpw');
  });

  it('refuses an SC name with no head bits', () => {
    // A plain -exl3-4.0bpw already means bundled calibration, so an SC build
    // cannot fall back to it: one name would then describe two artifacts.
    expect(() => exl3RepoName('M', 4, { sc: true })).toThrow(TypeError);
  });
});
