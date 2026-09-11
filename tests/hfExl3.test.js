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
});
