import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { defaultHeadBits, defaultVisionBits } from '../src/utils/archSupport.js';
import { exl3RepoName, quantizedVisionBits } from '../src/utils/hfExl3.js';

const table = JSON.parse(readFileSync('backend/arch_support.json', 'utf8'));

describe('arch defaults', () => {
  it('reads exllamav3 head bits off the generated table', () => {
    expect(defaultHeadBits()).toBe(table.default_head_bits);
  });

  it('gives a validated tower its cap', () => {
    expect(defaultVisionBits('Qwen3VLForConditionalGeneration')).toBe(6);
    // The MoE variants import their VisionModel from the dense sibling and
    // declare no cap of their own; a per-file grep missed them entirely.
    expect(defaultVisionBits('Qwen3VLMoeForConditionalGeneration')).toBe(6);
  });

  it('gives everything else 16, the same literal convert_model.py falls back to', () => {
    expect(defaultVisionBits('Qwen2_5_VLForConditionalGeneration')).toBe(16);
    expect(defaultVisionBits('Qwen3ForCausalLM')).toBe(16);
    expect(defaultVisionBits('NoSuchArch')).toBe(16);
  });

  it('every tower in the table is a real architecture', () => {
    const archs = new Set(table.architectures);
    for (const a of Object.keys(table.default_vision_bits)) expect(archs).toContain(a);
  });
});

describe('the -V suffix', () => {
  it('names a quantized tower', () => {
    expect(exl3RepoName('M', 4, { sc: true, headBits: 6, visionBits: 6 }))
      .toBe('M-exl3-SC-4.0bpw-H6-V6');
  });

  // 16 is how a request spells "copy the tower whole". Naming that -V16 claims
  // a quantized tower the repo does not have.
  it.each([16, 0, null, undefined])('says nothing for %s', (v) => {
    expect(quantizedVisionBits(v)).toBeNull();
    expect(exl3RepoName('M', 4, { visionBits: v })).toBe('M-exl3-4.0bpw');
  });
});
