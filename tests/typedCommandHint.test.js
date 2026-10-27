import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { looksLikeTypedCommand } = await import('../src/services/typedCommandHint.js');

describe('a slash command typed as text', () => {
  it('catches the 10-17 messages as they were posted', () => {
    expect(looksLikeTypedCommand('./quant url:ukisai/Swift-1.5-Qwen3.8-27b bpw:3.0,3.2,3.5 sc:True')).toBe(true);
    expect(looksLikeTypedCommand('/quant url:x bpw:4')).toBe(true);
    expect(looksLikeTypedCommand('/quants model:x')).toBe(true);
  });

  it('leaves ordinary talk about quants alone', () => {
    expect(looksLikeTypedCommand('quant bot is dead :(')).toBe(false);
    expect(looksLikeTypedCommand('I quant my own with SC')).toBe(false);
    expect(looksLikeTypedCommand('use /quant for that')).toBe(false);
  });
});
