import { describe, it, expect, vi } from 'vitest';

process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

// Capture what the controller would be launched with. The launcher never
// resolves here -- only the argv matters.
const h = vi.hoisted(() => ({ args: null }));
vi.mock('../src/services/detached.js', () => ({
  spawnDetached: (opts) => {
    h.args = opts.args;
    return { id: 'test', pid: process.pid, logPath: opts.logPath };
  },
  wait: () => new Promise(() => {}),
}));

const { runViaCli } = await import('../src/services/runpodCli.js');

// Reads the value that follows a flag, or undefined when the flag is absent.
function flag(args, name) {
  const i = args.indexOf(name);
  return i === -1 ? undefined : args[i + 1];
}

describe('run_runpod_job argv', () => {
  it('passes head bits when the request pinned them', () => {
    runViaCli({ modelId: 'org/model', variants: ['4.0'], headBits: 8 });
    expect(flag(h.args, '--head-bits')).toBe('8');
  });

  it('omits head bits otherwise, so exllamav3 picks its own default', () => {
    // This flag was missing altogether for a while and every quant silently
    // came out at the controller's default of 8.
    runViaCli({ modelId: 'org/model', variants: ['4.0'] });
    expect(h.args).not.toContain('--head-bits');
  });

  it('carries the model, variants and codebook', () => {
    runViaCli({ modelId: 'org/model', variants: ['3.0', '4.0'], codebook: 'mcg' });
    expect(flag(h.args, '--model')).toBe('org/model');
    expect(flag(h.args, '--variants')).toBe('3.0,4.0');
    expect(flag(h.args, '--codebook')).toBe('mcg');
  });

  it('maps vision fp16 to 16 bits and auto to no flag at all', () => {
    runViaCli({ modelId: 'org/model', variants: ['4.0'], vision: 'fp16' });
    expect(flag(h.args, '--vision-bits')).toBe('16');

    runViaCli({ modelId: 'org/model', variants: ['4.0'], vision: '6' });
    expect(flag(h.args, '--vision-bits')).toBe('6');

    runViaCli({ modelId: 'org/model', variants: ['4.0'], vision: 'auto' });
    expect(h.args).not.toContain('--vision-bits');
  });

  it('only passes a test prompt when there is one', () => {
    runViaCli({ modelId: 'org/model', variants: ['4.0'], testPrompt: 'draw a cat' });
    expect(flag(h.args, '--test-prompt')).toBe('draw a cat');

    runViaCli({ modelId: 'org/model', variants: ['4.0'] });
    expect(h.args).not.toContain('--test-prompt');
  });
});
