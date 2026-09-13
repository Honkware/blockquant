import { describe, it, expect, vi } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const h = vi.hoisted(() => ({ args: null }));
vi.mock('../src/services/detached.js', () => ({
  spawnDetached: (o) => { h.args = o.args; return {}; },
  wait: () => new Promise(() => {}),
}));
const { runViaCli } = await import('../src/services/runpodCli.js');
const flag = (a, n) => (a.indexOf(n) === -1 ? undefined : a[a.indexOf(n) + 1]);

describe('subfolder argv', () => {
  it('passes a subfolder when the model lives in one', () => {
    runViaCli({ modelId: 'org/multi', variants: ['4.0'], subfolder: 'BF16' });
    expect(flag(h.args, '--subfolder')).toBe('BF16');
  });

  it('sends nothing for the normal root-level repo', () => {
    runViaCli({ modelId: 'org/model', variants: ['4.0'] });
    expect(h.args).not.toContain('--subfolder');
  });
});
