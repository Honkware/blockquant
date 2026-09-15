import { describe, it, expect, vi } from 'vitest';
import { mkdtempSync, writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const h = vi.hoisted(() => ({ handle: null, tick: null }));
vi.mock('../src/services/detached.js', () => ({
  spawnDetached: (o) => ({ args: o.args }),
  // attachToCli drains on each tick and resolves when wait() resolves.
  wait: (handle, { onTick }) => { h.tick = onTick; return new Promise(() => {}); },
}));

const { attachToCli, scProgress } = await import('../src/services/runpodCli.js');

/** Write a real controller log, tail it the way attachToCli does, collect reports. */
function feed(lines, { sc = true } = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'bq-sc-'));
  const logPath = join(dir, 'ctrl.log');
  writeFileSync(logPath, lines.join('\n') + '\n');
  const seen = [];
  attachToCli({ logPath }, { variants: ['3.0'], sc, onProgress: (d) => seen.push(d) });
  h.tick();   // drain() reads the file from disk
  return seen;
}

// Verbatim from the controller log of the first successful SC run.
const REAL = [
  '[sc] donor Honkware/Qwen3.5-0.8B-exl3-4.0bpw ...',
  '[sc] donor ready (1.1 GB)',
  '[sc] sc_trace ...',
  '[sc] sc_trace: 137s conv  24 turn 0:  3072 tokens (max_new_tokens)   total    68,304',
  '[sc] sc_trace done -> cal.safetensors',
  '[sc] sc_rfn_probe ...',
  '[sc] sc_rfn_probe done -> rfn.json',
  '[sc] sc_measure ...',
  '[sc] sc_measure done -> measure.json',
  '[sc] sc_optimize ...',
  '[sc] sc_optimize done -> recipe-3.0.yaml',
];

describe('self-calibration in the progress stream', () => {
  it('reports every SC line instead of going silent', () => {
    // These matched no pattern at all, so report() never fired and the embed
    // sat unchanged for the whole ~42 min calibration.
    expect(feed(REAL).length).toBe(REAL.length);
  });

  it('calls the phase Calibrating', () => {
    expect(feed(REAL).every((d) => d.stage === 'Calibrating')).toBe(true);
  });

  it('moves the bar forward as stages land', () => {
    const seen = feed(REAL);
    const overalls = seen.map((d) => d.overall);
    expect(overalls[0]).toBeLessThan(overalls[overalls.length - 1]);
    expect(Math.max(...overalls)).toBeLessThanOrEqual(55);
    // Monotonic: the controller reprints its window, so stale lines must never
    // knock the bar backward.
    expect([...overalls].sort((a, b) => a - b)).toEqual(overalls);
  });

  it('shows elapsed minutes from a heartbeat', () => {
    const [d] = feed(['[sc] sc_trace: 137s conv 24 turn 0: 3072 tokens']);
    expect(d.message).toContain('sc_trace');
    expect(d.message).toContain('2m');
  });

  it('survives a heartbeat with no elapsed stamp', () => {
    // Older pods run the pre-elapsed heartbeat; upstream text changes freely.
    const [d] = feed(['[sc] sc_trace: -- Loading /quant/blockquant/donor']);
    expect(d.stage).toBe('Calibrating');
    expect(d.message).toContain('sc_trace');
  });

  it('weights the stages by what they actually cost', () => {
    // 25.5 min vs 12s vs 16.3 min vs 4s, measured. Equal quarters would park
    // the bar for most of an hour.
    expect(scProgress(new Set(['sc_rfn_probe']))).toBeLessThan(0.05);
    expect(scProgress(new Set(['sc_trace']))).toBeGreaterThan(0.5);
    expect(scProgress(new Set(['sc_trace', 'sc_rfn_probe', 'sc_measure', 'sc_optimize']))).toBe(1);
  });

  it('leaves a plain quant on the old split', () => {
    // No SC phase is coming, so the bar must not reserve room for one.
    const seen = feed(['[download] complete', '[progress] quantize 3.0 50% (12/24)'], { sc: false });
    const quant = seen[seen.length - 1];
    expect(quant.stage).toBe('Quantizing');
    expect(quant.overall).toBe(25 + Math.round(0.5 * 65));
  });

  it('gives an SC quantize phase its own narrower band', () => {
    const seen = feed(['[progress] quantize 3.0 50% (12/24)'], { sc: true });
    const quant = seen[seen.length - 1];
    expect(quant.stage).toBe('Quantizing');
    expect(quant.overall).toBe(55 + Math.round(0.5 * 35));
  });
});
