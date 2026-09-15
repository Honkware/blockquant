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

const { attachToCli, scProgress, scTraceFrac, scMeasureFrac, SC_TRACE_TOKENS } =
  await import('../src/services/runpodCli.js');

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


describe('colour codes from upstream', () => {
  // Verbatim shape from the controller log: turboderp colours the token
  // counter, and the heartbeat forwards whatever fragment is newest, so the
  // escapes ride along into the embed and render as literal junk.
  const esc = String.fromCharCode(27);

  it('never reach the embed', () => {
    const raw = `[sc] sc_trace: 137s conv  24 turn 0: ${esc}[33;1m 3072${esc}[0m tokens   total ${esc}[32;1m  68,304${esc}[0m`;
    const [d] = feed([raw]);
    expect(d.message).not.toContain(esc);
    expect(d.message).not.toContain('[33;1m');
    expect(d.message).toContain('3072');
  });

  it('do not eat the message budget', () => {
    const raw = `[sc] sc_trace: 137s ${esc}[33;1m${'x'.repeat(40)}${esc}[0m`;
    const [d] = feed([raw]);
    expect(d.message).toContain('x'.repeat(40));
  });

  it('leave a plain line untouched', () => {
    const [d] = feed(['[sc] sc_trace done -> cal.safetensors']);
    expect(d.message).toBe('sc_trace done');
  });
});


describe('granularity inside the long stage', () => {
  // sc_trace is the longest stage in the job -- 25.5 min of the 42 measured --
  // and crediting only on completion left the bar on one number for all of it.
  const beat = (n) =>
    `[sc] sc_trace: 137s conv  24 turn 0: 3072 tokens (max_new_tokens)   total ${n.toLocaleString('en-US')}`;

  it('climbs while sc_trace is still running', () => {
    const seen = feed([
      '[sc] sc_trace ...',
      beat(9359), beat(98741), beat(250692), beat(450000),
    ]);
    const bars = seen.map((d) => d.overall);
    expect(new Set(bars).size).toBeGreaterThan(2);
    expect(bars[bars.length - 1]).toBeGreaterThan(bars[0]);
  });

  it('reads the real counts from the measured run', () => {
    expect(scTraceFrac(beat(9359))).toBeCloseTo(9359 / SC_TRACE_TOKENS, 3);
    expect(scTraceFrac(beat(250692))).toBeCloseTo(250692 / SC_TRACE_TOKENS, 3);
  });

  it('clamps the wave overshoot instead of exceeding 100%', () => {
    // The stop is checked at wave boundaries, so the last wave runs past the
    // target: 798,117 generated against a 512,000 target, observed.
    expect(scTraceFrac(beat(798117))).toBe(1);
  });

  it('falls back to stage completion when the text does not parse', () => {
    // Upstream's wording moves with every EXLLAMAV3_REF bump; a miss must not
    // invent a number.
    expect(scTraceFrac('-- Loading /quant/blockquant/donor')).toBeNull();
    const [d] = feed(['[sc] sc_trace ...', '[sc] sc_trace: 12s -- Loading /quant/donor']);
    expect(d.stage).toBe('Calibrating');
  });

  it('never moves backward on a reprinted window', () => {
    // The controller reprints its tail, so an older, smaller count can arrive
    // after a newer one.
    const seen = feed(['[sc] sc_trace ...', beat(250692), beat(9359)]);
    const bars = seen.map((d) => d.overall);
    expect([...bars].sort((a, b) => a - b)).toEqual(bars);
  });

  it('a finished stage outranks any progress inside it', () => {
    const done = scProgress(new Set(['sc_trace']));
    const running = scProgress(new Set(), 'sc_trace', 0.99);
    expect(done).toBeGreaterThan(running);
  });

  it('gives no credit to a stage that reports nothing', () => {
    expect(scProgress(new Set(), 'sc_measure', 0)).toBe(0);
  });
});


describe('the other long stage', () => {
  // sc_measure ran 16.3 min of the 42 measured and credited nothing until it
  // finished -- a 39%-to-55% jump on the replayed log. It walks modules inside
  // passes and says so.
  const beat = (p, P, x, M) =>
    `[sc] sc_measure: 240s -- [pass ${p}/${P}] module ${x}/${M}: 3 pending, 2 new tensor(s)`;

  it('reads pass and module out of the line', () => {
    expect(scMeasureFrac(beat(1, 3, 12, 24))).toBeCloseTo((0 + 12 / 24) / 3, 3);
    expect(scMeasureFrac(beat(2, 3, 24, 24))).toBeCloseTo((1 + 1) / 3, 3);
    expect(scMeasureFrac(beat(3, 3, 24, 24))).toBe(1);
  });

  it('climbs while sc_measure is still running', () => {
    const seen = feed([
      '[sc] sc_trace done -> cal.safetensors',
      '[sc] sc_rfn_probe done -> rfn.json',
      '[sc] sc_measure ...',
      beat(1, 3, 1, 24), beat(1, 3, 12, 24), beat(2, 3, 6, 24), beat(3, 3, 20, 24),
    ]);
    const bars = seen.map((d) => d.overall);
    expect(new Set(bars).size).toBeGreaterThan(3);
    expect([...bars].sort((a, b) => a - b)).toEqual(bars);
  });

  it('ignores its own prose', () => {
    // The reference pass and the summary tables must not be mistaken for it.
    expect(scMeasureFrac('-- Reference pass')).toBeNull();
    expect(scMeasureFrac('-- Sum of per-tensor KLD at level 0: 0.0421')).toBeNull();
    expect(scMeasureFrac('-- Resuming: 50 tensors already measured')).toBeNull();
  });

  it('does not read a trace line as a measure line', () => {
    expect(scMeasureFrac('conv  24 turn 0: 3072 tokens   total   68,304')).toBeNull();
    expect(scTraceFrac('-- [pass 1/3] module 12/24: 3 pending')).toBeNull();
  });
});

describe('the startup band', () => {
  it('steps through the controller phases instead of one number', () => {
    // Renting, SSH and the image pull is ~5 min and used to sit at 4%.
    const seen = feed([
      '[1/6] Trying NVIDIA L40S  (~$1.09/hr SECURE)...',
      '[2/6] Waiting for SSH (up to 10 min)...',
      '[3/6] Bootstrapping (PyTorch, transformers, exllamav3, flash-attn)...',
    ]);
    const bars = seen.map((d) => d.overall);
    expect(new Set(bars).size).toBeGreaterThan(1);
    expect([...bars].sort((a, b) => a - b)).toEqual(bars);
    expect(seen.every((d) => d.stage === 'Provisioning')).toBe(true);
  });

  it('keeps the message moving through a long upload', () => {
    // upload_folder reports no percent, so the bar holds at 95 -- the elapsed
    // keepalive is all there is, and nothing used to match it.
    const [d] = feed(['[upload] 3.0 pushing... 120s']);
    expect(d.stage).toBe('Uploading');
    expect(d.message).toContain('2m');
  });
});
