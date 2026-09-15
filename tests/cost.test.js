import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { estimateCost, scCalibrationCost } = await import('../src/services/runpod.js');

describe('what the requester is quoted', () => {
  it('leaves a plain quant alone', () => {
    // Per-variant band, unchanged: independent pods, independent work.
    expect(estimateCost(3)).toMatchObject({ low: 4.5, high: 9.0, sc: false });
  });

  it('charges self-calibration once, not once per bitrate', () => {
    // The trace, probe and measure are per MODEL, and an SC job now runs every
    // variant on one pod. Multiplying them by variant count would overstate a
    // 3-bitrate job by two whole calibrations.
    const one = estimateCost(1, { sc: true, sizeGb: 1.63 });
    const three = estimateCost(3, { sc: true, sizeGb: 1.63 });
    const cal = scCalibrationCost(1.63);
    expect(three.low - one.low).toBeCloseTo(2 * 1.5, 5);
    expect(three.low).toBeCloseTo(3 * 1.5 + cal.low, 5);
  });

  it('says more for SC than for the same job without it', () => {
    // It used to quote them identically, which is the actual defect: a 42-min
    // calibration was invisible in the number.
    const plain = estimateCost(1, { sizeGb: 1.63 });
    const sc = estimateCost(1, { sc: true, sizeGb: 1.63 });
    expect(sc.low).toBeGreaterThan(plain.low);
    expect(sc.high).toBeGreaterThan(plain.high);
  });

  it('scales with the model, having been given a size at last', () => {
    const small = estimateCost(1, { sc: true, sizeGb: 1.63 });
    const big = estimateCost(1, { sc: true, sizeGb: 30 });
    expect(big.high).toBeGreaterThan(small.high);
  });

  it('never quotes more than a pod is allowed to bill', () => {
    // max_runtime is 8h and --max-price auto tops out at $2.80/hr, so the
    // controller kills a pod long before a naive extrapolation's ceiling.
    // 70 GB off a 1.6 GB measurement reached $124, i.e. 57 hours.
    for (const gb of [70, 200, 1000]) {
      expect(scCalibrationCost(gb).high).toBeLessThanOrEqual(8 * 2.8);
    }
  });

  it('still quotes something when the size is unknown', () => {
    // preflight can fail to read it; a missing size must not zero the band.
    const e = estimateCost(1, { sc: true, sizeGb: 0 });
    expect(e.low).toBeGreaterThan(1.5);
    expect(e.high).toBeGreaterThan(3.0);
  });

  it('keeps low below high everywhere it is defined', () => {
    for (const gb of [0, 0.5, 1.63, 15, 70, 400]) {
      for (const n of [1, 2, 5]) {
        const e = estimateCost(n, { sc: true, sizeGb: gb });
        expect(e.low).toBeLessThan(e.high);
      }
    }
  });
});
