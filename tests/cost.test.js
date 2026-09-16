import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { estimateCost, scCalibrationCost, convertCost, maxPricePerHour } =
  await import('../src/services/runpod.js');

describe('what the requester is quoted', () => {
  it('charges a plain quant per variant', () => {
    // Independent pods, independent work, so this really does multiply.
    const one = estimateCost(1, { sizeGb: 15.3 });
    const three = estimateCost(3, { sizeGb: 15.3 });
    expect(three.low).toBeCloseTo(3 * one.low, 6);
    expect(three.sc).toBe(false);
  });

  it('scales a plain quant with the model, which it never used to', () => {
    // A 0.8B and a 70B were quoted the same flat $1.50-3.00.
    const small = estimateCost(1, { sizeGb: 1.63 });
    const big = estimateCost(1, { sizeGb: 72 });
    expect(big.low).toBeGreaterThan(small.low * 5);
  });

  it('lands near the one conversion baseline the repo measured', () => {
    // run_runpod_job's ETA uses 3.7 h for a 35B at cal_rows 250; at that
    // size's $1.80/hr cap the band has to contain the resulting ~$6.7.
    const c = convertCost(72);
    expect(c.low).toBeLessThan(6.7);
    expect(c.high).toBeGreaterThan(6.7);
  });

  it('mirrors the launcher price tiers', () => {
    // Drift here makes every estimate wrong in the same direction.
    expect(maxPricePerHour(15)).toBe(0.80);
    expect(maxPricePerHour(40)).toBe(1.30);
    expect(maxPricePerHour(72)).toBe(1.80);
    expect(maxPricePerHour(200)).toBe(2.80);
    // SC floors the cap so a faster card is reachable at all.
    expect(maxPricePerHour(15, true)).toBe(1.30);
  });

  it('charges self-calibration once, not once per bitrate', () => {
    // The trace, probe and measure are per MODEL, and an SC job now runs every
    // variant on one pod. Multiplying them by variant count would overstate a
    // 3-bitrate job by two whole calibrations.
    const one = estimateCost(1, { sc: true, sizeGb: 1.63 });
    const three = estimateCost(3, { sc: true, sizeGb: 1.63 });
    const cal = scCalibrationCost(1.63);
    const per = convertCost(1.63, true);
    // The extra bitrates add conversions only -- no second calibration.
    expect(three.low - one.low).toBeCloseTo(2 * per.low, 6);
    expect(three.low).toBeCloseTo(3 * per.low + cal.low, 6);
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
      expect(scCalibrationCost(gb).high).toBeLessThanOrEqual(8 * maxPricePerHour(gb, true));
    }
  });

  it('still quotes something when the size is unknown', () => {
    // preflight can fail to read it; a missing size must not zero the band.
    const e = estimateCost(1, { sc: true, sizeGb: 0 });
    expect(e.low).toBeGreaterThan(0);
    expect(e.high).toBeGreaterThan(e.low);
    // And it must still cost more than the same job without calibration.
    expect(e.low).toBeGreaterThan(estimateCost(1, { sizeGb: 0 }).low);
  });

  it('caps by the tier the model will actually run on', () => {
    // The ceiling used the top tier's $2.80 for every model, overstating it
    // for anything under 100 GB -- a 52 GB model runs under a $1.80 cap.
    expect(scCalibrationCost(52).high).toBeLessThanOrEqual(8 * maxPricePerHour(52, true));
    expect(scCalibrationCost(500).high).toBeLessThanOrEqual(8 * 2.80);
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
