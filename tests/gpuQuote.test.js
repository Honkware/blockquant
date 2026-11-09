import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { planPods, quoteJob, vramFloor, convertHoursOn, klSplit } = await import('../src/services/gpuQuote.js');

// RunPod's listing as it stood on 2026-10-27, trimmed.
const card = (id, mem, secure, community, stock = 'High') =>
  ({ id, displayName: id.replace('NVIDIA ', ''), memoryInGb: mem, securePrice: secure,
     communityPrice: community, lowestPrice: { stockStatus: stock } });
const LIVE = [
  card('NVIDIA GeForce RTX 3090', 24, 0.50, 0.22, 'Low'),
  card('NVIDIA GeForce RTX 4090', 24, 0.89, 0.34, 'Low'),
  card('NVIDIA A40', 48, 0.59, 0.35),
  card('NVIDIA A100-SXM4-80GB', 80, 1.79, 1.39, 'Low'),
  card('NVIDIA RTX PRO 6000 Blackwell Server Edition', 96, 2.49, 1.69),
  card('NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb', 48, 1.29, 1.00),
  card('NVIDIA B200', 180, 7.99, 5.98, 'Low'),
  card('AMD Instinct MI300X OAM', 192, 2.39, 0.50, null),
];

describe('which pod the launcher would rent', () => {
  it('puts a 27B on the Blackwell, community first', () => {
    const pods = planPods(LIVE, { sizeGb: 55.6 });
    expect(pods[0]).toMatchObject({ id: 'NVIDIA RTX PRO 6000 Blackwell Server Edition', cloud: 'community', rate: 1.69 });
    // 24-48 GB cards cannot hold the fp16 for the KL eval.
    expect(pods.some((p) => p.memoryInGb < 64)).toBe(false);
  });

  it('never quotes a card the launcher excludes', () => {
    const ids = planPods(LIVE, { sizeGb: 1.8 }).map((p) => p.id);
    expect(ids.some((i) => /MIG|B200|MI300X/.test(i))).toBe(false);
  });

  it('respects the price cap per pod', () => {
    // An 8B is capped at $0.80/GPU, so the A100 and Blackwell are out.
    const pods = planPods(LIVE, { sizeGb: 16.4 });
    expect(pods.every((p) => p.rate <= 0.80)).toBe(true);
    expect(pods[0].id).toBe('NVIDIA GeForce RTX 4090');
  });

  it('holds an SC job to a card the fp16 fits on', () => {
    expect(vramFloor({ sizeGb: 55.6, sc: true })).toBeGreaterThan(64);
    expect(vramFloor({ sizeGb: 16.4 })).toBe(24);
  });
});

describe('the quote', () => {
  it('lands on the measured 27B Blackwell job', () => {
    // A Qwen3.8-27B on an RTX PRO 6000 Blackwell: a 1905 s pod. The low end is
    // that card at today's community rate.
    const q = quoteJob(LIVE, { variants: 1, sizeGb: 55.6 });
    expect(q.hours.low * 3600).toBeGreaterThan(1905 * 0.85);
    expect(q.hours.low * 3600).toBeLessThan(1905 * 1.25);
    expect(q.low).toBeCloseTo(q.hours.low * 1.69, 6);
  });

  it('prices the fallback when the first card is taken', () => {
    const q = quoteJob(LIVE, { variants: 1, sizeGb: 55.6 });
    expect(q.next.id).toBe('NVIDIA A100-SXM4-80GB');
    expect(q.high).toBeGreaterThan(q.low * 2);
  });

  it('puts the measured A100 job under the high end', () => {
    // 5748-6135 s on an A100 SXM; at the secure rate that is the top.
    const q = quoteJob(LIVE, { variants: 1, sizeGb: 55.6 });
    expect(q.high).toBeGreaterThan(6135 / 3600 * 1.39);
  });

  it('charges a pod per plain bitrate and one pod for SC', () => {
    const one = quoteJob(LIVE, { variants: 1, sizeGb: 16.4 });
    const three = quoteJob(LIVE, { variants: 3, sizeGb: 16.4 });
    expect(three.low).toBeCloseTo(3 * one.low, 6);
    const sc1 = quoteJob(LIVE, { variants: 1, sizeGb: 16.4, sc: true });
    const sc3 = quoteJob(LIVE, { variants: 3, sizeGb: 16.4, sc: true });
    expect(sc3.low - sc1.low).toBeLessThan(2 * one.low);   // no second boot or calibration
  });

  it('adds the donor when the job builds one', () => {
    const a = quoteJob(LIVE, { variants: 1, sizeGb: 55.6, sc: true });
    const b = quoteJob(LIVE, { variants: 1, sizeGb: 55.6, sc: true, needsDonor: true });
    expect(b.low).toBeGreaterThan(a.low);
  });

  it('gives up rather than guess when nothing is rentable', () => {
    expect(quoteJob([], { variants: 1, sizeGb: 16.4 })).toBe(null);
  });

  it('times a conversion off the card it runs on', () => {
    const slow = convertHoursOn({ speed: 1 }, { sizeGb: 16.4 });
    const fast = convertHoursOn({ speed: 2 }, { sizeGb: 16.4 });
    expect(slow.convert * 60).toBeCloseTo(31, 0);   // the 3090 run it is anchored on
    expect(fast.convert).toBeCloseTo(slow.convert / 2, 6);
  });
});

describe('a big MoE', () => {
  // NorthCUA-Mini: a 70.2 GB qwen3_5_moe that ran 6925 s on an RTX PRO 6000
  // for $4.98, 108.8 min of it converting.
  const moe = { variants: 1, sizeGb: 70.2, isMoe: true };

  it('times the conversion at what the job measured', () => {
    const h = convertHoursOn({ speed: 4.2 }, moe);
    expect(h.convert * 60).toBeGreaterThan(108.8 * 0.9);
    expect(h.convert * 60).toBeLessThan(108.8 * 1.1);
  });

  it('lands on the measured job when it runs on one pod', () => {
    const q = quoteJob(LIVE, { ...moe, split: false });
    expect(q.best.id).toBe('NVIDIA RTX PRO 6000 Blackwell Server Edition');
    expect(q.hours.low * 3600).toBeGreaterThan(6925 * 0.9);
    expect(q.hours.low * 3600).toBeLessThan(6925 * 1.1);
  });

  it('converts on the cheapest card and rents the Blackwell for KL', () => {
    expect(klSplit(LIVE, moe)).toMatchObject({ floor: 81 });
    const q = quoteJob(LIVE, moe);
    expect(q.best.memoryInGb).toBeLessThan(81);
    expect(q.klPod.pod.id).toBe('NVIDIA RTX PRO 6000 Blackwell Server Edition');
    // A boot, the download and the eval: minutes, not the conversion.
    expect(q.klPod.low).toBeLessThan(0.5);
    expect(q.low).toBeLessThan(quoteJob(LIVE, { ...moe, split: false }).low);
  });

  it('keeps a dense 27B on one pod', () => {
    expect(klSplit(LIVE, { sizeGb: 55.6 })).toBe(null);
    expect(quoteJob(LIVE, { variants: 1, sizeGb: 55.6 }).klPod).toBe(null);
  });

  it('never splits SC or a job with KL off', () => {
    expect(klSplit(LIVE, { ...moe, sc: true })).toBe(null);
    expect(klSplit(LIVE, { ...moe, kl: false })).toBe(null);
    expect(klSplit(LIVE, { ...moe, gpus: 2 })).toBe(null);
  });
});
