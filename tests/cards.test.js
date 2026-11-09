import { describe, it, expect } from 'vitest';
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const cards = await import('../src/utils/cards.js');
const { jobOutcome, jobSpend, retryMinVram, whereNow, modelLine, optionsLine, quoteFacts } =
  await import('../src/commands/quant.js');

const job = { id: '5bbfec93-aaaa', modelId: 'org/Model-27B', userId: 'u1', variants: ['4.0'], createdAt: 1 };
const field = (e, name) => e.toJSON().fields.find((f) => f.name === name)?.value;
const names = (e) => e.toJSON().fields.map((f) => f.name);
const ok = (v, extra = {}) => ({ variant: v, bpw: v, pushed: true, url: `https://huggingface.co/o/m-${v}`, ...extra });
const bad = (v, error) => ({ variant: v, bpw: v, pushed: false, url: null, error });
const spend = { cost: 0.86, pods: [{ gpu: 'NVIDIA GeForce RTX 4090', cloud: 'SECURE', cost: 0.86 }] };

describe('the result card', () => {
  it('says what failed in one line, red, with what it cost', () => {
    const results = [bad('4.0', 'Out of system RAM at lm_head (49 GB pod)')];
    const e = cards.resultCard(job, { results, outcome: jobOutcome(['4.0'], results), spend,
                                      startedAt: 0, endedAt: 58 * 60_000 });
    const j = e.toJSON();
    expect(j.title).toBe('Failed to quantize Model-27B');
    expect(j.color).toBe(cards.COLOR.failed);
    expect(names(e)).toEqual(['Error', 'GPU', 'Time', 'Cost']);
    expect(field(e, 'Error')).toBe('Out of system RAM at lm_head (49 GB pod)');
    expect(field(e, 'GPU')).toBe('RTX 4090');
    expect(field(e, 'Time')).toBe('58 minutes');
    expect(field(e, 'Cost')).toBe('$0.86');
  });

  it('puts size, KL and the repo on one line per bitrate', () => {
    const results = [ok('4.0', { sizeGb: 15.9, kl: 0.0315, sample: 'Rain on the roof.' })];
    const e = cards.resultCard({ ...job }, { results, outcome: jobOutcome(['4.0'], results), spend,
                                             collectionUrl: 'https://huggingface.co/collections/x' });
    expect(e.toJSON().title).toBe('Quantized Model-27B');
    expect(e.toJSON().url).toBe('https://huggingface.co/collections/x');
    expect(field(e, '4.0 bpw')).toBe('15.9 GB, KL 0.0315, [repo](https://huggingface.co/o/m-4.0)\n> Rain on the roof.');
  });

  it('marks a partial job amber and keeps the failed bitrate in its place', () => {
    const v = ['4.0', '5.0'];
    const results = [ok('4.0'), bad('5.0', 'cli exit 1')];
    const e = cards.resultCard({ ...job, variants: v }, { results, outcome: jobOutcome(v, results) });
    expect(e.toJSON().title).toBe('Partly quantized Model-27B');
    expect(e.toJSON().color).toBe(cards.COLOR.partial);
    expect(field(e, '5.0 bpw')).toBe('Failed: The controller exited with code 1');
  });

  it('never shows raw log text or emoji', () => {
    const results = [bad('4.0', 'Out of GPU memory at layer 41 of 64 (24 GB card)')];
    const s = JSON.stringify(cards.resultCard(job, { results, outcome: jobOutcome(['4.0'], results) }).toJSON());
    expect(s).not.toMatch(/\p{Extended_Pictographic}/u);
  });
});

describe('the request card', () => {
  const j = { ...job, card: { model: '27.8B dense with vision, 55.6 GB', gpu: 'RTX PRO 6000', estimate: '$0.95 to $1.70', balance: 8.66 } };

  it('is purple while pending and lists only labelled facts', () => {
    const e = cards.requestCard(j);
    expect(e.toJSON().color).toBe(cards.COLOR.request);
    expect(e.toJSON().title).toBe('Quantize Model-27B');
    expect(names(e)).toEqual(['Requested by', 'Bitrate', 'Balance', 'Model', 'GPU', 'Estimate']);
    expect(field(e, 'Balance')).toBe('$8.66');
  });

  it('turns green on approval and grey on denial, same fields plus who decided', () => {
    const a = cards.requestCard(j, { state: 'approved', by: 'admin', at: 1_000 });
    expect(a.toJSON().color).toBe(cards.COLOR.done);
    expect(field(a, 'Approved by')).toBe('<@admin> <t:1:R>');
    const d = cards.requestCard(j, { state: 'denied', by: 'admin' });
    expect(d.toJSON().color).toBe(cards.COLOR.idle);
    expect(field(d, 'Denied by')).toBe('<@admin>');
  });
});

describe('the progress card', () => {
  it('gives each bitrate a bar and a plain status line', () => {
    const e = cards.progressCard(job, {
      startedAt: 0,
      rows: [
        { variant: '4.0', gpu: 'NVIDIA RTX PRO 6000 Blackwell Server Edition', stage: 'Quantizing', overall: 43,
          layer: 28, layers: 64, etaAt: 24 * 60_000 },
        { variant: '5.0', gpu: 'NVIDIA L40S', stage: 'Downloading', overall: 10, dl: { done: 31, total: 55.6 } },
      ],
      spent: 1.42,
      estimate: '$1.90 to $3.40',
    });
    expect(field(e, '4.0 bpw on RTX PRO 6000')).toBe('`█████████░░░░░░░░░░░` 43%\nLayer 28 of 64, done <t:1440:R>');
    expect(field(e, '5.0 bpw on L40S')).toMatch(/\nDownloading, 31\.0 of 55\.6 GB$/);
    expect(field(e, 'Cost so far')).toBe('$1.42');
  });
});

describe('small pieces', () => {
  it('shortens card names', () => {
    expect(cards.gpuName('NVIDIA RTX PRO 6000 Blackwell Server Edition')).toBe('RTX PRO 6000');
    expect(cards.gpuName('NVIDIA GeForce RTX 4090')).toBe('RTX 4090');
    expect(cards.gpuName('NVIDIA A100-SXM4-80GB')).toBe('A100-SXM4-80GB');
  });

  it('reads durations like a person would say them', () => {
    expect(cards.duration(20)).toBe('under a minute');
    expect(cards.duration(60)).toBe('1 minute');
    expect(cards.duration(72 * 60)).toBe('1 hour 12 minutes');
  });

  it('retries out-of-memory on the next size up, and nothing else', () => {
    expect(retryMinVram(['Out of GPU memory at layer 41 of 64 (24 GB card)'])).toBe(32);
    expect(retryMinVram(['Out of system RAM at lm_head (49 GB pod)'])).toBe(32);
    expect(retryMinVram(['Out of system RAM at lm_head'], 48)).toBe(80);
    expect(retryMinVram(['HfHubHTTPError: 403 Forbidden'], null)).toBe(null);
  });

  it('says where a stopped job had got to', () => {
    expect(whereNow([{ stage: 'Quantizing', layer: 28, layers: 64 }])).toBe('at layer 28 of 64');
    expect(whereNow([{ stage: 'Downloading' }])).toBe('while downloading');
  });

  it('builds the request facts', () => {
    expect(modelLine({ paramsB: 27.8, hasVision: true, sizeGb: 55.6 })).toBe('27.8B dense with vision, 55.6 GB');
    expect(modelLine({ isMoe: true, sizeGb: 70.2 })).toBe('MoE, 70.2 GB');
    expect(optionsLine({ variants: ['4.0'] })).toBe(null);
    expect(optionsLine({ sc: true, scLite: true, donorBpw: '6.0', headBits: 6, minVram: 48 }))
      .toBe('Self-calibration, trace only\nBuilds a 6.0 bpw donor first\nHead 6 bits\nAt least 48 GB of VRAM');
    expect(quoteFacts({ live: true, low: 0.95, high: 1.7, best: { name: 'RTX PRO 6000' } }))
      .toEqual({ gpu: 'RTX PRO 6000', estimate: '$0.95 to $1.70' });
    expect(quoteFacts({ live: false, low: 1, high: 1.05 }).estimate).toBe('$1.00');
  });

  it('spend still sums every pod', () => {
    expect(jobSpend([{ podCost: { pods: [{ gpu: 'a', cloud: 'SECURE', cost: 1 }] } }]).cost).toBe(1);
  });
});
