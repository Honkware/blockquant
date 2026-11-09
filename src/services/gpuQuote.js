// The quote, off the card the launcher will actually rent.
//
// The old band multiplied a per-size price CAP by 0.66-1.0, because the bot had
// no idea which card a job would land on. It can know: the launcher ranks
// cards by static price over conversion speed and tries each on community then
// secure under the cap, all of which is in backend/gpu_table.json and RunPod's
// live listing. So this runs the same selection against live prices and stock
// and quotes the first card that would be taken, with the next as the
// fallback. Mirrors _auto_gpu_ids, _kl_min_vram, _sc_min_vram and the cloud
// sweep in backend/scripts/run_runpod_job.py -- change one, change both.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { maxPricePerHour, calibrationHours, estimateCost, getBalance } from './runpod.js';

const TABLE = JSON.parse(fs.readFileSync(
  path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'backend', 'gpu_table.json'), 'utf8'));
const UNKNOWN_PRICE = 9.99;
const GRAPHQL = 'https://api.runpod.io/graphql';
const MIN_VRAM = 24;

const staticPrice = (id) => TABLE.cards[id]?.price ?? UNKNOWN_PRICE;
export const convertSpeed = (id) => TABLE.cards[id]?.speed ?? 1.0;

function excluded(id) {
  const ex = TABLE.exclude;
  return id.includes('MIG') || id === 'unknown'
    || [...ex.no_kernels, ...ex.non_cuda].some((t) => id.includes(t))
    || ex.weak.includes(id);
}

export function vramFloor({ sizeGb = 0, sc = false, kl = true, gpus = 1 }) {
  if (!sizeGb) return MIN_VRAM;
  if (sc) return Math.max(MIN_VRAM, Math.floor(sizeGb * 1.25) + 1);
  if (!kl) return MIN_VRAM;
  const need = Math.ceil((sizeGb * 1.1 + 3) / Math.max(1, gpus));
  return need <= 96 ? Math.max(MIN_VRAM, need) : MIN_VRAM;
}

/**
 * Cards in the order the launcher would try them, each with the cloud and live
 * rate it would rent at. `inStock` is RunPod's own stock flag; a card it lists
 * with none is kept, last, since the sweep tries it anyway.
 */
// minVram is the launcher's --min-vram: 24, or more on a retry after a job ran
// out of memory on a smaller card.
export function planPods(gpuTypes, { sizeGb = 0, sc = false, kl = true, gpus = 1, minVram = MIN_VRAM } = {}) {
  const n = Math.max(1, gpus || 1);
  const cap = maxPricePerHour(sizeGb, sc) * n;
  const usable = (floor) => (gpuTypes || []).filter((g) =>
    g?.id && !excluded(g.id) && (g.memoryInGb || 0) >= floor);
  let cards = usable(Math.max(minVram, vramFloor({ sizeGb, sc, kl, gpus: n })));
  // The launcher drops back to the plain floor when nothing that big exists.
  if (!cards.length) cards = usable(minVram);
  cards.sort((a, b) => staticPrice(a.id) / convertSpeed(a.id) - staticPrice(b.id) / convertSpeed(b.id)
    || (b.memoryInGb || 0) - (a.memoryInGb || 0));
  const out = [];
  for (const g of cards) {
    for (const [cloud, price] of [['community', g.communityPrice], ['secure', g.securePrice]]) {
      if (!price || price * n > cap) continue;
      out.push({ id: g.id, name: g.displayName || g.id, cloud, rate: price * n, speed: convertSpeed(g.id),
                 inStock: !!g.lowestPrice?.stockStatus, memoryInGb: g.memoryInGb });
    }
  }
  // Stable: in-stock first, launcher order within each.
  return [...out.filter((p) => p.inStock), ...out.filter((p) => !p.inStock)];
}

/** Hours one conversion takes on `pod`, download and upload included. */
export function convertHoursOn(pod, { sizeGb = 0, isMoe = false, gpus = 1, kl = true } = {}) {
  const gb = Number(sizeGb) || 0;
  const scale = Math.max(1, gpus || 1) ** TABLE.gpu_scaling;
  // An MoE takes about 3.45x what its size says, on whatever card: two jobs
  // on two cards agree (moe_note in gpu_table.json).
  const convert = (gb * TABLE.base_min_per_gb) / 60 / (pod?.speed || 1) / scale
    * (isMoe ? TABLE.moe_factor : 1);
  return {
    // ~4 min to boot, ~1 GB/s down, an upload in under a minute: the logs.
    overhead: 0.08 + gb / 3600,
    convert,
    convertHigh: convert * 1.15,
    kl: kl ? 0.03 : 0,
  };
}

/**
 * Whether the launcher converts on the cheapest card and rents one that holds
 * the fp16 only for the KL eval, and that card's VRAM floor. Mirrors the split
 * in run_runpod_job.py's main and _kl_split_saving: static prices, cards ranked
 * without the cap, one pod per bitrate.
 */
export function klSplit(gpuTypes, { sizeGb = 0, isMoe = false, sc = false, kl = true, gpus = 1,
  split = true, minVram = MIN_VRAM } = {}) {
  if (!split || sc || !kl || gpus !== 1 || !sizeGb) return null;
  const floor = vramFloor({ sizeGb });
  if (floor <= minVram) return null;
  const ranked = (min) => (gpuTypes || [])
    .filter((g) => g?.id && !excluded(g.id) && (g.memoryInGb || 0) >= min)
    .sort((a, b) => staticPrice(a.id) / convertSpeed(a.id) - staticPrice(b.id) / convertSpeed(b.id)
      || (b.memoryInGb || 0) - (a.memoryInGb || 0));
  const [big] = ranked(floor);
  const [small] = ranked(minVram);
  if (!big || !small || big.id === small.id) return null;
  const b = convertHoursOn({ speed: convertSpeed(big.id) }, { sizeGb, isMoe });
  const s = convertHoursOn({ speed: convertSpeed(small.id) }, { sizeGb, isMoe, kl: false });
  const whole = (b.overhead + b.convert + b.kl) * staticPrice(big.id);
  const twoPods = (s.overhead + s.convert) * staticPrice(small.id) + (b.overhead + b.kl) * staticPrice(big.id);
  return whole - twoPods >= TABLE.kl_split_min_saving ? { floor, saving: whole - twoPods } : null;
}

// How far down the sweep the high end looks. RunPod's stock flag is per card,
// not per pool, and "Low" often means none: the top card and its sibling SKU
// can both be gone, and then the job lands two or three places down.
const FALLBACK_DEPTH = 4;

/**
 * Low and high for the whole job, and the cards behind them. Low is the first
 * card the launcher would take, at the fast end of every stage. High is the
 * dearest of the next few it would fall back to, at the slow end.
 */
export function quoteJob(gpuTypes, {
  variants = 1, sizeGb = 0, isMoe = false, sc = false, scLite = false,
  needsDonor = false, gpus = 1, kl = true, split = true, minVram = MIN_VRAM,
} = {}) {
  const n = Math.max(0, variants || 0);
  // A split job converts with KL off and pays a second, short pod for it.
  const splitKl = klSplit(gpuTypes, { sizeGb, isMoe, sc, kl, gpus, split, minVram });
  const pods = planPods(gpuTypes, { sizeGb, sc, kl: kl && !splitKl, gpus, minVram });
  if (!pods.length) return null;
  const best = pods[0];
  const opts = { sizeGb, isMoe, gpus, kl: kl && !splitKl };
  const overhead = (pod) => convertHoursOn(pod, opts).overhead;
  const perVariant = (pod, high) => {
    const h = convertHoursOn(pod, opts);
    return (high ? h.convertHigh : h.convert) + h.kl;
  };
  const cal = calibrationHours(sizeGb, { traceOnly: scLite });
  // SC is one pod that calibrates once and then converts every bitrate; plain
  // is a pod per bitrate, each paying its own boot and download.
  const hours = (pod, high) => (sc
    ? overhead(pod) + (high ? cal.high : cal.low) + n * perVariant(pod, high)
    : n * (overhead(pod) + perVariant(pod, high)));
  const lowH = hours(best, false);
  let low = lowH * best.rate;
  const worst = pods.slice(0, FALLBACK_DEPTH)
    .map((pod) => ({ pod, cost: hours(pod, true) * pod.rate }))
    .reduce((a, b) => (b.cost > a.cost ? b : a));
  const next = worst.pod;
  const highH = hours(next, true);
  let high = worst.cost;
  let donor = null;
  if (sc && needsDonor) {
    const dp = planPods(gpuTypes, { sizeGb, sc: false, kl, gpus });
    if (dp.length) {
      const plainLeg = (pod, high) => overhead(pod) + perVariant(pod, high);
      const dHigh = Math.max(...dp.slice(0, FALLBACK_DEPTH).map((pod) => plainLeg(pod, true) * pod.rate));
      donor = { low: plainLeg(dp[0], false) * dp[0].rate, high: dHigh, pod: dp[0] };
      low += donor.low;
      high += donor.high;
    }
  }
  let klPod = null;
  if (splitKl) {
    // Its own boot and fp16 download, then the eval.
    const kp = planPods(gpuTypes, { sizeGb, kl: true, gpus });
    const leg = (pod) => n * (convertHoursOn(pod, { sizeGb, gpus }).overhead + 0.03) * pod.rate;
    if (kp.length) {
      klPod = { pod: kp[0], low: leg(kp[0]),
                high: Math.max(...kp.slice(0, FALLBACK_DEPTH).map(leg)) };
      low += klPod.low;
      high += klPod.high;
    }
  }
  return { low, high: Math.max(high, low), best, next: next === best ? null : next,
           hours: { low: lowH, high: highH }, donor, klPod };
}

let cache = { at: 0, data: null };

/** RunPod's live listing, cached for ten minutes. null when it cannot be read. */
export async function liveGpuTypes(gpus = 1) {
  if (cache.data && Date.now() - cache.at < 600_000 && cache.gpus === gpus) return cache.data;
  const key = process.env.RUNPOD_API_KEY;
  if (!key) return null;
  try {
    const resp = await fetch(GRAPHQL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body: JSON.stringify({
        query: `query { gpuTypes { id displayName memoryInGb securePrice communityPrice
                lowestPrice(input:{gpuCount:${Math.max(1, gpus | 0)}}){ stockStatus } } }`,
      }),
      signal: AbortSignal.timeout(10_000),
    });
    if (!resp.ok) return null;
    const data = (await resp.json())?.data?.gpuTypes;
    if (!Array.isArray(data)) return null;
    cache = { at: Date.now(), data, gpus };
    return data;
  } catch {
    return null;
  }
}

/**
 * The quote: off the card the launcher would rent at today's live price when
 * RunPod's listing can be read, off the size-based band when it cannot.
 */
export async function quoteCost(variantCount, opts = {}) {
  const live = await liveGpuTypes(opts.gpuCount || 1);
  // The launcher's switches, read the way it reads them.
  const on = (v) => !['0', 'false', 'no', ''].includes(String(v ?? '1').trim().toLowerCase());
  const q = live && quoteJob(live, {
    kl: on(process.env.BLOCKQUANT_KL_EVAL), split: on(process.env.BLOCKQUANT_KL_SPLIT),
    ...opts, variants: variantCount, gpus: opts.gpuCount || 1,
  });
  if (q) return { ...q, sc: !!opts.sc, donor: !!opts.needsDonor, live: true };
  return { ...estimateCost(variantCount, opts), live: false };
}

/** One-line preflight string for the approval embed. */
export async function costPreflightLine(variantCount, opts = {}) {
  const [bal, est] = await Promise.all([getBalance(), quoteCost(variantCount, opts)]);
  // Whole dollars hide the difference between $0.40 and $1.40 on a small job,
  // which is most of what a self-calibrated 0.8B costs.
  const fmt = (v) => (v < 10 ? v.toFixed(2) : v.toFixed(0));
  const parts = est.sc ? [est.donor ? 'donor + calibration' : 'calibration'] : [];
  if (est.klPod) parts.push(`KL on ${est.klPod.pod.name} after`);
  const incl = parts.length ? ` (incl. ${parts.join(', ')})` : '';
  const mins = (h) => (h >= 2 ? `${h.toFixed(1)} h` : `${Math.round(h * 60)} min`);
  // Say which card, because that is what the number rests on: a 27B is a
  // dollar on a Blackwell and three and a half on an A100 if the Blackwells
  // are taken, and a bare range does not say which way it went.
  const costStr = est.live
    ? `~$${fmt(est.low)}${incl} on ${est.best.name} (${est.best.cloud}, $${est.best.rate.toFixed(2)}/h, `
      + `~${mins(est.hours.low)})`
      + (est.high > est.low * 1.15
        ? `; up to $${fmt(est.high)} if it falls back to ${est.next ? `${est.next.name} (${est.next.cloud})` : 'a slower run'}`
        : '')
    : `~$${fmt(est.low)}-${fmt(est.high)}${incl}`;
  if (!bal) {
    return `**Est. RunPod cost:** ${costStr} (balance unavailable)`;
  }
  const warn = bal.balance < est.high ? '  ⚠️ may exceed balance' : '';
  return `**RunPod:** $${bal.balance.toFixed(2)} balance  ·  est. cost ${costStr}${warn}`;
}
