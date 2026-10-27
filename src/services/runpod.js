import { getLogger } from '../logger.js';

const log = getLogger('runpod');

const GRAPHQL = 'https://api.runpod.io/graphql';
const REST_PODS = 'https://rest.runpod.io/v1/pods';

/** List all pods via the REST API. Best-effort: [] on any failure. */
export async function listPods() {
  const key = process.env.RUNPOD_API_KEY;
  if (!key) return [];
  try {
    const resp = await fetch(REST_PODS, { headers: { Authorization: `Bearer ${key}` } });
    if (!resp.ok) return [];
    const data = await resp.json();
    return Array.isArray(data) ? data : data.pods || [];
  } catch (err) {
    log.debug(`listPods failed: ${err.message}`);
    return [];
  }
}

/** Terminate a pod via the REST API (Bearer; the GraphQL path 403s on rpa_ keys). */
export async function terminatePod(id) {
  const key = process.env.RUNPOD_API_KEY;
  if (!key) return false;
  try {
    const resp = await fetch(`${REST_PODS}/${id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${key}` },
    });
    return resp.ok;
  } catch (err) {
    log.debug(`terminatePod ${id} failed: ${err.message}`);
    return false;
  }
}

// Everything below is calibrated on measured runs rather than guessed, and
// quotes a band from community rates to secure ones because a job can land in
// either pool.
//
//   conversion   an 8B (15.3 GB) took 31.0 min on one RTX 3090
//   sc_trace     Swift-Qwen3.8-27b (51.75 GB), 4.0bpw donor: 132.7 min on
//                2x A100 SXM with -tp, ~4,800 tok/min
//   sc_measure   that same job reached module 44/67 in 42.5 min, i.e. ~0.5x
//                of sc_trace, in line with the 0.5-0.8x seen on a 0.8B
//
// The trace used to be scaled off a 1.1 GB donor on an L40S and halved at the
// low end for MTP drafting. The 27B came in at 2.2 h against a quoted
// 0.9-1.2 h: tensor parallel bought nothing over the one-card estimate, and
// MTP did not engage. So it is anchored on the 27B now, the size where the
// money is, and nothing assumes drafting helps.
const CONVERT_MIN_PER_GB = 31.0 / 15.26;      // measured, one 3090
const BIG_CARD_SPEEDUP = 1.8;                 // A100-class vs a 3090, convert
const CONVERT_H_AT_72GB = 3.7;                // the older 35B anchor, kept as the ceiling
const TRACE_H_REF = 132.7 / 60;               // measured, 27B
const TRACE_REF_GB = 51.75;

/** Mirrors _recommend_max_price in backend/scripts/run_runpod_job.py. */
export function maxPricePerHour(sizeGb, sc = false) {
  const gb = Number(sizeGb) || 0;
  let cap;
  if (!gb) cap = 1.5;
  else if (gb <= 20) cap = 0.80;
  else if (gb <= 50) cap = 1.30;
  else if (gb <= 100) cap = 1.80;
  else cap = 2.80;
  // Self-calibration is allowed more card than a plain convert of the same
  // model: sc_trace is bandwidth-bound and sc_measure drops to a CPU-bound
  // streaming path unless the fp16 weights fit resident.
  if (!sc) return cap;
  return !gb || gb <= 20 ? Math.max(1.30, cap) : Math.max(2.10, cap);
}

/** Community and secure rates for the pod a job this size will land on. */
export function rateBand(sizeGb, sc = false, gpus = 1) {
  // The cap is per GPU-hour and the launcher multiplies it by the pod's GPU
  // count. Quoting one card's rate put a 2x A100 job at half what it billed.
  const cap = maxPricePerHour(sizeGb, sc) * Math.max(1, gpus || 1);
  // Community runs 25-30% under secure on every card checked, and a job can
  // land in either pool, so the quote spans both rather than picking one.
  return { low: cap * 0.66, high: cap };
}

/** Hours to convert one variant of a model this size. */
export function convertHours(sizeGb, isMoe = false) {
  const gb = Number(sizeGb) > 0 ? Number(sizeGb) : 0;
  const measured = (gb * CONVERT_MIN_PER_GB) / 60 / (gb > 20 ? BIG_CARD_SPEEDUP : 1);
  // The 3.7h-at-72GB anchor is a 35B MoE on a community NVL. Our own timing --
  // 8B in 31 min on a 3090, a far slower card -- implies ~2.4h at that size,
  // and both cannot describe the same work: a MoE has many times the tensors
  // to quantize. So the anchor is the ceiling for MoE and irrelevant to a
  // dense model, where carrying it doubled the width of a term we have
  // measured directly.
  const anchor = CONVERT_H_AT_72GB * (gb / 72);
  const high = isMoe ? Math.max(measured, anchor) : measured * 1.6;
  // Renting, image pull and the download, before a single layer is quantized.
  return { low: 0.25 + measured, high: 0.4 + high };
}

/** Hours of self-calibration -- trace, probe, measure -- paid once per job. */
export function calibrationHours(sizeGb) {
  const gb = Number(sizeGb) > 0 ? Number(sizeGb) : 0;
  if (!gb) return { low: 0.75, high: 3.0 };
  // sc_trace is bandwidth-bound generation through the donor, so it goes with
  // the donor's size and that goes with the model's. The 27B's 132.7 min
  // includes loading; a tiny model is mostly loading.
  const traceH = Math.max(0.08, TRACE_H_REF * (gb / TRACE_REF_GB));
  // Plus sc_measure at 0.5x and 0.8x of that; the probe is under a minute.
  return { low: traceH * 1.5, high: traceH * 1.8 };
}

/** The one-off calibration cost band for a self-calibrated job. */
export function scCalibrationCost(sizeGb, gpus = 1) {
  const h = calibrationHours(sizeGb);
  const r = rateBand(sizeGb, true, gpus);
  const ceiling = 8 * r.high;                          // a pod cannot outlive max_runtime
  return { low: Math.min(ceiling, h.low * r.low), high: Math.min(ceiling, h.high * r.high) };
}

/** Hours and dollars for converting one variant of a model this size. */
export function convertCost(sizeGb, sc = false, isMoe = false, gpus = 1) {
  // Hours do not shrink with gpus: the 27B donor converted in 57.7 min on two
  // A100s, about what one is estimated at. A second card is for VRAM.
  const h = convertHours(sizeGb, isMoe);
  const r = rateBand(sizeGb, sc, gpus);
  return { low: h.low * r.low, high: h.high * r.high };
}

/**
 * Fetch the RunPod credit balance + current burn rate. Best-effort: returns
 * null on any failure so a preflight can degrade gracefully rather than block.
 */
export async function getBalance() {
  const key = process.env.RUNPOD_API_KEY;
  if (!key) return null;
  try {
    const resp = await fetch(GRAPHQL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        query: 'query { myself { clientBalance currentSpendPerHr } }',
      }),
    });
    if (!resp.ok) {
      log.debug(`RunPod balance lookup HTTP ${resp.status}`);
      return null;
    }
    const data = await resp.json();
    const me = data?.data?.myself;
    if (!me || me.clientBalance == null) return null;
    return {
      balance: Number(me.clientBalance),
      spendPerHr: Number(me.currentSpendPerHr) || 0,
    };
  } catch (err) {
    log.debug(`RunPod balance lookup failed: ${err.message}`);
    return null;
  }
}

/** Conservative cost band for N variants, e.g. { low: 4.5, high: 9 }. */
export function estimateCost(variantCount,
                             { sc = false, sizeGb = 0, needsDonor = false, isMoe = false,
                               gpuCount = 1 } = {}) {
  const n = Math.max(0, variantCount || 0);
  const per = convertCost(sizeGb, sc, isMoe, gpuCount);
  let low = n * per.low;
  let high = n * per.high;
  if (!sc) return { low, high, sc: false };
  // Once for the job, not once per bitrate.
  const cal = scCalibrationCost(sizeGb, gpuCount);
  low += cal.low;
  high += cal.high;
  // The donor is another whole conversion, and the job builds it itself now,
  // so quoting without it would understate what the requester is agreeing to.
  if (needsDonor) {
    // On the same GPU count: the donor pod is launched with the job's options.
    const donor = convertCost(sizeGb, false, isMoe, gpuCount);
    low += donor.low;
    high += donor.high;
  }
  return { low, high, sc: true, calibration: cal, donor: needsDonor };
}

/** One-line preflight string for the approval embed; '' if balance unknown. */
export async function costPreflightLine(variantCount, opts = {}) {
  const bal = await getBalance();
  const est = estimateCost(variantCount, opts);
  // Whole dollars hide the difference between $0.40 and $1.40 on a small job,
  // which is most of what a self-calibrated 0.8B costs.
  const fmt = (v) => (v < 10 ? v.toFixed(2) : v.toFixed(0));
  const parts = est.sc ? [est.donor ? 'donor + calibration' : 'calibration'] : [];
  const costStr = `~$${fmt(est.low)}-${fmt(est.high)}${parts.length ? ` (incl. ${parts[0]})` : ''}`;
  if (!bal) {
    return `**Est. RunPod cost:** ${costStr} (balance unavailable)`;
  }
  const lowBalance = bal.balance < est.high;
  const warn = lowBalance ? '  ⚠️ may exceed balance' : '';
  return `**RunPod:** $${bal.balance.toFixed(2)} balance  ·  est. cost ${costStr}${warn}`;
}
