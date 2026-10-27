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
//   conversion   an 8B (15.3 GB) took 31.0 min on one RTX 3090, 17.9 on two;
//                Qwen3.8-27B ~95 min on one A100 SXM, 57.7 on two, and a
//                1905 s pod on an RTX PRO 6000 Blackwell
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
//
// That trace also ran one conversation at a time: on a model with recurrent
// state the generator's batch is the cache's state slots, which 1.5.0's
// sc_trace left at 1. 1.6.5 sets 32, which should make the trace on Qwen3.5/
// 3.8-family models a fraction of this. Nothing here assumes it until a run
// measures it -- every guess in this file so far has been wrong.
const CONVERT_MIN_PER_GB = 31.0 / 15.26;      // measured, one 3090
// Against that 3090. The launcher ranks cards by price over speed, so a big
// model lands on the fastest card per dollar when it is in stock (the RTX PRO
// 6000 Blackwell, 4.5x: ~25 of its 1905 s pod converting 57 GB) and falls back
// to an A100 (about 1x -- not the 1.8x this used to assume) when it is not.
// Small models land on 4090/L40S-class cards or a 3090.
const FAST_CARD = { big: 4.5, small: 2.0 };
const SLOW_CARD = { big: 0.95, small: 1.0 };
// Two 3090s converted 1.73x faster than one; two A100s about 1.65x.
const GPU_SCALING = 0.79;                     // speedup = gpus ** this
const CONVERT_H_AT_72GB = 3.7;                // the older 35B anchor, kept as the ceiling
const TRACE_H_REF = 132.7 / 60;               // measured, 27B
const TRACE_REF_GB = 51.75;
const TRACE_REF_DONOR_BPW = 4.0;              // what that trace ran through
const DONOR_BPW = 6.0;                        // what a trace runs through now

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
export function convertHours(sizeGb, isMoe = false, gpus = 1) {
  const gb = Number(sizeGb) > 0 ? Number(sizeGb) : 0;
  const tier = gb > 20 ? 'big' : 'small';
  const scale = Math.max(1, gpus || 1) ** GPU_SCALING;
  const at = (speed) => (gb * CONVERT_MIN_PER_GB) / 60 / speed / scale;
  const fast = at(FAST_CARD[tier]);
  const measured = at(SLOW_CARD[tier]);
  // The 3.7h-at-72GB anchor is a 35B MoE on a community NVL. Our own timing --
  // 8B in 31 min on a 3090, a far slower card -- implies ~2.4h at that size,
  // and both cannot describe the same work: a MoE has many times the tensors
  // to quantize. So the anchor is the ceiling for MoE and irrelevant to a
  // dense model, where carrying it doubled the width of a term we have
  // measured directly.
  const anchor = CONVERT_H_AT_72GB * (gb / 72) / scale;
  // The slow end is already the slowest card measured (A100 PCIe, a 1.98 h
  // pod for a 27B), so it gets a tenth for variance, not a third.
  const high = isMoe ? Math.max(measured, anchor) : measured * 1.1;
  // Renting, image pull, the download and the upload: ~6 min in the logs when
  // nothing goes wrong, which is the low end.
  return { low: 0.1 + fast, high: 0.4 + high };
}

/** Hours of self-calibration -- trace, probe, measure -- paid once per job. */
export function calibrationHours(sizeGb, { donorBpw = DONOR_BPW, traceOnly = false } = {}) {
  const gb = Number(sizeGb) > 0 ? Number(sizeGb) : 0;
  if (!gb) return traceOnly ? { low: 0.3, high: 1.5 } : { low: 0.75, high: 3.0 };
  // sc_trace is generation through the donor, so it goes with the donor's
  // size: the model's, times the donor's bits over the measured one's. The
  // 27B's 132.7 min includes loading; a tiny model is mostly loading.
  const donorScale = Number(donorBpw) / TRACE_REF_DONOR_BPW;
  const traceH = Math.max(0.08, TRACE_H_REF * (gb / TRACE_REF_GB) * donorScale);
  // sc_mode:trace stops there. Otherwise sc_measure at 0.5x and 0.8x of the
  // trace; the probe is under a minute.
  if (traceOnly) return { low: traceH, high: traceH * 1.15 };
  return { low: traceH * 1.5, high: traceH * 1.8 };
}

/** The one-off calibration cost band for a self-calibrated job. */
export function scCalibrationCost(sizeGb, gpus = 1, opts = {}) {
  const h = calibrationHours(sizeGb, opts);
  const r = rateBand(sizeGb, true, gpus);
  const ceiling = 8 * r.high;                          // a pod cannot outlive max_runtime
  return { low: Math.min(ceiling, h.low * r.low), high: Math.min(ceiling, h.high * r.high) };
}

/** Hours and dollars for converting one variant of a model this size. */
export function convertCost(sizeGb, sc = false, isMoe = false, gpus = 1) {
  // More cards are faster but not proportionally: 1.73x for two 3090s, so a
  // second one costs about 15% more per job and saves 40% of the wait.
  const h = convertHours(sizeGb, isMoe, gpus);
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
                               gpuCount = 1, scLite = false, donorBpw = DONOR_BPW } = {}) {
  const n = Math.max(0, variantCount || 0);
  const per = convertCost(sizeGb, sc, isMoe, gpuCount);
  let low = n * per.low;
  let high = n * per.high;
  if (!sc) return { low, high, sc: false };
  // Once for the job, not once per bitrate.
  const cal = scCalibrationCost(sizeGb, gpuCount, { traceOnly: scLite, donorBpw });
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
