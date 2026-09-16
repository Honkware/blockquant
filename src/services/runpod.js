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

// The conversion itself, anchored on the one baseline the repo actually
// measured: ~3.7 h for a 35B (~72 GB) at cal_rows 250, the same figure
// run_runpod_job's ETA uses. Conversion is roughly linear in weights, so
// scale from there rather than quoting one number for every model -- a 0.8B
// and a 70B were being given the same $1.50-3.00.
const CONVERT_H_AT_72GB = 3.7;
const CONVERT_REF_GB = 72;
// Renting, image pull and the download, before a single layer is quantized.
const OVERHEAD_H_LOW = 0.25;
const OVERHEAD_H_HIGH = 0.40;

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
  // streaming path unless the fp16 weights fit resident. $2.10 reaches an
  // H100 80GB or an H200 NVL, which do both.
  if (!sc) return cap;
  return !gb || gb <= 20 ? Math.max(1.30, cap) : Math.max(2.10, cap);
}

/** Hours and dollars for converting one variant of a model this size. */
export function convertCost(sizeGb, sc = false) {
  const gb = Number(sizeGb) > 0 ? Number(sizeGb) : 0;
  const h = CONVERT_H_AT_72GB * (gb / CONVERT_REF_GB);
  const cap = maxPricePerHour(gb, sc);
  // A plain small model goes cheapest-first and a big one capable-first, so
  // the rate lands somewhere between half the cap and the cap itself.
  return {
    low: (OVERHEAD_H_LOW + h * 0.85) * (cap * 0.5),
    high: (OVERHEAD_H_HIGH + h * 1.30) * cap,
  };
}

// Self-calibration, paid ONCE per job: sc_trace, sc_rfn_probe and sc_measure
// depend on the model rather than the bitrate, and an SC job now runs every
// variant on one pod, so this does not multiply by variant count. Only the
// conversions do.
//
// One measured run: Qwen3.5-0.8B (1.6 GB) took 42 minutes of calibration on an
// L40S at $1.09/hr, about $0.77. How that grows is NOT known -- the two long
// stages are bound by different things (sc_trace by GPU bandwidth through the
// donor, sc_measure by CPU), so a single scale factor cannot describe both,
// and one point across a 40x size range is not a curve. So this is a
// deliberately wide band anchored on that measurement and flagged as an
// estimate. quant.py now records per-stage durations in the job result; fit
// this from those once a few SC jobs have run, and delete the apology.
const SC_CAL_LOW_PER_GB = 0.35;
const SC_CAL_HIGH_PER_GB = 1.60;
const SC_CAL_FLOOR_LOW = 0.75;
const SC_CAL_FLOOR_HIGH = 2.50;

// A pod cannot outlive max_runtime (8h in poll.py) -- the controller kills it
// -- and --max-price auto caps a big model's card at $2.80/hr. So no single
// pod can bill past this no matter what the size scaling says, and quoting a
// number the system would never let happen is its own kind of wrong. Extrapo-
// lating 1.6 GB to 70 GB put the ceiling at $124, or 57 hours.
const MAX_RUNTIME_H = 8;

/** The one-off calibration cost band for a self-calibrated job. */
export function scCalibrationCost(sizeGb) {
  const gb = Number(sizeGb) > 0 ? Number(sizeGb) : 0;
  // A pod cannot outlive max_runtime (8h in poll.py) at a rate above its own
  // size tier's cap, so nothing can bill past this however the size scaling
  // extrapolates -- and quoting a figure the system would never permit is its
  // own kind of wrong. This used to use the top tier's $2.80 for every model,
  // which overstated the ceiling for anything under 100 GB.
  const ceiling = MAX_RUNTIME_H * maxPricePerHour(gb, true);
  return {
    low: Math.min(ceiling, Math.max(SC_CAL_FLOOR_LOW, gb * SC_CAL_LOW_PER_GB)),
    high: Math.min(ceiling, Math.max(SC_CAL_FLOOR_HIGH, gb * SC_CAL_HIGH_PER_GB)),
  };
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
export function estimateCost(variantCount, { sc = false, sizeGb = 0 } = {}) {
  const n = Math.max(0, variantCount || 0);
  const per = convertCost(sizeGb, sc);
  const low = n * per.low;
  const high = n * per.high;
  if (!sc) return { low, high, sc: false };
  // Once for the job, not once per bitrate.
  const cal = scCalibrationCost(sizeGb);
  return { low: low + cal.low, high: high + cal.high, sc: true, calibration: cal };
}

/** One-line preflight string for the approval embed; '' if balance unknown. */
export async function costPreflightLine(variantCount, opts = {}) {
  const bal = await getBalance();
  const est = estimateCost(variantCount, opts);
  // Whole dollars hide the difference between $0.40 and $1.40 on a small job,
  // which is most of what a self-calibrated 0.8B costs.
  const fmt = (v) => (v < 10 ? v.toFixed(2) : v.toFixed(0));
  const costStr = `~$${fmt(est.low)}-${fmt(est.high)}${est.sc ? ' (incl. calibration)' : ''}`;
  if (!bal) {
    return `**Est. RunPod cost:** ${costStr} (balance unavailable)`;
  }
  const lowBalance = bal.balance < est.high;
  const warn = lowBalance ? '  ⚠️ may exceed balance' : '';
  return `**RunPod:** $${bal.balance.toFixed(2)} balance  ·  est. cost ${costStr}${warn}`;
}
