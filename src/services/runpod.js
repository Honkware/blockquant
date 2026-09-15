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

// Rough per-variant cost band for an EXL3 quant on the cheap auto-selected
// GPUs (~$0.16-0.69/hr, ~2.5-3 h each). Deliberately conservative so the
// admin sees a realistic ceiling, not a best case.
const COST_PER_VARIANT_LOW = 1.5;
const COST_PER_VARIANT_HIGH = 3.0;

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
const MAX_PRICE_PER_HOUR = 2.80;
const MAX_POD_COST = MAX_RUNTIME_H * MAX_PRICE_PER_HOUR;

/** The one-off calibration cost band for a self-calibrated job. */
export function scCalibrationCost(sizeGb) {
  const gb = Number(sizeGb) > 0 ? Number(sizeGb) : 0;
  return {
    low: Math.min(MAX_POD_COST, Math.max(SC_CAL_FLOOR_LOW, gb * SC_CAL_LOW_PER_GB)),
    high: Math.min(MAX_POD_COST, Math.max(SC_CAL_FLOOR_HIGH, gb * SC_CAL_HIGH_PER_GB)),
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
  const low = n * COST_PER_VARIANT_LOW;
  const high = n * COST_PER_VARIANT_HIGH;
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
