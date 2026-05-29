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
export function estimateCost(variantCount) {
  const n = Math.max(0, variantCount || 0);
  return { low: n * COST_PER_VARIANT_LOW, high: n * COST_PER_VARIANT_HIGH };
}

/** One-line preflight string for the approval embed; '' if balance unknown. */
export async function costPreflightLine(variantCount) {
  const bal = await getBalance();
  const est = estimateCost(variantCount);
  const costStr = `~$${est.low.toFixed(0)}-${est.high.toFixed(0)}`;
  if (!bal) {
    return `**Est. RunPod cost:** ${costStr} (balance unavailable)`;
  }
  const lowBalance = bal.balance < est.high;
  const warn = lowBalance ? '  ⚠️ may exceed balance' : '';
  return `**RunPod:** $${bal.balance.toFixed(2)} balance  ·  est. cost ${costStr}${warn}`;
}
