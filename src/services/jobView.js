// What the bot is actually doing, for /queue, /history, /health and /diag.
//
// Those read the local job queue and model cache, which RunPod jobs never go
// through, so they reported an idle bot while pods ran. /history asked the API
// dashboard, which only the local pipeline feeds, and has said "No jobs yet!"
// for every job this bot has run. These read jobs.json and the controllers
// that are really alive.
import * as db from './db.js';

const S = db.JOB_STATUS;
const ICON = {
  [S.running]: '▶️', [S.queued]: '⏳', [S.pending_approval]: '🕓', [S.completed]: '✅',
  [S.failed]: '❌', [S.stopped]: '⏹️', [S.rejected]: '✖️', [S.interrupted]: '⚠️',
};
const FINISHED = [S.completed, S.failed, S.stopped, S.rejected, S.interrupted];

const when = (ms) => (ms ? `<t:${Math.floor(ms / 1000)}:R>` : '');

function describe(job) {
  const kind = job.sc ? (job.scLite ? ' · SCT' : ' · SC') : '';
  return `\`${job.modelId}\` ${(job.variants || []).join(', ')}${kind}`;
}

export function counts(jobs) {
  const out = {};
  for (const j of Object.values(jobs || {})) out[j.status ?? 'unknown'] = (out[j.status ?? 'unknown'] ?? 0) + 1;
  return out;
}

/** Running and waiting jobs, oldest first, as embed lines. */
export function activeLines(jobs) {
  const live = Object.values(jobs || {})
    .filter((j) => [S.running, S.queued, S.pending_approval].includes(j.status))
    .sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
  return live.map((j) => {
    const since = j.status === S.pending_approval ? `asked ${when(j.createdAt)}`
      : `started ${when(j.approvedAt ?? j.createdAt)}`;
    return `${ICON[j.status]} ${describe(j)} · <@${j.userId}> · ${since}`;
  });
}

/** The last `limit` jobs that ended, newest first, with what they cost. */
export function historyLines(jobs, limit = 8) {
  const ended = (j) => j.completedAt ?? j.failedAt ?? j.stoppedAt ?? j.rejectedAt ?? j.updatedAt ?? 0;
  return Object.values(jobs || {})
    .filter((j) => FINISHED.includes(j.status))
    .sort((a, b) => ended(b) - ended(a))
    .slice(0, limit)
    .map((j) => {
      const cost = j.actualCost?.cost != null ? ` · $${j.actualCost.cost.toFixed(2)}` : '';
      return `${ICON[j.status] ?? '•'} ${describe(j)}${cost} · ${when(ended(j))}`;
    });
}

/** Controllers still running on this box, from the detached registry. */
export function liveControllers(records) {
  return (records || []).filter((r) => r.running);
}
