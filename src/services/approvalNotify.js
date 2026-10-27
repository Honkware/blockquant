// A request for approval was a card in #quant-bot and nothing else, so it sat
// until an admin happened to scroll past it. The antislop request waited the
// better part of a day; others waited longer and their requesters went and
// quantized it themselves.
import config from '../config.js';
import { getLogger } from '../logger.js';
import * as db from './db.js';

const log = getLogger('approval-notify');

const HOUR = 3_600_000;

export function cardUrl(job) {
  return job.channelId && job.cardMessageId
    ? `https://discord.com/channels/${config.GUILD_ID}/${job.channelId}/${job.cardMessageId}`
    : null;
}

/** Best-effort DM to every admin; one that has DMs closed is skipped. */
export async function notifyAdmins(client, text) {
  await Promise.all(config.ADMIN_IDS.map((id) =>
    client.users.fetch(id)
      .then((u) => u.send(text))
      .catch((err) => log.debug(`could not DM admin ${id}: ${err.message}`))));
}

/**
 * Requests still awaiting approval after `olderThan`, not reminded about in the
 * last `every`. A reminder per job per interval, not one per tick: the tick is
 * hourly and nobody wants that many DMs about one card.
 */
export function stalePending(jobs, now, { olderThan = 6 * HOUR, every = 12 * HOUR } = {}) {
  return Object.values(jobs ?? {}).filter((j) =>
    j.status === db.JOB_STATUS.pending_approval
    && now - (j.createdAt ?? now) >= olderThan
    && now - (j.remindedAt ?? 0) >= every);
}

export async function remindPending(client, { now = Date.now() } = {}) {
  const due = stalePending(await db.loadJobs(), now);
  if (!due.length) return 0;
  const lines = due.map((j) => {
    const h = Math.round((now - j.createdAt) / HOUR);
    const url = cardUrl(j);
    return `• \`${j.modelId}\` ${(j.variants || []).join(', ')}${j.sc ? ' SC' : ''} by ${j.username || j.userId}, `
      + `waiting ${h}h${url ? `: ${url}` : ''}`;
  });
  await notifyAdmins(client, `Still awaiting approval:\n${lines.join('\n')}`);
  for (const j of due) await db.patchJob(j.id, { remindedAt: now });
  log.info(`reminded admins about ${due.length} pending request(s)`);
  return due.length;
}
