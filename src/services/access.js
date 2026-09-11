import config from '../config.js';
import * as db from './db.js';

export function isAdmin(userId) {
  return config.ADMIN_IDS.includes(userId);
}

/**
 * The role that skips admin approval, or null while none is set. It lives in
 * the settings store rather than .env because an admin sets it from Discord
 * with /config quanter-role, and .env would need a restart to take.
 */
export async function quanterRoleId() {
  const settings = await db.loadSettings().catch(() => ({}));
  return settings.quanterRoleId || null;
}

/**
 * Role ids off an interaction's member. That member is a GuildMember when
 * discord.js could build one (roles is a manager with a cache) and the raw
 * payload otherwise (roles is a plain array of ids), so take either shape
 * instead of betting on which one a given interaction carries.
 */
export function memberRoleIds(member) {
  const roles = member?.roles;
  if (!roles) return [];
  if (Array.isArray(roles)) return roles;
  if (roles.cache) return [...roles.cache.keys()];
  return [];
}

export async function isQuanter(member) {
  const roleId = await quanterRoleId();
  return !!roleId && memberRoleIds(member).includes(roleId);
}

// What counts against a quanter's one-job cap. pending_approval is not here on
// purpose: it is a request an admin has not let run, so it is renting nothing.
const ACTIVE = [db.JOB_STATUS.queued, db.JOB_STATUS.running];

/** The jobs `userId` owns that are queued or running, oldest first. */
export function activeJobsFor(jobs, userId) {
  return Object.values(jobs ?? {})
    .filter((job) => job.userId === userId && ACTIVE.includes(job.status))
    .sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
}

// Held between the scan and the caller's write. A scan of the job file cannot
// see a request that is still being validated, so two /quant's typed back to
// back would both find nothing running and both launch pods. Add-and-check
// here is synchronous, which is what closes that window.
const claims = new Set();

/**
 * Decide how a /quant request is handled, and reserve the caller's slot when it
 * will run without approval.
 *
 * A quanter gets one running job at a time, because nothing else stands between
 * them and as many pods as they can type. An admin is never capped, and anyone
 * else is not capped here either — their request waits on the Approve button,
 * which is the gate.
 *
 * Returns { admin, quanter, ok, running }. Call releaseRunSlot once the new job
 * is written (the scan can see it from then on) or the request is abandoned.
 */
export async function claimRunSlot({ userId, member, loadJobs = db.loadJobs }) {
  const admin = isAdmin(userId);
  const quanter = !admin && (await isQuanter(member));
  if (!quanter) return { admin, quanter, ok: true, running: null };

  if (claims.has(userId)) return { admin, quanter, ok: false, running: null };
  claims.add(userId);
  try {
    const running = activeJobsFor(await loadJobs(), userId)[0] ?? null;
    if (running) {
      claims.delete(userId);
      return { admin, quanter, ok: false, running };
    }
    return { admin, quanter, ok: true, running: null };
  } catch (err) {
    claims.delete(userId);
    throw err;
  }
}

export function releaseRunSlot(userId) {
  claims.delete(userId);
}
