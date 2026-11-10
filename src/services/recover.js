// What a restart has to put back together.
//
// Controllers outlive the bot (services/detached.js), so a restart leaves jobs
// whose pods are still working, and jobs whose controller finished with nobody
// tailing it. Both go back to the code that was driving the card, with the
// controllers they own. A finished one is worth handing over too: its log and
// exit status are still on disk, and replaying them posts the real result
// without renting a pod again.
//
// The caller has to be logged in. The thread and the card are fetched over
// REST, and discord.js has no token for that until client.login() resolves.
import * as db from './db.js';
import { drop, list } from './detached.js';
import { getLogger } from '../logger.js';

const log = getLogger('recover');

const OPEN = [db.JOB_STATUS.queued, db.JOB_STATUS.running, db.JOB_STATUS.interrupted];
const LOST = 'Interrupted by a bot restart';

// A resume settles hours after it starts, so the job is read again here: one
// that reached its own ending in the meantime keeps it.
function failIfOpen(jobId, error) {
  return db.patchJob(jobId, (j) =>
    OPEN.includes(j.status) ? { ...j, status: db.JOB_STATUS.failed, error, failedAt: Date.now() } : j
  );
}

/**
 * Reconnect every unfinished job. `resume({ client, job, resumeFrom, lost })`
 * gets the job's controllers as variant -> handle, or `lost` when none are
 * left, and resolves when the job ends.
 */
export async function recoverJobs({ client, resume }) {
  const jobs = await db.listRecoverableJobs();
  const carded = new Map(jobs.filter((j) => j.threadId && j.progressMessageId).map((j) => [j.id, j]));

  const byJob = new Map();
  for (const c of list()) {
    const job = c.kind === 'quant' ? carded.get(c.meta?.jobId) : null;
    if (job) {
      if (!byJob.has(job.id)) byJob.set(job.id, new Map());
      for (const v of c.meta.variants || []) byJob.get(job.id).set(v, c);
    } else if (c.running) {
      log.warn(`${c.kind} ${c.meta?.modelId || c.id} is still running (pid ${c.pid}) with no card to drive: ${c.logPath}`);
    } else {
      log.warn(`${c.kind} ${c.meta?.modelId || c.id} finished while the bot was down: ${c.logPath}`);
      drop(c);
    }
  }

  for (const job of jobs) {
    if (!carded.has(job.id)) {
      await failIfOpen(job.id, LOST);
      continue;
    }
    const handles = byJob.get(job.id);
    log.info(`resuming job ${job.id} (${job.modelId}) with ${handles?.size ?? 0} controller(s)`);
    // Not awaited: this settles when the quant does.
    resume({ client, job, resumeFrom: handles ?? new Map(), lost: handles ? null : LOST }).catch((err) => {
      log.error(`resume failed for ${job.id}: ${err.message}`);
      return failIfOpen(job.id, err.message).catch((e) => log.error(`could not close ${job.id}: ${e.message}`));
    });
  }
}
