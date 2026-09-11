import { MessageFlags } from 'discord.js';
import { getLogger } from '../logger.js';
import config from '../config.js';
import * as db from '../services/db.js';
import * as embeds from '../utils/embeds.js';
import { runApprovedJob } from './quant.js';
import { toUserMessage } from '../errors/taxonomy.js';

const log = getLogger('cmd:approval');

// In-flight approvals, by job id. The DB status guard below blocks repeat
// clicks across restarts, but there is an async window between reading the job
// and writing its new status during which a rapid second click (or a duplicate
// interaction delivery) could slip through and launch a SECOND set of pods.
// This Set is checked-and-set synchronously with no await in between, so it
// closes that window within a single bot process.
const inFlightApprovals = new Set();

/**
 * Handle the Approve / Deny buttons on a pending quant request.
 * Only admins (config.ADMIN_IDS) may act; the request is looked up by the job
 * id encoded in the button customId (`bq:approve:<id>` / `bq:deny:<id>`).
 */
export async function handleApproval(interaction) {
  const [ns, action, jobId] = interaction.customId.split(':');
  if (ns !== 'bq' || !jobId) return false;

  if (!config.ADMIN_IDS.includes(interaction.user.id)) {
    await interaction.reply({
      content: 'Only an admin can approve or deny requests.',
      flags: MessageFlags.Ephemeral,
    });
    return true;
  }

  const jobs = await db.loadJobs();
  const job = jobs[jobId];
  if (!job) {
    await interaction.reply({
      content: 'That request is no longer available.',
      flags: MessageFlags.Ephemeral,
    });
    return true;
  }

  if (job.status !== db.JOB_STATUS.pending_approval) {
    await interaction.reply({
      content: `This request was already handled (status: ${job.status}).`,
      flags: MessageFlags.Ephemeral,
    });
    return true;
  }

  // Synchronous re-entry lock: must come before any await so two near-
  // simultaneous clicks can't both pass. Only the approve path launches pods,
  // but lock here so a deny can't race an approve either.
  if (inFlightApprovals.has(jobId)) {
    await interaction.reply({
      content: 'This request is already being handled.',
      flags: MessageFlags.Ephemeral,
    });
    return true;
  }
  inFlightApprovals.add(jobId);

  // Lock the buttons immediately so a second admin can't double-act.
  const decidedEmbed = (verb, color) =>
    embeds[color](
      `Request ${verb}`,
      [
        `**Model:** [\`${job.modelId}\`](https://huggingface.co/${job.modelId})`,
        `**Variants:** ${job.variants.join(', ')}`,
        `**Requested by:** <@${job.userId}>`,
        `**${verb} by:** <@${interaction.user.id}>`,
      ].join('\n')
    );

  try {
    if (action === 'deny') {
      await db.patchJob(jobId, {
        status: db.JOB_STATUS.rejected,
        rejectedAt: Date.now(),
        rejectedBy: interaction.user.id,
      });
      await interaction.update({ embeds: [decidedEmbed('Denied', 'warning')], components: [] });
      log.info(`Request ${jobId} denied by ${interaction.user.tag}`);
      return true;
    }

    if (action === 'approve') {
      // Flip status to running BEFORE launching anything. The pending_approval
      // guard above only blocks repeat clicks once this is persisted, so doing
      // it here (not inside the hours-long run) is what stops a second click
      // from spawning a duplicate set of controllers/pods.
      await db.patchJob(jobId, {
        status: db.JOB_STATUS.running,
        approvedAt: Date.now(),
        approvedBy: interaction.user.id,
      });
      await interaction.update({ embeds: [decidedEmbed('Approved', 'success')], components: [] });
      log.info(`Request ${jobId} approved by ${interaction.user.tag}`);
      try {
        await runApprovedJob({ interaction, job });
      } catch (err) {
        log.error(`Failed to start approved job ${jobId}`, { error: err.message });
        await db.patchJob(jobId, {
          status: db.JOB_STATUS.failed,
          failedAt: Date.now(),
          error: err.message,
        });
        await interaction.followUp({
          content: `Could not start the job: ${toUserMessage(err) || err.message}`,
          flags: MessageFlags.Ephemeral,
        });
      }
      return true;
    }

    return false;
  } finally {
    inFlightApprovals.delete(jobId);
  }
}
