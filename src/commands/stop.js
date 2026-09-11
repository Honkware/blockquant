import fs from 'node:fs';
import { MessageFlags, ActionRowBuilder, ButtonBuilder, ButtonStyle } from 'discord.js';
import { getLogger } from '../logger.js';
import config from '../config.js';
import * as db from '../services/db.js';
import * as embeds from '../utils/embeds.js';
import { list as listControllers } from '../services/detached.js';
import { terminatePod } from '../services/runpod.js';
import { isAdmin, isQuanter } from '../services/access.js';

const log = getLogger('cmd:stop');

// The controller prints this once per pod it rents, and a retry rents another,
// so a log can hold several. Same line runpodCli.js reads for progress.
const POD_ID = /Pod ID:\s*(\S+)/g;

// A controller record carries the model and when it started, not the job id:
// the meta is built in runpodCli.js. So a job's controllers are matched on the
// model plus "started no earlier than the job did".
const CLOCK_SLACK_MS = 60_000;

const STOPPABLE = [db.JOB_STATUS.queued, db.JOB_STATUS.running];

// Stops in flight, by job id. Checked synchronously before any await so a
// double-clicked Stop button doesn't send two DELETEs and two SIGTERMs.
const stopping = new Set();

export function stopButtons(jobId) {
  return new ActionRowBuilder().addComponents(
    new ButtonBuilder().setCustomId(`bq:stop:${jobId}`).setLabel('Stop').setStyle(ButtonStyle.Danger)
  );
}

export function stoppedEmbed(job) {
  return embeds.warning(
    'Job stopped',
    [
      `**Model:** [\`${job.modelId}\`](https://huggingface.co/${job.modelId})`,
      `**Variants:** ${(job.variants ?? []).join(', ') || '—'}`,
      `**Requested by:** <@${job.userId}>`,
      job.stoppedBy ? `**Stopped by:** <@${job.stoppedBy}>` : '',
      job.stoppedPods?.length ? `**Pods terminated:** ${job.stoppedPods.join(', ')}` : '',
    ]
      .filter(Boolean)
      .join('\n')
  );
}

/** Admins stop anything; a quanter stops their own; nobody else stops at all. */
export async function canStop({ userId, member, job }) {
  if (isAdmin(userId)) return true;
  if (!job || job.userId !== userId) return false;
  return isQuanter(member);
}

export function controllersForJob(job, list = listControllers) {
  const since = (job.startedAt ?? job.approvedAt ?? job.createdAt ?? 0) - CLOCK_SLACK_MS;
  return list().filter(
    (c) => c.running && c.meta?.modelId === job.modelId && (c.startedAt ?? 0) >= since
  );
}

function podIdsFromLog(logPath) {
  let text;
  try {
    text = fs.readFileSync(logPath, 'utf8');
  } catch {
    return [];
  }
  return [...text.matchAll(POD_ID)].map((m) => m[1]);
}

/**
 * Terminate the pods this job is renting, then kill its controllers.
 *
 * The order is the whole point. A controller killed first leaves its pod up
 * with nobody to end it, and the reaper deliberately ignores pods younger than
 * 15 minutes, so the bill runs to RunPod's 8h max_runtime. Killing the pod
 * first costs at most a controller noticing its pod vanished, which it reports
 * as a failed variant.
 */
export async function stopJob(
  job,
  { by, kill = (pid, sig) => process.kill(pid, sig), sleep = (ms) => new Promise((r) => setTimeout(r, ms)) } = {}
) {
  const controllers = controllersForJob(job);
  // Two sources because neither covers the other: the live progress stream
  // records pod ids on the job as they are announced, and the log is all that
  // is left for a job whose bot process has since restarted.
  const podIds = new Set(job.podIds ?? []);
  for (const c of controllers) {
    for (const id of podIdsFromLog(c.logPath)) podIds.add(id);
  }

  const terminated = [];
  const failed = [];
  for (const id of podIds) {
    if (await terminatePod(id)) terminated.push(id);
    else failed.push(id);
  }

  const killed = [];
  for (const c of controllers) {
    try {
      kill(c.pid, 'SIGTERM');
      killed.push(c.pid);
    } catch (err) {
      log.debug(`controller ${c.pid} was already gone: ${err.message}`);
    }
  }
  // A controller waiting on an SSH read can sit on SIGTERM for a long time.
  if (killed.length) {
    await sleep(config.PROCESS_KILL_GRACE_MS);
    for (const pid of killed) {
      try {
        kill(pid, 0);
        kill(pid, 'SIGKILL');
      } catch {
        /* it took the SIGTERM */
      }
    }
  }

  await db.patchJob(job.id, {
    status: db.JOB_STATUS.stopped,
    stoppedAt: Date.now(),
    stoppedBy: by,
    stoppedPods: terminated,
  });

  log.warn(
    `Job ${job.id} stopped by ${by}: pods ${terminated.join(', ') || 'none'}` +
      `${failed.length ? ` (failed to terminate ${failed.join(', ')})` : ''}, controllers ${killed.join(', ') || 'none'}`
  );
  return { pods: terminated, failedPods: failed, killed };
}

/** Rewrite the request card and the progress embed, and say so in the thread. */
async function announceStopped(client, jobId) {
  const job = (await db.loadJobs().catch(() => ({})))[jobId];
  if (!job) return;
  const embed = stoppedEmbed(job);

  const channel = job.channelId ? await client.channels.fetch(job.channelId).catch(() => null) : null;
  if (channel && job.cardMessageId) {
    const card = await channel.messages.fetch(job.cardMessageId).catch(() => null);
    await card?.edit({ embeds: [embed], components: [] }).catch(() => {});
  }

  const thread = job.threadId ? await client.channels.fetch(job.threadId).catch(() => null) : null;
  if (!thread) return;
  if (job.progressMessageId) {
    const msg = await thread.messages.fetch(job.progressMessageId).catch(() => null);
    await msg?.edit({ embeds: [embed], components: [] }).catch(() => {});
  }
  await thread
    .send(`<@${job.userId}> this job was stopped by <@${job.stoppedBy}>. Its pods are terminated.`)
    .catch(() => {});
}

function summary(result) {
  const pods = result.pods.length ? `Terminated ${result.pods.join(', ')}.` : 'No pod was up to terminate.';
  const ctl = result.killed.length ? ` Killed controller ${result.killed.join(', ')}.` : '';
  const bad = result.failedPods.length
    ? ` RunPod refused to terminate ${result.failedPods.join(', ')} — check the console.`
    : '';
  return `${pods}${ctl}${bad}`;
}

async function resolveJob(userId, wanted) {
  const jobs = await db.loadJobs();
  const live = Object.values(jobs).filter((j) => STOPPABLE.includes(j.status));
  if (wanted) {
    const key = wanted.trim().toLowerCase();
    return live.find((j) => j.id === key || j.id.startsWith(key)) ?? null;
  }
  return (
    live
      .filter((j) => j.userId === userId)
      .sort((a, b) => (b.startedAt ?? b.createdAt ?? 0) - (a.startedAt ?? a.createdAt ?? 0))[0] ?? null
  );
}

async function liveJobList() {
  const jobs = await db.loadJobs().catch(() => ({}));
  return Object.values(jobs)
    .filter((j) => STOPPABLE.includes(j.status))
    .slice(0, 10)
    .map((j) => `\`${j.id.slice(0, 8)}\` · \`${j.modelId}\` · <@${j.userId}>`)
    .join('\n');
}

/** /stop [job] — no job means the caller's own running one. */
export async function handleStop(interaction) {
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });

  const wanted = interaction.options.getString('job');
  const job = await resolveJob(interaction.user.id, wanted);
  if (!job) {
    const list = isAdmin(interaction.user.id) ? await liveJobList() : '';
    return interaction.editReply({
      embeds: [
        embeds.error(
          'Nothing to stop',
          wanted
            ? `No running job matches \`${wanted}\`.`
            : ['You have no running job.', list && `\nRunning now:\n${list}`].filter(Boolean).join('\n')
        ),
      ],
    });
  }

  if (!(await canStop({ userId: interaction.user.id, member: interaction.member, job }))) {
    return interaction.editReply({
      embeds: [
        embeds.error('Not allowed', 'An admin can stop any job, and a quanter can stop their own.'),
      ],
    });
  }

  if (stopping.has(job.id)) {
    return interaction.editReply({ embeds: [embeds.warning('Already stopping', 'That job is being stopped.')] });
  }
  stopping.add(job.id);
  try {
    const result = await stopJob(job, { by: interaction.user.id });
    await announceStopped(interaction.client, job.id);
    return interaction.editReply({
      embeds: [embeds.warning(`Stopped \`${job.modelId}\``, summary(result))],
    });
  } finally {
    stopping.delete(job.id);
  }
}

/** The Stop button on the request card and on the progress embed. */
export async function handleStopButton(interaction) {
  const jobId = interaction.customId.split(':')[2];
  const job = (await db.loadJobs())[jobId];
  if (!job) {
    await interaction.reply({ content: 'That job is no longer available.', flags: MessageFlags.Ephemeral });
    return true;
  }
  if (!(await canStop({ userId: interaction.user.id, member: interaction.member, job }))) {
    await interaction.reply({
      content: 'An admin can stop any job, and a quanter can stop their own.',
      flags: MessageFlags.Ephemeral,
    });
    return true;
  }
  if (!STOPPABLE.includes(job.status) || stopping.has(jobId)) {
    await interaction.reply({
      content: `That job is not running (status: ${job.status}).`,
      flags: MessageFlags.Ephemeral,
    });
    return true;
  }

  stopping.add(jobId);
  try {
    await interaction.deferUpdate();
    const result = await stopJob(job, { by: interaction.user.id });
    await announceStopped(interaction.client, jobId);
    await interaction.followUp({ content: summary(result), flags: MessageFlags.Ephemeral }).catch(() => {});
  } finally {
    stopping.delete(jobId);
  }
  return true;
}
