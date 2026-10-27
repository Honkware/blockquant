import { MessageFlags } from 'discord.js';
import config from '../config.js';
import * as db from '../services/db.js';
import * as detached from '../services/detached.js';
import * as jobQueue from '../services/queue.js';
import * as workspace from '../services/workspace.js';
import * as embeds from '../utils/embeds.js';
import { getBalance } from '../services/runpod.js';
import { activeLines, historyLines, liveControllers } from '../services/jobView.js';

const reply = (interaction, embed) => interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });

export async function handleHistory(interaction) {
  const lines = historyLines(await db.loadJobs().catch(() => ({})));
  return reply(interaction, embeds.info('📜 Recent Jobs', lines.join('\n') || 'No jobs yet.'));
}

export async function handleQueueStatus(interaction) {
  const [jobs, settings] = await Promise.all([
    db.loadJobs().catch(() => ({})),
    db.loadSettings().catch(() => ({})),
  ]);
  const lines = activeLines(jobs);
  // The local queue only runs jobs for a local GPU, which the bot does not
  // send it today; mention it only when it is actually holding something.
  const { waiting, active } = jobQueue.status();
  return reply(interaction, embeds.info('📋 Queue', [
    settings.paused ? '**Paused:** new jobs wait for `/resume`' : '',
    lines.join('\n') || 'Nothing running or waiting.',
    waiting || active ? `**Local GPU queue:** ${active} running, ${waiting} waiting` : '',
  ].filter(Boolean).join('\n')));
}

export async function handleHealth(interaction) {
  const [jobs, settings, bal, cache] = await Promise.all([
    db.loadJobs().catch(() => ({})),
    db.loadSettings().catch(() => ({})),
    getBalance(),
    workspace.getCacheStats().catch(() => ({ entries: 0, totalBytes: 0 })),
  ]);
  const uptimeSec = Math.floor(process.uptime());
  const uptime =
    uptimeSec > 3600
      ? `${Math.floor(uptimeSec / 3600)}h ${Math.floor((uptimeSec % 3600) / 60)}m`
      : `${Math.floor(uptimeSec / 60)}m ${uptimeSec % 60}s`;
  const vals = Object.values(jobs);
  const running = vals.filter((j) => j.status === db.JOB_STATUS.running).length;
  const pending = vals.filter((j) => j.status === db.JOB_STATUS.pending_approval).length;
  const image = (config.RUNPOD_IMAGE || '').split(':').pop();

  return reply(interaction, embeds.info('🩺 Health', [
    `**Uptime:** ${uptime}${settings.paused ? '  ·  **paused**' : ''}`,
    `**Jobs:** ${running} running, ${pending} awaiting approval, ${liveControllers(detached.list()).length} controller(s) alive`,
    `**RunPod:** ${bal ? `$${bal.balance.toFixed(2)} balance, $${bal.spendPerHr.toFixed(2)}/h now` : 'balance unavailable'}`,
    image ? `**Image:** \`${image.slice(0, 19)}\`` : '',
    cache.entries ? `**Local cache:** ${cache.entries} model(s), ${(cache.totalBytes / 1024 ** 3).toFixed(2)} GB` : '',
    `**Node:** ${process.version}`,
  ].filter(Boolean).join('\n')));
}
