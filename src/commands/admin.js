import { MessageFlags } from 'discord.js';
import os from 'node:os';
import config from '../config.js';
import * as db from '../services/db.js';
import * as detached from '../services/detached.js';
import * as jobQueue from '../services/queue.js';
import { liveControllers } from '../services/jobView.js';
import * as workspace from '../services/workspace.js';
import * as embeds from '../utils/embeds.js';

function isAdmin(userId) {
  return config.ADMIN_IDS.includes(userId);
}

/**
 * /config quanter-role [role] — set the role that skips approval, or clear it
 * by leaving the option off. Stored in the settings file rather than .env
 * because it has to take effect without a restart.
 */
export async function handleConfig(interaction) {
  if (!isAdmin(interaction.user.id)) {
    return interaction.reply({
      embeds: [embeds.error('🚫 Forbidden', 'Admin only.')],
      flags: MessageFlags.Ephemeral,
    });
  }

  const role = interaction.options.getRole('role');
  await db.patchSettings({ quanterRoleId: role ? role.id : null });

  return interaction.reply({
    embeds: [
      role
        ? embeds.success(
            'Quanter role set',
            `Members of <@&${role.id}> can run \`/quant\` without approval, one job at a time, and \`/stop\` their own.`
          )
        : embeds.warning('Quanter role cleared', 'Every request needs admin approval again.'),
    ],
    flags: MessageFlags.Ephemeral,
  });
}

export async function handlePause(interaction) {
  if (!isAdmin(interaction.user.id)) {
    return interaction.reply({
      embeds: [embeds.error('🚫 Forbidden', 'Admin only.')],
      flags: MessageFlags.Ephemeral,
    });
  }
  // Persisted, and checked where RunPod jobs start (approval and the quanter
  // path). It used to pause only the local queue, which RunPod jobs never go
  // through, so it said "no new jobs will start" and stopped none of them.
  await db.patchSettings({ paused: true });
  jobQueue.pause();
  await interaction.reply({
    embeds: [embeds.warning('⏸️ Paused', 'New jobs wait for `/resume`. Running jobs keep going; `/stop` ends one.')],
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleResume(interaction) {
  if (!isAdmin(interaction.user.id)) {
    return interaction.reply({
      embeds: [embeds.error('🚫 Forbidden', 'Admin only.')],
      flags: MessageFlags.Ephemeral,
    });
  }
  await db.patchSettings({ paused: false });
  jobQueue.resume();
  await interaction.reply({
    embeds: [embeds.success('▶️ Resumed', 'Requests waiting for approval can be approved again.')],
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleDiag(interaction) {
  if (!isAdmin(interaction.user.id)) {
    return interaction.reply({
      embeds: [embeds.error('🚫 Forbidden', 'Admin only.')],
      flags: MessageFlags.Ephemeral,
    });
  }

  const q = jobQueue.diagnostics();
  const jobs = await db.loadJobs().catch(() => ({}));
  const counts = Object.values(jobs).reduce((acc, job) => {
    const k = job.status ?? 'unknown';
    acc[k] = (acc[k] ?? 0) + 1;
    return acc;
  }, {});
  const mem = process.memoryUsage();
  const cache = await workspace.getCacheStats().catch(() => ({ entries: 0, totalBytes: 0 }));
  const controllers = liveControllers(detached.list());

  await interaction.reply({
    embeds: [
      embeds.info(
        '🧰 Diagnostics',
        [
          `**Host:** ${os.hostname()} (${os.platform()} ${os.release()})`,
          `**Node:** ${process.version}`,
          `**Controllers alive:** ${controllers.length}${controllers.length
            ? ` (${controllers.map((c) => `${c.meta?.modelId ?? c.id} pid ${c.pid}`).join(', ').slice(0, 300)})` : ''}`,
          `**Local queue:** waiting=${q.waiting}, active=${q.active}, paused=${q.paused}`,
          `**Job DB:** ${
            Object.entries(counts)
              .map(([k, v]) => `${k}=${v}`)
              .join(', ') || 'empty'
          }`,
          `**Memory:** rss=${(mem.rss / 1048576).toFixed(1)}MB heap=${(mem.heapUsed / 1048576).toFixed(1)}MB`,
          `**Local cache:** ${cache.entries} model(s), ${(cache.totalBytes / 1024 ** 3).toFixed(2)} GB`,
        ].join('\n')
      ),
    ],
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleCache(interaction) {
  if (!isAdmin(interaction.user.id)) {
    return interaction.reply({
      embeds: [embeds.error('🚫 Forbidden', 'Admin only.')],
      flags: MessageFlags.Ephemeral,
    });
  }

  const action = interaction.options.getString('action') ?? 'status';
  if (action === 'clear') {
    await workspace.clearCache();
    return interaction.reply({
      embeds: [embeds.warning('🧹 Cache Cleared', 'All cached model files were removed.')],
      flags: MessageFlags.Ephemeral,
    });
  }

  if (action === 'prune') {
    const pruned = await workspace.pruneCache('manual');
    const stats = await workspace.getCacheStats();
    return interaction.reply({
      embeds: [
        embeds.info(
          '🧹 Cache Pruned',
          `Removed **${pruned.removed}** entr${pruned.removed === 1 ? 'y' : 'ies'}.\nNow holding **${stats.entries}** model(s), **${(stats.totalBytes / 1024 ** 3).toFixed(2)} GB**.`
        ),
      ],
      flags: MessageFlags.Ephemeral,
    });
  }

  const stats = await workspace.getCacheStats();
  const top = stats.models
    .slice(0, 5)
    .map(
      (m) =>
        `• \`${m.modelId}\` — ${(m.sizeBytes / 1024 ** 3).toFixed(2)} GB (used <t:${Math.floor((m.lastUsedAt ?? Date.now()) / 1000)}:R>)`
    )
    .join('\n');

  return interaction.reply({
    embeds: [
      embeds.info(
        '🗃️ Cache Status',
        [
          `**Entries:** ${stats.entries}`,
          `**Total size:** ${(stats.totalBytes / 1024 ** 3).toFixed(2)} GB`,
          top || 'No cache entries',
        ].join('\n')
      ),
    ],
    flags: MessageFlags.Ephemeral,
  });
}
