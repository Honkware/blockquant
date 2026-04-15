import { MessageFlags } from 'discord.js';
import os from 'node:os';
import config from '../config.js';
import * as db from '../services/db.js';
import * as jobQueue from '../services/queue.js';
import * as workspace from '../services/workspace.js';
import * as embeds from '../utils/embeds.js';

function isAdmin(userId) {
  return config.ADMIN_IDS.includes(userId);
}

export async function handleGive(interaction) {
  if (!isAdmin(interaction.user.id)) {
    return interaction.reply({
      embeds: [embeds.error('🚫 Forbidden', 'Admin only.')],
      flags: MessageFlags.Ephemeral,
    });
  }

  const target = interaction.options.getUser('user', true);
  const amount = interaction.options.getNumber('amount', true);

  if (amount === 0) {
    return interaction.reply({
      embeds: [embeds.warning('🤔', 'Amount cannot be zero.')],
      flags: MessageFlags.Ephemeral,
    });
  }

  const users = await db.loadUsers();
  if (!users[target.id]) users[target.id] = { exp: 0, lastQuant: 0 };
  users[target.id].exp += amount;
  await db.saveUsers(users);

  const verb = amount > 0 ? 'Gave' : 'Deducted';
  await interaction.reply({
    embeds: [
      embeds.success(
        '💰 EXP Updated',
        `${verb} **${Math.abs(amount)}** EXP ${amount > 0 ? 'to' : 'from'} <@${target.id}>.\nNew balance: **${users[target.id].exp}**`
      ),
    ],
    allowedMentions: { users: [] },
  });
}

export async function handlePause(interaction) {
  if (!isAdmin(interaction.user.id)) {
    return interaction.reply({
      embeds: [embeds.error('🚫 Forbidden', 'Admin only.')],
      flags: MessageFlags.Ephemeral,
    });
  }
  jobQueue.pause();
  await interaction.reply({
    embeds: [embeds.warning('⏸️ Queue Paused', 'No new jobs will start until resumed.')],
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
  jobQueue.resume();
  await interaction.reply({
    embeds: [embeds.success('▶️ Queue Resumed', 'Jobs will now continue processing.')],
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

  await interaction.reply({
    embeds: [
      embeds.info(
        '🧰 Diagnostics',
        [
          `**Host:** ${os.hostname()} (${os.platform()} ${os.release()})`,
          `**Node:** ${process.version}`,
          `**Queue:** waiting=${q.waiting}, active=${q.active}, paused=${q.paused}`,
          `**Job DB:** ${
            Object.entries(counts)
              .map(([k, v]) => `${k}=${v}`)
              .join(', ') || 'empty'
          }`,
          `**Memory:** rss=${(mem.rss / 1048576).toFixed(1)}MB heap=${(mem.heapUsed / 1048576).toFixed(1)}MB`,
          `**Cache:** ${cache.entries} model(s), ${(cache.totalBytes / 1024 ** 3).toFixed(2)} GB`,
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
