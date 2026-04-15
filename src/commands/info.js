import { MessageFlags } from 'discord.js';
import * as db from '../services/db.js';
import * as jobQueue from '../services/queue.js';
import * as workspace from '../services/workspace.js';
import * as embeds from '../utils/embeds.js';

export async function handleScore(interaction) {
  const target = interaction.options.getUser('user') ?? interaction.user;
  const users = await db.loadUsers();
  const entry = users[target.id] ?? { exp: 0, lastQuant: 0 };

  const lastQuant = entry.lastQuant ? `<t:${Math.floor(entry.lastQuant / 1000)}:R>` : 'Never';

  await interaction.reply({
    embeds: [
      embeds.info(
        `📊 ${target.displayName}`,
        `**EXP:** ${entry.exp}\n**Last Quant:** ${lastQuant}`
      ),
    ],
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleLeaderboard(interaction) {
  const users = await db.loadUsers();
  const sorted = Object.entries(users)
    .sort(([, a], [, b]) => (b.exp ?? 0) - (a.exp ?? 0))
    .slice(0, 10);

  if (sorted.length === 0) {
    return interaction.reply({
      embeds: [embeds.info('🏆 Leaderboard', 'No users yet!')],
      flags: MessageFlags.Ephemeral,
    });
  }

  const lines = sorted.map(([id, u], i) => {
    const medal = ['🥇', '🥈', '🥉'][i] ?? `**${i + 1}.**`;
    return `${medal} <@${id}> — **${u.exp ?? 0}** EXP`;
  });

  await interaction.reply({
    embeds: [embeds.info('🏆 Leaderboard', lines.join('\n'))],
    allowedMentions: { users: [] },
  });
}

export async function handleQueueStatus(interaction) {
  const { waiting, active, paused } = jobQueue.status();
  const statusIcon = paused ? '⏸️ Paused' : '▶️ Active';

  await interaction.reply({
    embeds: [
      embeds.info(
        '📋 Queue Status',
        [`**Status:** ${statusIcon}`, `**Active Jobs:** ${active}`, `**Waiting:** ${waiting}`].join(
          '\n'
        )
      ),
    ],
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleHealth(interaction) {
  const q = jobQueue.diagnostics();
  const cache = await workspace.getCacheStats().catch(() => ({ entries: 0, totalBytes: 0 }));
  const uptimeSec = Math.floor(process.uptime());
  const uptime =
    uptimeSec > 3600
      ? `${Math.floor(uptimeSec / 3600)}h ${Math.floor((uptimeSec % 3600) / 60)}m`
      : `${Math.floor(uptimeSec / 60)}m ${uptimeSec % 60}s`;

  await interaction.reply({
    embeds: [
      embeds.info(
        '🩺 Health',
        [
          `**Uptime:** ${uptime}`,
          `**Queue:** waiting=${q.waiting}, active=${q.active}, paused=${q.paused}`,
          `**Cache:** ${cache.entries} model(s), ${(cache.totalBytes / 1024 ** 3).toFixed(2)} GB`,
          `**Node:** ${process.version}`,
        ].join('\n')
      ),
    ],
    flags: MessageFlags.Ephemeral,
  });
}
