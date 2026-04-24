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
  const type = interaction.options.getString('type') || 'exp';

  if (type === 'models') {
    // Fetch from API dashboard
    const API_URL = process.env.BLOCKQUANT_API_URL || 'http://localhost:8000';
    try {
      const resp = await fetch(`${API_URL}/dashboard/api/leaderboard?limit=10`);
      if (!resp.ok) throw new Error('API error');
      const data = await resp.json();

      if (!data || data.length === 0) {
        return interaction.reply({
          embeds: [embeds.info('🏆 Quantization Leaderboard', 'No models quantized yet!')],
          flags: MessageFlags.Ephemeral,
        });
      }

      const lines = data.map((m, i) => {
        const medal = ['🥇', '🥈', '🥉'][i] ?? `**${i + 1}.**`;
        return `${medal} **${m.model_id}** — ${m.quant_count} quant(s) (${m.top_variant})`;
      });

      return interaction.reply({
        embeds: [embeds.info('🏆 Top Quantized Models', lines.join('\n'))],
      });
    } catch (err) {
      return interaction.reply({
        embeds: [embeds.info('🏆 Leaderboard', `Failed to fetch: ${err.message}`)],
        flags: MessageFlags.Ephemeral,
      });
    }
  }

  // Default: EXP leaderboard
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

export async function handleHistory(interaction) {
  const API_URL = process.env.BLOCKQUANT_API_URL || 'http://localhost:8000';
  try {
    const resp = await fetch(`${API_URL}/dashboard/api/recent?limit=5`);
    if (!resp.ok) throw new Error('API error');
    const jobs = await resp.json();

    if (!jobs || jobs.length === 0) {
      return interaction.reply({
        embeds: [embeds.info('📜 History', 'No jobs yet!')],
        flags: MessageFlags.Ephemeral,
      });
    }

    const lines = jobs.map((j, i) => {
      const status = j.success === true ? '✅' : j.success === false ? '❌' : '⏳';
      const time = j.wall_time_seconds ? `${j.wall_time_seconds.toFixed(0)}s` : '—';
      const model = j.model_id || 'unknown';
      const variants = Array.isArray(j.variants) ? j.variants.join(',') : j.variants;
      return `${status} **${model}** @ ${variants} (${time})`;
    });

    await interaction.reply({
      embeds: [embeds.info('📜 Recent Jobs', lines.join('\n'))],
      flags: MessageFlags.Ephemeral,
    });
  } catch (err) {
    await interaction.reply({
      embeds: [embeds.info('📜 History', `Failed to fetch: ${err.message}`)],
      flags: MessageFlags.Ephemeral,
    });
  }
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
