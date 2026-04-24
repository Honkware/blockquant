import { EmbedBuilder } from 'discord.js';
import { progressBar, truncate } from './format.js';
import { sanitizeErrorText } from '../errors/taxonomy.js';

const COLORS = {
  info: 0x5865f2, // blurple
  success: 0x57f287, // green
  warning: 0xfee75c, // yellow
  error: 0xed4245, // red
  pending: 0xeb459e, // fuchsia
};

// ── Quant Job Embeds ────────────────────────────────────────────────────────

export function jobQueued({ url, bpws, categories, userId }) {
  return new EmbedBuilder()
    .setTitle('📦 Quantization Job Queued')
    .setColor(COLORS.info)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `\`${truncate(url, 80)}\``, inline: false },
      { name: 'BPW', value: bpws.map((b) => `\`${b}\``).join('  '), inline: true },
      { name: 'Categories', value: categories.join(', ') || 'None', inline: true }
    )
    .setTimestamp();
}

export function jobProgress({
  url,
  userId,
  stage,
  progress,
  overall,
  message,
  currentBPW,
  bpwIndex,
  totalBPWs,
}) {
  const bar = progressBar(Math.round(overall ?? progress ?? 0));
  const bpwText =
    currentBPW != null
      ? `\`${currentBPW}\` bpw  (${(bpwIndex ?? 0) + 1}/${totalBPWs ?? '?'})`
      : '—';

  return new EmbedBuilder()
    .setTitle('⚙️ Quantization In Progress')
    .setColor(COLORS.pending)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `\`${truncate(url, 80)}\``, inline: false },
      { name: 'Stage', value: stage ?? 'Working…', inline: true },
      { name: 'Current', value: bpwText, inline: true },
      { name: 'Progress', value: bar, inline: false },
      { name: 'Status', value: truncate(message || 'Processing…', 200), inline: false }
    )
    .setTimestamp();
}

export function jobComplete({ url, userId, results }) {
  const lines = results.map((r) => {
    const icon = r.reused ? '♻️' : r.pushed ? '✅' : '❌';
    const href = r.treeUrl || r.url;
    const link = href
      ? `[Link](${href})`
      : r.error
        ? `Error: ${truncate(r.error, 60)}`
        : 'No URL';
    return `${icon}  **${r.bpw} bpw** — ${r.duration} — ${link}`;
  });

  return new EmbedBuilder()
    .setTitle('✅ Quantization Complete')
    .setColor(COLORS.success)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `\`${truncate(url, 80)}\``, inline: false },
      { name: 'Results', value: lines.join('\n') || 'No results', inline: false }
    )
    .setTimestamp();
}

export function jobFailed({ url, userId, error }) {
  const safeError = sanitizeErrorText(error);
  return new EmbedBuilder()
    .setTitle('❌ Quantization Failed')
    .setColor(COLORS.error)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `\`${truncate(url, 80)}\``, inline: false },
      { name: 'Error', value: `\`\`\`${truncate(safeError, 500)}\`\`\``, inline: false }
    )
    .setTimestamp();
}

// ── Generic Embeds ──────────────────────────────────────────────────────────

export function info(title, description) {
  return new EmbedBuilder().setTitle(title).setDescription(description).setColor(COLORS.info);
}

export function success(title, description) {
  return new EmbedBuilder().setTitle(title).setDescription(description).setColor(COLORS.success);
}

export function error(title, description) {
  return new EmbedBuilder().setTitle(title).setDescription(description).setColor(COLORS.error);
}

export function warning(title, description) {
  return new EmbedBuilder().setTitle(title).setDescription(description).setColor(COLORS.warning);
}
