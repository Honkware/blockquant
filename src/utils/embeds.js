import { EmbedBuilder } from 'discord.js';
import { progressBar, truncate } from './format.js';

const COLORS = {
  info: 0x5865f2, // blurple
  success: 0x57f287, // green
  warning: 0xfee75c, // yellow
  error: 0xed4245, // red
  pending: 0xeb459e, // fuchsia
};

// ── Quant Job Embeds ────────────────────────────────────────────────────────

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
      { name: 'Model', value: `[\`${truncate(url, 80)}\`](https://huggingface.co/${url})`, inline: false },
      { name: 'Stage', value: stage ?? 'Working…', inline: true },
      { name: 'Current', value: bpwText, inline: true },
      { name: 'Progress', value: bar, inline: false },
      { name: 'Status', value: truncate(message || 'Processing…', 200), inline: false }
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
