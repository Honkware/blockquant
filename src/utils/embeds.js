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

const STAGE_META = {
  Preparing: { icon: '🧰', label: 'Preparing workspace' },
  Cached: { icon: '🗂️', label: 'Restoring from cache' },
  Downloading: { icon: '⬇️', label: 'Downloading source model' },
  Quantizing: { icon: '⚙️', label: 'Quantizing model weights' },
  Compiling: { icon: '🧱', label: 'Compiling EXL3 artifacts' },
  Uploading: { icon: '☁️', label: 'Uploading to Hugging Face' },
  Reusing: { icon: '♻️', label: 'Reusing existing upload' },
};

function stageLine(stage) {
  const meta = STAGE_META[stage] ?? { icon: '🔄', label: stage || 'Working' };
  return `${meta.icon} ${meta.label}`;
}

// ── Quant Job Embeds ────────────────────────────────────────────────────────

export function jobQueued({ url, bpws, categories, userId }) {
  const sorted = [...bpws].sort((a, b) => a - b);
  return new EmbedBuilder()
    .setTitle('📦 Job Queued')
    .setColor(COLORS.info)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `\`${truncate(url, 80)}\``, inline: false },
      { name: 'Variants', value: sorted.map((b) => `\`${b}\``).join('  '), inline: true },
      { name: 'Categories', value: categories.join(', ') || 'General', inline: true }
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
  const overallPct = Math.round(overall ?? progress ?? 0);
  const stagePct = Math.round(progress ?? 0);
  const bar = progressBar(overallPct);
  const bpwText =
    currentBPW != null
      ? `\`${currentBPW}\` bpw  (${(bpwIndex ?? 0) + 1}/${totalBPWs ?? '?'})`
      : '—';
  const stageText = stageLine(stage);
  const safeStatus = truncate((message || 'Processing...').replace(/\s+/g, ' ').trim(), 220);

  return new EmbedBuilder()
    .setTitle('⚙️ Quantization In Progress')
    .setColor(COLORS.pending)
    .setDescription(`Requested by <@${userId}>\n${stageText}`)
    .addFields(
      { name: 'Model', value: `\`${truncate(url, 80)}\``, inline: false },
      { name: 'Current Variant', value: bpwText, inline: true },
      { name: 'Stage Progress', value: `${stagePct}%`, inline: true },
      { name: 'Overall Progress', value: bar, inline: false },
      { name: 'Status', value: safeStatus || 'Processing...', inline: false }
    )
    .setTimestamp();
}

export function jobComplete({ url, userId, results }) {
  const sorted = [...results].sort((a, b) => Number(a.bpw) - Number(b.bpw));
  const pushed = sorted.filter((r) => r.pushed).length;
  const reused = sorted.filter((r) => r.reused).length;
  const lines = sorted.map((r) => {
    const icon = r.reused ? '♻️' : r.pushed ? '✅' : '❌';
    const href = r.treeUrl || r.url;
    const link = href
      ? `[Link](${href})`
      : r.error
        ? `Error: ${truncate(r.error, 60)}`
        : 'No URL';
    return `${icon}  **${r.bpw} bpw** — ${r.duration} — ${link}`;
  });

  const FIELD_LIMIT = 1024;
  const resultFields = [];
  let chunk = [];
  let chunkLen = 0;
  for (const line of lines) {
    const sep = chunkLen === 0 ? 0 : 1;
    if (chunkLen + sep + line.length > FIELD_LIMIT && chunk.length > 0) {
      resultFields.push({
        name: resultFields.length === 0 ? 'Results' : '\u200b',
        value: chunk.join('\n'),
        inline: false,
      });
      chunk = [];
      chunkLen = 0;
    }
    chunk.push(line);
    chunkLen += (chunkLen === 0 ? 0 : 1) + line.length;
  }
  resultFields.push({
    name: resultFields.length === 0 ? 'Results' : '\u200b',
    value: chunk.join('\n') || 'No results',
    inline: false,
  });

  return new EmbedBuilder()
    .setTitle('✅ Quantization Complete')
    .setColor(COLORS.success)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `\`${truncate(url, 80)}\``, inline: false },
      {
        name: 'Summary',
        value: `Successful: **${pushed}/${sorted.length}**${reused ? ` • Reused: **${reused}**` : ''}`,
        inline: false,
      },
      ...resultFields
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
