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
      { name: 'Model', value: `[\`${truncate(url, 80)}\`](https://huggingface.co/${url})`, inline: false },
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
      { name: 'Model', value: `[\`${truncate(url, 80)}\`](https://huggingface.co/${url})`, inline: false },
      { name: 'Stage', value: stage ?? 'Working…', inline: true },
      { name: 'Current', value: bpwText, inline: true },
      { name: 'Progress', value: bar, inline: false },
      { name: 'Status', value: truncate(message || 'Processing…', 200), inline: false }
    )
    .setTimestamp();
}

export function jobProgressParallel({ url, userId, variants, state }) {
  const ICON = {
    Complete: '✅', Failed: '❌', Retrying: '🔁', Uploading: '⬆️',
    Quantizing: '⚙️', Downloading: '⬇️', Provisioning: '⏳',
  };
  const done = variants.filter((v) => (state[v] || {}).stage === 'Complete').length;
  const anyFailed = variants.some((v) => (state[v] || {}).stage === 'Failed');

  const lines = variants.map((v) => {
    const s = state[v] || {};
    const stage = s.stage || 'Provisioning';
    const icon = ICON[stage] || '⏳';
    if (stage === 'Complete' && s.url) {
      return `${icon} \`${v}\` **done** · [repo](${s.url})`;
    }
    const pct = Math.round(s.overall ?? 0);
    const mins = s.startedAt ? Math.floor((Date.now() - s.startedAt) / 60000) : null;
    const elapsed = mins != null ? ` · ${mins}m` : '';
    const detail = s.message ? ` · ${truncate(s.message, 28)}` : '';
    return `${icon} \`${v}\` ${stage.toLowerCase()} · ${pct}%${elapsed}${detail}`;
  });

  const color = anyFailed ? COLORS.warning : done === variants.length ? COLORS.success : COLORS.pending;
  return new EmbedBuilder()
    .setTitle(`⚙️ Quantizing · ${done}/${variants.length} done`)
    .setColor(color)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `[\`${truncate(url, 80)}\`](https://huggingface.co/${url})`, inline: false },
      { name: 'Variants', value: lines.join('\n') || 'starting...', inline: false }
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
    // Join only the parts we actually have, so an empty duration (the RunPod
    // path doesn't set one) doesn't render a doubled "— —" separator.
    const parts = [`${icon}  **${r.bpw} bpw**`];
    if (r.duration) parts.push(r.duration);
    parts.push(link);
    let line = parts.join(' — ');
    // Optional smoke-test reply: an SVG is rendered + attached separately, so
    // just flag it; otherwise quote a one-line text preview under the variant.
    if (r.sample) {
      if (/<svg[\s\S]*?<\/svg>/i.test(r.sample)) {
        line += `\n> 🎨 SVG rendered below`;
      } else {
        const preview = truncate(r.sample.replace(/\s*\n+\s*/g, ' ').trim(), 280);
        if (preview) line += `\n> 💬 ${preview}`;
      }
    }
    return line;
  });

  const embed = new EmbedBuilder()
    .setTitle('✅ Quantization Complete')
    .setColor(COLORS.success)
    .setDescription(`Requested by <@${userId}>`)
    .addFields({ name: 'Model', value: `[\`${truncate(url, 80)}\`](https://huggingface.co/${url})`, inline: false });

  // Chunk result lines into fields of ≤1024 characters each
  const FIELD_LIMIT = 1024;
  let chunk = '';
  let fieldIndex = 0;
  for (const line of lines) {
    // Truncate a single line that exceeds the limit on its own
    const safeLine = line.length > FIELD_LIMIT ? truncate(line, FIELD_LIMIT) : line;
    const candidate = chunk ? `${chunk}\n${safeLine}` : safeLine;
    if (candidate.length > FIELD_LIMIT) {
      embed.addFields({ name: fieldIndex === 0 ? 'Results' : '\u200b', value: chunk || '\u200b', inline: false });
      chunk = safeLine;
      fieldIndex++;
    } else {
      chunk = candidate;
    }
  }
  embed.addFields({ name: fieldIndex === 0 ? 'Results' : '\u200b', value: chunk || 'No results', inline: false });

  return embed.setTimestamp();
}

export function jobFailed({ url, userId, error }) {
  const safeError = sanitizeErrorText(error);
  return new EmbedBuilder()
    .setTitle('❌ Quantization Failed')
    .setColor(COLORS.error)
    .setDescription(`Requested by <@${userId}>`)
    .addFields(
      { name: 'Model', value: `[\`${truncate(url, 80)}\`](https://huggingface.co/${url})`, inline: false },
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
