import { getLogger } from '../logger.js';
import config from '../config.js';
import * as hf from '../services/huggingface.js';
import * as embeds from '../utils/embeds.js';
import { findExl3Quants, recentBotQuants } from '../services/exl3Index.js';
import { toUserMessage } from '../errors/taxonomy.js';

const log = getLogger('cmd:quants');

const ours = (id) => config.LEDGER_ORGS.some((o) => id.toLowerCase().startsWith(`${o.toLowerCase()}/`));
const fmtN = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n));

function line(q) {
  const tags = [
    q.bpw != null ? `${q.bpw}` : null,
    q.kind,
    q.branches?.length ? `branches: ${q.branches.slice(0, 6).join(', ')}${q.branches.length > 6 ? ', …' : ''}` : null,
  ].filter(Boolean).join(' · ');
  const dl = q.downloads ? ` · ⬇ ${fmtN(q.downloads)}` : '';
  return `${ours(q.id) ? '🧱 ' : ''}[\`${q.id}\`](https://huggingface.co/${q.id})${tags ? ` · ${tags}` : ''}${dl}`;
}

// Not ephemeral: the point is that everyone can see what already exists before
// asking for it, or before spending six hours uploading it themselves.
export async function handleQuants(interaction) {
  await interaction.deferReply();
  const input = (interaction.options.getString('model') || '').trim();
  try {
    if (!input) {
      if (!config.LEDGER_ORGS.length) {
        return interaction.editReply({ embeds: [embeds.error('No ledger', 'HF_ORG is not set.')] });
      }
      const recent = await recentBotQuants(config.LEDGER_ORGS);
      return interaction.editReply({
        embeds: [embeds.info('🧱 Latest bot quants',
          recent.length ? recent.map(line).join('\n') : 'Nothing published yet.')],
      });
    }
    const modelId = hf.parseModelId(input);
    const found = await findExl3Quants(modelId);
    if (!found.length) {
      return interaction.editReply({
        embeds: [embeds.info(`No EXL3 quants of ${modelId}`,
          'Nothing on Hugging Face yet. `/quant` can make one.')],
      });
    }
    const shown = found.slice(0, 20).map(line);
    if (found.length > shown.length) shown.push(`…and ${found.length - shown.length} more`);
    return interaction.editReply({
      embeds: [embeds.info(`EXL3 quants of ${modelId} (${found.length})`, shown.join('\n').slice(0, 4000))],
    });
  } catch (err) {
    log.warn(`/quants ${input} failed: ${err.message}`);
    return interaction.editReply({ embeds: [embeds.error('Lookup failed', toUserMessage(err))] });
  }
}
