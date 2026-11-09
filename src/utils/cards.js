// The four cards a /quant job shows: the request in the channel, then the
// progress, the result and a stop in its thread.
//
// They read as a status display, not a message: labelled facts, no narration,
// no emoji, and nothing that addresses whoever is reading. The bar colour is
// the state. Times that move are Discord timestamps (<t:..:R>), which tick in
// the client, so the bot only edits when something actually changed.
import { ActionRowBuilder, ButtonBuilder, ButtonStyle, EmbedBuilder } from 'discord.js';
import { truncate } from './format.js';
import { answerOf } from './svg.js';

export const COLOR = {
  request: 0x8957e5,
  running: 0x5865f2,
  done: 0x2ea043,
  partial: 0xd29922,
  failed: 0xda3633,
  idle: 0x888780,
};

const hf = (id) => `https://huggingface.co/${id}`;
const short = (id) => String(id).split('/').pop();
const rel = (ms) => `<t:${Math.floor(ms / 1000)}:R>`;
const inline = (name, value) => ({ name, value: String(value), inline: true });
const block = (name, value) => ({ name, value: truncate(String(value), 1024), inline: false });

export const money = (v) => `$${Number(v).toFixed(2)}`;

/** "NVIDIA RTX PRO 6000 Blackwell Server Edition" -> "RTX PRO 6000". */
export function gpuName(id) {
  return String(id || '')
    .replace(/^NVIDIA\s+/, '')
    .replace(/^GeForce\s+/, '')
    .replace(/\s+Blackwell.*$/, '')
    .replace(/\s+(PCIe|SXM\d?)(-\d+GB)?$/, '')
    .trim();
}

export function duration(secs) {
  const m = Math.round(secs / 60);
  if (m < 1) return 'under a minute';
  const h = Math.floor(m / 60);
  const mm = m % 60;
  const unit = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
  if (!h) return unit(mm, 'minute');
  return mm ? `${unit(h, 'hour')} ${unit(mm, 'minute')}` : unit(h, 'hour');
}

/** 20 cells of 5%, in inline code so the glyph widths hold on mobile. */
export function bar(pct) {
  const p = Math.max(0, Math.min(100, Math.round(pct || 0)));
  const n = Math.round(p / 5);
  return `\`${'█'.repeat(n)}${'░'.repeat(20 - n)}\` ${p}%`;
}

function frame(color, title, url, job, at) {
  const e = new EmbedBuilder().setColor(color).setTitle(truncate(title, 256)).setFooter({ text: `Job ${job.id.slice(0, 8)}` });
  if (url) e.setURL(url);
  if (at) e.setTimestamp(at);
  return e;
}

// The controller's failure line is already one sentence (see _explain in
// run_runpod_job.py). These are the few the bot makes up itself.
export function errorLine(err) {
  const s = String(err || '').trim();
  let m;
  if (!s) return 'Unknown error';
  if ((m = /^cli exit (\d+)$/.exec(s))) return `The controller exited with code ${m[1]}`;
  if ((m = /^cli killed \((\w+)\)$/.exec(s))) return `The controller was killed (${m[1]})`;
  if ((m = /run_runpod_job\.py exited (\d+) with no uploads/.exec(s))) return `The controller exited with code ${m[1]}`;
  if ((m = /run_runpod_job\.py killed by (\w+) with no uploads/.exec(s))) return `The controller was killed (${m[1]})`;
  if (s === 'no upload URL seen') return 'Finished without uploading anything';
  return truncate(s.replace(/\s+/g, ' '), 300);
}

/** A link to the job's thread, beside whatever buttons the card keeps. */
export function threadRow(threadUrl, keep = []) {
  return new ActionRowBuilder().addComponents(
    ...keep,
    new ButtonBuilder().setStyle(ButtonStyle.Link).setLabel('Open thread').setURL(threadUrl)
  );
}

// ── Request card ───────────────────────────────────────────────────────────

/**
 * The request, in the channel. One message for its whole life: posted
 * pending, then edited to approved, denied or started.
 *
 * job.card holds what the card says beyond the job itself, worked out once at
 * /quant time: { model, gpu, estimate, balance, options, community, note }.
 */
export function requestCard(job, { state = 'pending', by = null, at = null } = {}) {
  const c = job.card || {};
  const color = state === 'pending' ? COLOR.request : state === 'denied' ? COLOR.idle : COLOR.done;
  const e = frame(color, `Quantize ${short(job.modelId)}`, hf(job.modelId), job, job.createdAt);
  e.addFields(
    inline('Requested by', `<@${job.userId}>`),
    inline(job.variants.length > 1 ? 'Bitrates' : 'Bitrate', job.variants.join(', ')),
    inline('Balance', c.balance != null ? money(c.balance) : 'Unknown'),
  );
  if (c.model) e.addFields(block('Model', c.model));
  if (c.gpu) e.addFields(inline('GPU', c.gpu));
  if (c.estimate) e.addFields(inline('Estimate', c.estimate));
  if (state !== 'pending') {
    const who = state === 'started' ? job.userId : by;
    const label = { approved: 'Approved by', denied: 'Denied by', started: 'Started by' }[state];
    e.addFields(inline(label, `<@${who}>${at ? ` ${rel(at)}` : ''}`));
  }
  if (c.options) e.addFields(block('Options', c.options));
  if (c.community?.length) e.addFields(block('Already on HF', c.community.join('\n')));
  if (c.note && state === 'pending') e.addFields(block('Note', c.note));
  return e;
}

// ── Progress card ──────────────────────────────────────────────────────────

const SC_STAGE = { sc_trace: 'tracing', sc_rfn_probe: 'probing', sc_measure: 'measuring', sc_optimize: 'optimizing' };

/** The second line under a bitrate's bar. */
export function rowStatus(r) {
  switch (r.stage) {
    case 'Provisioning':
      return r.provStep >= 4 ? 'Starting the quant' : r.provStep >= 2 ? 'Starting the pod' : 'Waiting for a GPU';
    case 'Queued':
      return r.message || 'Waiting';
    case 'Downloading':
      return r.dl ? `Downloading, ${r.dl.done.toFixed(1)} of ${r.dl.total.toFixed(1)} GB` : 'Downloading';
    case 'Calibrating':
      return `Self-calibration, ${SC_STAGE[r.scStage] || 'starting'}`;
    case 'Quantizing':
      if (!r.layer) return 'Quantizing';
      return `Layer ${r.layer} of ${r.layers}` + (r.etaAt ? `, done ${rel(r.etaAt)}` : '');
    case 'Measuring KL':
      return 'Measuring KL divergence';
    case 'Testing':
      return 'Running the test prompt';
    case 'Uploading':
      return 'Uploading';
    case 'Retrying':
      return `Retrying, ${String(r.message || '').replace(/(\d+)\/(\d+)/, '$1 of $2')}`;
    case 'Complete':
      return 'Done';
    case 'Failed':
      return `Failed: ${errorLine(r.error || r.message)}`;
    default:
      return r.stage || 'Working';
  }
}

/** rows: [{ variant, label?, gpu?, overall, stage, ... }] in display order. */
export function progressCard(job, { rows, startedAt, spent = null, estimate = null }) {
  const e = frame(COLOR.running, `Quantizing ${short(job.modelId)}`, hf(job.modelId), job, startedAt)
    .setDescription(`Started ${rel(startedAt)}`);
  for (const r of rows) {
    const name = `${r.label || `${r.variant} bpw`}${r.gpu ? ` on ${gpuName(r.gpu)}` : ''}`;
    e.addFields(block(name, `${bar(r.overall)}\n${rowStatus(r)}`));
  }
  if (spent != null) e.addFields(inline('Cost so far', money(spent)));
  if (estimate) e.addFields(inline('Estimate', estimate));
  return e;
}

// ── Result card ────────────────────────────────────────────────────────────

function resultLine(r) {
  if (!r.pushed) return `Failed: ${errorLine(r.error)}`;
  const facts = [
    r.sizeGb != null ? `${Number(r.sizeGb).toFixed(1)} GB` : null,
    r.kl != null ? `KL ${Number(r.kl).toFixed(4)}` : null,
    `[repo](${r.treeUrl || r.url})`,
  ].filter(Boolean).join(', ');
  if (!r.sample) return facts;
  const answer = answerOf(r.sample);
  if (/<svg[\s\S]*?<\/svg>/i.test(answer)) return facts;
  if (!answer) return `${facts}\n> No answer within the token budget`;
  return `${facts}\n> ${truncate(answer.replace(/\s+/g, ' ').trim(), 200)}`;
}

const rowLabel = (r) => (r.drafter ? 'Drafter' : `${r.bpw ?? r.variant} bpw${r.donor ? ' donor' : ''}`);

/**
 * The job's last word, posted as a new message (the mention in its content is
 * what notifies the requester; an edit would not) in place of the progress card.
 * outcome is jobOutcome(); spend is jobSpend().
 */
export function resultCard(job, { results, outcome, spend = null, startedAt = null, endedAt = Date.now(),
                                  collectionUrl = null }) {
  const kind = outcome.allDone ? 'done' : outcome.status === 'completed' ? 'partial' : 'failed';
  const title = { done: 'Quantized', partial: 'Partly quantized', failed: 'Failed to quantize' }[kind];
  const e = frame(COLOR[kind], `${title} ${short(job.modelId)}`, collectionUrl || hf(job.modelId), job, endedAt);
  const failedAll = kind === 'failed' && results.filter((r) => !r.donor && !r.drafter).length === 1;
  if (failedAll) {
    e.addFields(block('Error', errorLine(results.find((r) => !r.donor && !r.drafter).error)));
  } else {
    for (const r of results) e.addFields(block(rowLabel(r), resultLine(r)));
  }
  e.addFields(...closingFields({ spend, startedAt, endedAt }));
  return e;
}

/** GPU, Time and Cost, the same three on every card that ends a job. */
function closingFields({ spend, startedAt, endedAt }) {
  const out = [];
  const cards = [...new Set((spend?.pods || []).map((p) => gpuName(p.gpu)))];
  if (cards.length) out.push(inline('GPU', cards.join(', ')));
  if (startedAt != null) out.push(inline('Time', duration((endedAt - startedAt) / 1000)));
  if (spend) out.push(inline('Cost', money(spend.cost)));
  return out;
}

export function stoppedCard(job, { where = null, spend = null, startedAt = null } = {}) {
  const at = job.stoppedAt || Date.now();
  const e = frame(COLOR.idle, `Stopped ${short(job.modelId)}`, hf(job.modelId), job, at);
  if (job.stoppedBy) e.addFields(block('Stopped by', `<@${job.stoppedBy}>${where ? ` ${where}` : ''}`));
  e.addFields(...closingFields({ spend, startedAt: startedAt ?? job.startedAt ?? job.approvedAt, endedAt: at }));
  return e;
}
