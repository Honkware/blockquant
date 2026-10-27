import { getLogger } from '../logger.js';
import {
  MessageFlags,
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
} from 'discord.js';
import { randomUUID } from 'node:crypto';
import config from '../config.js';
import * as hf from '../services/huggingface.js';
import * as queue from '../services/queue.js';
import * as quantizer from '../services/quantizer.js';
import * as db from '../services/db.js';
import * as embeds from '../utils/embeds.js';
import { sanitizeErrorText, toUserMessage } from '../errors/taxonomy.js';
import { defaultHeadBits } from '../utils/archSupport.js';
import { exl3RepoName } from '../utils/hfExl3.js';
import { isApiAvailable, submitJob, pollJob } from '../services/api-client.js';
import {
  getBalance, convertHours, calibrationHours,
} from '../services/runpod.js';
import { runViaCli, attachToCli, runVariantWithRetry, finalizeCollection } from '../services/runpodCli.js';
import { answerOf, extractSvg, renderSvgToPng } from '../utils/svg.js';
import { claimRunSlot, releaseRunSlot } from '../services/access.js';
import { findExl3Quants } from '../services/exl3Index.js';
import { costPreflightLine, quoteCost } from '../services/gpuQuote.js';
import { notifyAdmins, cardUrl } from '../services/approvalNotify.js';
import { stopButtons, stoppedEmbed } from './stop.js';

const log = getLogger('cmd:quant');

/**
 * Throttle progress embed edits to max once per `ms` (leading + trailing), so
 * a burst of progress events becomes a smooth ~once-per-4s cadence instead of
 * spamming Discord (which rate-limits message edits and would freeze the embed).
 * Exposes `.cancel()` to drop any pending trailing edit — call it before a final
 * embed write so a late trailing render can't clobber the "Complete" frame.
 */
function throttle(fn, ms = 4000) {
  let last = 0;
  let pending = null;
  const wrapped = (...args) => {
    const now = Date.now();
    if (now - last >= ms) {
      last = now;
      return fn(...args);
    }
    clearTimeout(pending);
    pending = setTimeout(
      () => {
        pending = null;
        last = Date.now();
        fn(...args);
      },
      ms - (now - last)
    );
  };
  wrapped.cancel = () => {
    clearTimeout(pending);
    pending = null;
  };
  return wrapped;
}

function approvalButtons(jobId) {
  return new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setCustomId(`bq:approve:${jobId}`)
      .setLabel('Approve')
      .setStyle(ButtonStyle.Success),
    new ButtonBuilder()
      .setCustomId(`bq:deny:${jobId}`)
      .setLabel('Deny')
      .setStyle(ButtonStyle.Danger)
  );
}

/**
 * /quant: validate the request, then post it for admin approval instead of
 * running immediately. The actual quantization runs from runApprovedJob once
 * an admin clicks Approve (see approval.js).
 */
// What the tower gets, in the terms the requester cares about -- a bitrate or
// an untouched copy. Which of the two came from the arch and which they typed
// is not interesting once it is settled.
function visionSummary(flight, bits) {
  if (!flight.hasVision) return 'none';
  return bits >= 1 && bits <= 8 ? `${bits} bpw` : 'fp16, copied';
}

// One pod per bitrate is right for plain quants: the work is independent and
// they finish in parallel. Under SC it is not -- sc_trace, sc_rfn_probe and
// sc_measure depend on the model, not the bitrate, and quant.py already shares
// them across every variant in a job (only sc_optimize is per-bitrate).
// Fanning out meant N pods each generating the same 512,000 calibration
// tokens: measured at ~25 min and ~16 min on a 0.8B, so three bitrates paid
// for that twice over and got nothing back.
// What a finished job achieved, judged on the variants that were asked for.
// A self-calibrated job's own donor lands in `results` too, and on the first
// 27B run it was the only thing that did: the 3.6bpw SC variant died in
// sc_measure, the 4.0 donor pushed, the note read "partial upload success
// (1/2)" without saying which, and the job was recorded as completed.
// The donor is flagged rather than told apart by bitrate: a 4.0 SC target gets
// a 4.0 donor, and by label alone the donor would pass for the SC variant.
export function jobOutcome(variants, results) {
  const asked = new Set(variants.map(String));
  const label = (r) => String(r.variant ?? r.bpw);
  const wanted = (r) => !r.donor && !r.drafter && asked.has(label(r));
  const requested = results.filter(wanted);
  const failed = requested.filter((r) => !r.pushed);
  const extras = results.filter((r) => !wanted(r) && r.pushed);
  return {
    status: requested.length && failed.length === requested.length ? 'failed' : 'completed',
    allDone: requested.length > 0 && failed.length === 0,
    failed: failed.map(label),
    reason: failed.map((r) => r.error).find(Boolean) || null,
    extras: extras.map((r) => r.url || label(r)),
  };
}

/** What the job's pods cost in total, and on what, from the [cost] lines. */
export function jobSpend(results) {
  const pods = (results || []).flatMap((r) => r.podCost?.pods || []);
  if (!pods.length) return null;
  const cards = [...new Set(pods.map((p) => `${p.gpu} (${p.cloud.toLowerCase()})`))].join(', ');
  return { cost: pods.reduce((a, p) => a + p.cost, 0), cards, pods };
}

// The least a donor may be. sc_trace generates the whole calibration set
// through it, so its errors are what the quant gets calibrated toward. This
// used to be the target rounded up, floored at 4.0 for speed -- a 3.6 SC was
// traced through a 4.0 -- while turboderp's SC branches and ezexl3 both trace
// through 6bpw or better, and sharkwipf measured a 6bpw teacher against a full
// bf16 one on Gemma 4 26B-A4B and found no difference outside noise. Nothing
// says 4.0 is that close.
export const DONOR_FLOOR_BPW = 6;

// What to quantize as a donor when none exists: at least the floor, at least
// the target (sc_rfn_probe walks its module tree against it), and rounded up to
// a half-bit so it is a sensible artifact in its own right. A 6.0 is also the
// quant people ask for most once there is an SC family.
export function donorBitrateFor(targetBpw) {
  const t = Number(targetBpw);
  if (!Number.isFinite(t)) throw new TypeError(`Invalid target bpw: ${targetBpw}`);
  const half = Math.ceil(t * 2) / 2;
  return Math.min(8, Math.max(DONOR_FLOOR_BPW, half)).toFixed(1);
}

// The sets #quant-bot keeps asking for by hand: 2.05-ish through 3.5 for
// 16-24 GB cards, and the usual 4-6. Anything else is still typed out.
export const BPW_PRESETS = {
  low: ['2.0', '2.5', '3.0', '3.5'],
  mid: ['4.0', '4.5', '5.0'],
  high: ['6.0', '8.0'],
  ladder: ['2.0', '2.5', '3.0', '3.5', '4.0', '5.0', '6.0'],
};

/** Presets expanded in place, numbers left as typed, duplicates dropped. */
export function expandBpw(input) {
  const out = [];
  for (const tok of String(input).split(',').map((t) => t.trim().toLowerCase()).filter(Boolean)) {
    out.push(...(BPW_PRESETS[tok] ?? [tok]));
  }
  return [...new Set(out)];
}

// How long a pod may run before the controller kills it. The launcher default
// is 8h, which is a conversion with room to spare and not an SC ladder: those
// run every bitrate on one pod after the calibration, and a 27B low ladder is
// ~3.5h of calibration plus four conversions. Killed at 8h it has billed the
// lot and published part. This is the high estimate with a quarter on top --
// the backstop for a hung pod, not a budget.
export function podRuntimeHours({ sc = false, scLite = false, sizeGb = 0, isMoe = false, nVariants = 1 }) {
  const conv = convertHours(sizeGb, isMoe).high;
  const h = sc ? calibrationHours(sizeGb, { traceOnly: scLite }).high + nVariants * conv : conv;
  return Math.min(24, Math.max(8, Math.ceil(h * 1.25)));
}

export function podGroups(variants, sc) {
  return sc ? [variants.slice()] : variants.map((v) => [v]);
}

export async function handleQuant(interaction) {
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });

  // Every early return below is a reply only the requester sees, and none was
  // logged. From 09-18 to 10-25 five /quant's ended between preflight and the
  // job write with nothing on our side to say why, and "Quant bot is dead :("
  // in #quant-bot was the only report of it. One line per reply says where
  // each request ended.
  const editReply = interaction.editReply.bind(interaction);
  interaction.editReply = (payload) => {
    const title = payload?.embeds?.[0]?.data?.title;
    if (title) log.info(`/quant ${interaction.options.getString('url')} by ${interaction.user.username}: ${title}`);
    return editReply(payload);
  };

  const urlInput = interaction.options.getString('url', true);
  const bpwInput = interaction.options.getString('bpw', true);
  // Optional smoke-test prompt: run on each finished quant so the requester
  // sees a real reply. Capped so it stays a quick check, not a chat session.
  const testPrompt = (interaction.options.getString('prompt') || '').trim().slice(0, 1000) || null;
  // Trellis codebook. Recorded in the quant itself, so it decides which
  // ExLlamaV3 builds can read what we publish; config.CODEBOOK is the default.
  const codebook = interaction.options.getString('codebook') || config.CODEBOOK;
  // sc_mode implies sc: nobody picks a calibration depth for a quant they did
  // not want calibrated. trace is turboderp's advice for big models -- the
  // model's own sampled output as calibration plus -hq 2, without measuring.
  const scMode = interaction.options.getString('sc_mode');
  const sc = (interaction.options.getBoolean('sc') ?? false) || !!scMode;
  const scLite = sc && scMode === 'trace';
  const mtpBits = interaction.options.getNumber('mtp_bits');
  const ngramBits = interaction.options.getInteger('ngram_bits');
  // sc_mode:trace without it means -hq 2, and the name has to say so up front.
  const hqOpt = interaction.options.getInteger('hq');
  const hq = hqOpt ?? (scLite ? 2 : null);
  const draftInput = (interaction.options.getString('draft_model') || '').trim();
  // Omitted means exllamav3 decides, and its default is NOT a flat 6: it is the
  // tower's own declared default_vision_bits, 6 on the arches with a validated
  // tower and 16 -- copy it whole -- everywhere else. 1.4.9 is where that split
  // appeared; older builds copied every tower, so the same request gives a
  // different artifact now, and 16 is the way back. preflight resolves which
  // off the generated table, so the name can say -V6 without asking.
  // exllamav3's converter runs one worker thread per device and tile-splits
  // large tensors across them, and it shows: an 8B took 31.0 min on one 3090
  // and 17.9 on two, a 27B ~95 min on one A100 and 57.7 on two. Not linear,
  // so a second card is ~15-20% more per job for ~40% less wait. Opt-in.
  const gpuCount = interaction.options.getInteger('gpu_count');
  const visionBits = interaction.options.getInteger('vision_bits');
  // Only for repos that keep their model in a subdirectory. Preflight picks it
  // on its own when there is exactly one unquantized candidate, so this is the
  // tie-breaker, not the normal path.
  const subfolderOpt = (interaction.options.getString('subfolder') || '').trim().replace(/^\/+|\/+$/g, '');
  // Omitted means exllamav3 decides.
  const headBits = interaction.options.getInteger('head_bits');
  const userId = interaction.user.id;

  // The request is intentionally just model + bpw; the rest is exllamav3's own
  // defaults unless the requester says otherwise.
  const category = 'General';
  const format = 'exl3';
  const provider = 'runpod';

  // ── Parse & validate variant list ─────────────────────────────────────────
  let variants = expandBpw(bpwInput);
  if (variants.length === 0) {
    return interaction.editReply({
      embeds: [embeds.error('Invalid Variants', 'Provide comma-separated values, e.g. `3.0,4.0,5.0`')],
    });
  }
  // A typo used to be filtered out as NaN and the rest ran without it.
  const unknown = variants.filter((t) => Number.isNaN(Number(t)));
  if (unknown.length) {
    return interaction.editReply({
      embeds: [
        embeds.error(
          'Invalid BPW',
          `Not a number or a preset: ${unknown.join(', ')}. Presets: ` +
            Object.entries(BPW_PRESETS).map(([k, v]) => `\`${k}\` (${v.join(', ')})`).join(', ')
        ),
      ],
    });
  }

  // By value, so 4 and 4.0 are one variant, not two pods.
  const bpws = [...new Set(variants.map((s) => parseFloat(s)).filter((n) => !isNaN(n)))];
  if (bpws.length === 0) {
    return interaction.editReply({
      embeds: [embeds.error('Invalid BPW', 'Provide comma-separated numbers, e.g. `3.0,4.0,5.0`')],
    });
  }
  const invalid = bpws.filter((b) => b < 1 || b > 8);
  if (invalid.length) {
    return interaction.editReply({
      embeds: [
        embeds.error('BPW Out of Range', `Values must be between 1-8. Got: ${invalid.join(', ')}`),
      ],
    });
  }
  // Normalize to one-decimal form so repo names are consistent
  // (3 -> "3.0", matching the -exl3-3.0bpw convention).
  variants = bpws.map((b) => (Number.isInteger(b) ? b.toFixed(1) : String(b)));

  // exllamav3 takes 1-8, or 16 for an unquantized head; 9-15 are not lattice
  // sizes it has codebooks for. Discord caps the range at 1-16, so only the
  // hole in the middle needs catching.
  if (visionBits !== null && visionBits > 8 && visionBits !== 16) {
    return interaction.editReply({
      embeds: [
        embeds.error('Invalid vision bits', 'Vision bits must be 1-8, or 16 to copy the tower unquantized.'),
      ],
    });
  }
  if (headBits !== null && headBits > 8 && headBits !== 16) {
    return interaction.editReply({
      embeds: [
        embeds.error('Invalid head bits', 'Head bits must be 1-8, or 16 to leave the head unquantized.'),
      ],
    });
  }
  // -mb takes whole bits, the three half-integer mul1 rates, or 16.
  if (mtpBits !== null && !(Number.isInteger(mtpBits) && (mtpBits <= 8 || mtpBits === 16))
      && ![1.5, 2.5, 3.5].includes(mtpBits)) {
    return interaction.editReply({
      embeds: [embeds.error('Invalid MTP bits', 'MTP bits must be 1-8, 1.5, 2.5, 3.5, or 16.')],
    });
  }

  // A plain quant leaves the defaults out of its name -- that is what a bare
  // `-exl3-4.0bpw` means. SC states everything, because the recipe picks head
  // bits per budget and turboderp's SC branches name them (H3 through H6 on one
  // model), so a bare bitrate would not identify the artifact.
  if (sc && headBits === null) {
    return interaction.editReply({
      embeds: [
        embeds.error(
          'Self-calibration needs head bits',
          'The recipe carries head bits and they go in the published name, so there is ' +
            'no default to fall back on. Pass `head_bits` (1-8, or 16).'
        ),
      ],
    });
  }

  // ── Parse model URL ───────────────────────────────────────────────────────
  let modelId;
  try {
    modelId = hf.parseModelId(urlInput);
  } catch {
    return interaction.editReply({
      embeds: [embeds.error('Invalid Model', 'Provide a valid HuggingFace URL or `org/model` ID.')],
    });
  }

  // ── Pre-flight: validate HF token + model exists ──────────────────────────
  await interaction.editReply({
    embeds: [embeds.info('Running pre-flight checks...', 'Verifying HuggingFace credentials and model access...')],
  });

  let flight;
  try {
    flight = await hf.preflight(urlInput);
  } catch (err) {
    log.error('Preflight failed', { error: err.message, userId });
    return interaction.editReply({ embeds: [embeds.error('Pre-flight Failed', toUserMessage(err))] });
  }

  if (!flight.modelExists) {
    return interaction.editReply({
      embeds: [embeds.error('Model Not Found', `\`${modelId}\` does not exist or is not accessible.`)],
    });
  }
  // Under SC the name carries the tower state, so it has to be settled before
  // the job runs -- inspectUploadRepo checks the final name up front. This used
  // to make the requester state it by hand; preflight resolves the arch default
  // off the generated table now, so it names itself.
  const effVisionBits = visionBits ?? (flight.hasVision ? flight.visionBitsAuto : null);
  // SC has to pin head bits rather than leave them to the converter: the name
  // states them, and run_runpod_job refuses --sc without them for that reason.
  // Resolving this only for the name left the flag unsent and every SC job from
  // Discord died at launch unless the requester happened to type a number.
  const effHeadBits = sc ? headBits ?? defaultHeadBits() : headBits;
  // Pinning the tower to the number the arch already picks says nothing the
  // plain name does not, and would publish a -V6 beside an identical unsuffixed
  // build. Drop it back to auto so both spellings land on one repo.
  const towerBits = visionBits !== null && visionBits === flight.visionBitsAuto ? null : visionBits;

  // The drafter has to be one exllamav3 converts (DFlash2DraftModel and kin),
  // which preflight checks against the same table. Found out here, not after
  // the model's own quants have been paid for.
  let draftModel = null;
  if (draftInput) {
    try {
      draftModel = hf.parseModelId(draftInput);
      await hf.preflight(draftModel);
    } catch (err) {
      return interaction.editReply({
        embeds: [embeds.error('Drafter not usable', `\`${draftInput}\`: ${toUserMessage(err)}`)],
      });
    }
  }

  // Preflight resolves this when the repo has one obvious source; an explicit
  // option wins so someone can quant the FP8 copy if they really mean to.
  const subfolder = subfolderOpt || flight.subfolder || null;
  if (!flight.canWrite) {
    return interaction.editReply({
      embeds: [embeds.error('Token Error', 'The HF token does not have write permissions. Update `HF_TOKEN` in .env.')],
    });
  }

  // ── Runtime setup checks (local EXL3 only) ────────────────────────────────
  if (format === 'exl3' && provider === 'local') {
    try {
      await quantizer.validateSetup();
    } catch (err) {
      log.error('Quantizer setup check failed', { error: err.message, userId });
      return interaction.editReply({
        embeds: [embeds.error('Quantizer Setup Error', toUserMessage(err) || err.message)],
      });
    }
  }

  // ── Pre-check upload targets for idempotency (EXL3 local only) ────────────
  const modelName = modelId.split('/').pop();
  const precheckedRepos = {};
  const alreadyUploaded = [];
  if (format === 'exl3') {
    try {
      for (const bpw of bpws) {
        const repoName = exl3RepoName(modelName, bpw, {
          sc,
          scLite,
          headBits: sc ? effHeadBits : null,
          visionBits: sc ? effVisionBits : towerBits,
          mtpBits,
          ngramBits,
          hq,
        });
        const state = await hf.inspectUploadRepo(repoName, {
          sourceModel: modelId,
          bpw,
          hasVision: flight.hasVision,
          quantOptions: { headBits: effHeadBits, visionBits: towerBits },
        });
        precheckedRepos[String(bpw)] = state;
        // config_missing is a repo whose config.json could not be read -- an
        // upload still in flight, or weights that never landed. That is not a
        // settings conflict, and blocking on it would leave no way to retry.
        if (state.exists && state.settingsMatch === false && state.reason !== 'config_missing') {
          return interaction.editReply({
            embeds: [
              embeds.error(
                'Existing Repo Conflict',
                // vision_tower_differs is the one worth spelling out. The plain
                // name means "the tower default of the day", and that default
                // moved at 1.4.9 from copying the tower whole to quantizing a
                // validated one -- so the existing repo and a re-run today are
                // different weights under one name. Naming every VL repo -V{n}
                // to avoid this would be noise on the many to serve the few, so
                // say it here instead, where it actually bites.
                String(state.reason).includes('vision_tower_differs')
                  ? `\`${state.repoId}\` has an fp16 vision tower — it predates tower ` +
                    'quantization, so a re-run now would be different weights under the same ' +
                    'name. Pass `vision_bits:16` to rebuild it as-is, or `vision_bits:6` to ' +
                    'quantize the tower and publish alongside it as `-V6`.'
                  : `\`${state.repoId}\` was quantized with different settings (${state.reason ?? 'settings mismatch'}).`
              ),
            ],
          });
        }
        if (state.exists && state.settingsMatch) {
          alreadyUploaded.push(`${bpw} bpw`);
        }
      }
    } catch (err) {
      return interaction.editReply({ embeds: [embeds.error('Repo Pre-check Failed', toUserMessage(err))] });
    }

    if (alreadyUploaded.length === bpws.length) {
      return interaction.editReply({
        embeds: [
          embeds.success('Already Quantized', `Matching uploads already exist for all requested BPWs: ${alreadyUploaded.join(', ')}`),
        ],
      });
    }
  }

  // ── Self-calibration donor ────────────────────────────────────────────────
  // The trace is generated by a quant OF THIS MODEL: sc_rfn_probe walks both
  // module trees together and sc_measure runs that trace through this model's
  // embedding, so a quant of anything else fails in two different ways. Look
  // for one already on the hub rather than making the requester find it, and
  // refuse clearly when there is none -- a plain quant has to exist first.
  let donorRepo = null;
  let donorBpw = null;
  if (sc) {
    const wanted = Math.max(...bpws);
    try {
      // An existing 4.0 no longer qualifies; the job builds a 6.0 instead.
      donorRepo = await hf.findDonorQuant(modelName, Math.max(DONOR_FLOOR_BPW, wanted));
    } catch (err) {
      return interaction.editReply({
        embeds: [embeds.error('Could not look for a donor', toUserMessage(err))],
      });
    }
    // No donor yet: build one first rather than making the requester run a
    // separate command and watch for it to finish. The donor requirement is an
    // implementation detail of self-calibration, not something to hand back.
    if (!donorRepo) {
      donorBpw = donorBitrateFor(wanted);
      log.info(`SC has no donor for ${modelId}; will build ${donorBpw}bpw first`);
    } else {
      log.info(`SC donor for ${modelId}: ${donorRepo}`);
    }
  }

  // ── Cost gate: refuse if RunPod can't even afford the optimistic estimate ─
  // The work to actually do is the variants minus the ones already uploaded.
  const billable = Math.max(1, variants.length - alreadyUploaded.length);
  // One set of options for the gate and the quote. The gate used to price a
  // plain quant of unknown size, so an SC job on a 27B cleared it at $1.50.
  const costOpts = {
    sc, scLite, sizeGb: flight.sizeGb, needsDonor: !!donorBpw, isMoe: !!flight.isMoe,
    gpuCount: gpuCount || 1,
  };
  if (provider === 'runpod') {
    const bal = await getBalance();
    const est = await quoteCost(billable, costOpts);
    if (bal && est.low > bal.balance) {
      return interaction.editReply({
        embeds: [
          embeds.error(
            'Insufficient RunPod balance',
            `Estimated cost ~$${est.low.toFixed(0)}-${est.high.toFixed(0)} for ${billable} variant(s), ` +
              `but the RunPod balance is only $${bal.balance.toFixed(2)}. Top up before requesting.`
          ),
        ],
      });
    }
  }

  // ── Persist the request, then run it or post it for approval ──────────────
  // A quanter's job starts here instead of waiting on a button, so the one-job
  // cap in claimRunSlot is the only thing between one of them and a wallet's
  // worth of pods.
  const slot = await claimRunSlot({ userId, member: interaction.member });
  // Paused: a quanter's job waits for approval like anyone's, rather than
  // starting on its own while an admin has said nothing new should.
  const paused = !!(await db.loadSettings().catch(() => ({}))).paused;
  if (paused && slot.quanter) {
    releaseRunSlot(userId);
    slot.quanter = false;
  }
  if (!slot.ok) {
    return interaction.editReply({
      embeds: [
        embeds.error(
          'One job at a time',
          slot.running
            ? `Your \`${slot.running.modelId}\` job is still running. Stop it with \`/stop\`, or wait for it to finish.`
            : 'Another request of yours is still being submitted.'
        ),
      ],
    });
  }

  const jobId = randomUUID();
  try {
    await db.upsertJob({
      id: jobId,
      status: slot.quanter ? db.JOB_STATUS.queued : db.JOB_STATUS.pending_approval,
      createdAt: Date.now(),
      userId,
      username: interaction.user.username,
      modelId,
      url: urlInput,
      format,
      variants,
      bpws,
      testPrompt,
      codebook,
      sc,
      scLite,
      mtpBits,
      ngramBits,
      hq,
      draftModel,
      donorRepo,
      donorBpw,
      visionBits: towerBits,
      headBits: effHeadBits,
      gpuCount,
      // For sizing each pod's runtime limit at launch.
      sizeGb: flight.sizeGb || 0,
      isMoe: !!flight.isMoe,
      subfolder,
      categories: [category],
      provider,
      precheckedRepos,
      alreadyUploaded,
      channelId: interaction.channelId,
      threadId: null,
      progressMessageId: null,
      partialResults: [],
    });
  } finally {
    // The scan can see the job from here on, so the claim has done its job.
    releaseRunSlot(userId);
  }

  const costLine = provider === 'runpod'
    ? await costPreflightLine(variants.length, costOpts)
    : '';
  // Kept on the job so the completion can put what it cost beside it.
  if (provider === 'runpod') {
    const q = await quoteCost(variants.length, costOpts).catch(() => null);
    if (q) {
      await db.patchJob(jobId, { quote: { low: q.low, high: q.high, live: q.live, card: q.best?.id ?? null } })
        .catch(() => {});
    }
  }

  // Someone else may already have made it. Said on the card so an admin is not
  // paying for a copy by accident -- #quant-bot has had two people uploading the
  // same model the same week. Best-effort: a slow HF search costs one line.
  const ourOrgs = config.LEDGER_ORGS.map((o) => `${o.toLowerCase()}/`);
  const community = await Promise.race([
    findExl3Quants(modelId),
    new Promise((r) => setTimeout(() => r([]), 5000)),
  ]).then((qs) => qs.filter((q) => q.bpw != null
      && !ourOrgs.some((o) => q.id.toLowerCase().startsWith(o))
      && bpws.some((b) => Math.abs(q.bpw - b) < 0.051)))
    .catch(() => []);
  const pinned = [
    draftModel ? `drafter ${draftModel}` : '',
    mtpBits != null ? `-mb ${mtpBits}` : '',
    ngramBits != null ? `-ngb ${ngramBits}` : '',
    hq ? `-hq ${hq}` : '',
  ].filter(Boolean).join(' · ');

  const requestEmbed = embeds.info(
    slot.quanter ? 'Quantization request · starting' : 'Quantization request · awaiting approval',
    [
      `**Model:** [\`${modelId}\`](https://huggingface.co/${modelId})`,
      `**Variants:** ${variants.join(', ')}  ·  **Format:** ${format.toUpperCase()}`,
      subfolder ? `**Subfolder:** \`${subfolder}\`` : '',
      `**Head bits:** ${effHeadBits ?? defaultHeadBits()}  ·  **Codebook:** \`${codebook}\`  ·  **Vision:** ${visionSummary(flight, effVisionBits)}`,
      sc ? `**Self-calibration:** ${scLite ? 'trace only (no measurement)' : 'full'}  ·  **Donor:** ${
        donorRepo ? `\`${donorRepo}\`` : `builds ${donorBpw}bpw first`}` : '',
      pinned ? `**Pinned:** \`${pinned}\`` : '',
      // Measured on 3090s and A100s: two cards convert ~1.7x faster for
      // ~15-20% more per job. SC's trace and measurement use one card, so
      // there the extra cards only speed up the conversion at the end.
      (gpuCount || 1) > 1
        ? `**GPUs:** ${gpuCount}, conversion ~${(gpuCount ** 0.79).toFixed(1)}x faster, `
          + `~${Math.round((gpuCount ** 0.21 - 1) * 100)}% more per job${sc ? ' (calibration still runs on one)' : ''}`
        : '',
      `**Provider:** ${provider}`,
      costLine,
      `**Requested by:** <@${userId}>`,
      alreadyUploaded.length ? `**Reuses existing:** ${alreadyUploaded.join(', ')}` : '',
      community.length
        ? `**Already on HF from others:** ${community.slice(0, 3)
          .map((q) => `[${q.id}](https://huggingface.co/${q.id}) (${q.bpw}${q.kind ? ` ${q.kind}` : ''})`).join(', ')}`
          + (community.length > 3 ? ` +${community.length - 3} more, see \`/quants\`` : '')
        : '',
      `**Job:** \`${jobId.slice(0, 8)}\``,
      '',
      slot.quanter
        ? 'Starting now — the requester holds the quanter role. Stop it with the button or `/stop`.'
        : paused ? 'The bot is paused. An admin can approve this after `/resume`.'
          : 'An admin must approve before this runs.',
    ].filter(Boolean).join('\n')
  );

  // The card is posted either way, so a quanter's unapproved job is still
  // something an admin can see and stop.
  const card = await interaction.channel.send({
    embeds: [requestEmbed],
    components: [slot.quanter ? stopButtons(jobId) : approvalButtons(jobId)],
  });
  await db.patchJob(jobId, { cardMessageId: card.id });
  if (!slot.quanter) {
    notifyAdmins(interaction.client,
      `\`${modelId}\` ${variants.join(', ')}${sc ? (scLite ? ' SC (trace)' : ' SC') : ''} from `
        + `${interaction.user.username} is awaiting approval${costLine ? ` (${costLine.replace(/\*\*/g, '')})` : ''}: `
        + `${cardUrl({ channelId: interaction.channelId, cardMessageId: card.id })}`).catch(() => {});
  }

  if (slot.quanter) {
    await interaction.editReply({
      embeds: [
        embeds.success(
          'Request started',
          'Your quantization is starting — watch the thread. `/stop` ends it and terminates its pods.'
        ),
      ],
    });
    const job = (await db.loadJobs())[jobId];
    try {
      await runApprovedJob({ interaction, job });
    } catch (err) {
      log.error(`Failed to start quanter job ${jobId}`, { error: err.message });
      await db.patchJob(jobId, {
        status: db.JOB_STATUS.failed,
        failedAt: Date.now(),
        error: sanitizeErrorText(err.message),
      });
      await interaction
        .followUp({
          content: `Could not start the job: ${toUserMessage(err) || err.message}`,
          flags: MessageFlags.Ephemeral,
        })
        .catch(() => {});
    }
    return;
  }

  await interaction.editReply({
    embeds: [
      embeds.success(
        'Request submitted',
        'Your quantization request was posted for admin approval. You will be pinged in a thread once it starts.'
      ),
    ],
  });
}

/**
 * Run a previously-approved job. Called from approval.js with the approving
 * admin's button interaction and the stored pending-approval job record. Owns
 * the progress thread + live embed + execution, exactly as before, minus the
 * EXP accounting.
 */
/**
 * Pick a job back up after a bot restart, driving the embed it already has.
 *
 * runApprovedJob touches the interaction in exactly two places -- the channel
 * to open a thread in, and approvedBy -- and the second only runs when this is
 * not a resume. So a shim carrying the client is all it needs, and the whole
 * completion path (cards, collection, the thread's final message) stays shared
 * with a normal run rather than being reimplemented here and drifting.
 */
export function resumeJob({ client, job, resumeFrom }) {
  return runApprovedJob({ interaction: { channel: null, client }, job, resumeFrom });
}

export async function runApprovedJob({ interaction, job, resumeFrom = null }) {
  const {
    id: jobId,
    userId,
    username,
    modelId,
    url: urlInput,
    format,
    variants,
    bpws,
    testPrompt = null,
    codebook = config.CODEBOOK,
    visionBits = null,
    // Older records predate the option; null keeps exllamav3's default.
    headBits = null,
    gpuCount = null,
    subfolder = null,
    sc = false,
    scLite = false,
    mtpBits = null,
    ngramBits = null,
    hq = null,
    draftModel = null,
    donorRepo = null,
    donorBpw = null,
    sizeGb = 0,
    isMoe = false,
    categories,
    provider,
    precheckedRepos = {},
  } = job;
  const category = (categories && categories[0]) || 'General';

  const channel =
    interaction.channel ||
    (await interaction.client.channels.fetch(job.channelId).catch(() => null));
  if (!channel || !channel.threads) {
    throw new Error('Could not resolve a channel to open the progress thread in.');
  }
  // Resuming: the thread and the embed outlived the bot, so edit the message
  // that is already there. Opening a second thread would strand the first one
  // mid-progress with nothing ever finishing it.
  const thread = resumeFrom
    ? await channel.threads.fetch(job.threadId).catch(() => null)
    : await channel.threads.create({
        name: `⚡ ${modelId.split('/').pop()} [${username || 'request'}]`,
        autoArchiveDuration: 1440,
      });
  if (!thread) throw new Error(`Progress thread ${job.threadId} is gone; nothing to resume into.`);

  const progressMsg = resumeFrom
    ? await thread.messages.fetch(job.progressMessageId)
    : await thread.send({
        embeds: [embeds.jobQueued({ url: modelId, bpws: variants, categories: [category], userId })],
        components: [stopButtons(jobId)],
      });

  await db.patchJob(jobId, resumeFrom
    ? { status: db.JOB_STATUS.running, resumedAt: Date.now() }
    : {
        status: db.JOB_STATUS.queued,
        approvedAt: Date.now(),
        approvedBy: interaction.user.id,
        threadId: thread.id,
        progressMessageId: progressMsg.id,
      });

  const updateEmbed = throttle(async (data) => {
    try {
      await progressMsg.edit({ embeds: [embeds.jobProgress({ url: modelId, userId, ...data })] });
    } catch (err) {
      log.debug(`Failed to update embed: ${err.message}`);
    }
  });

  // Assigned by the runpod path below; declared here so handleComplete/
  // handleError can cancel its pending render before writing the final embed.
  let renderParallel = null;

  async function handleComplete(results) {
    try {
      // Drop any pending throttled progress render so it can't fire after this
      // and overwrite the final "Complete" embed with a stale frame.
      renderParallel?.cancel();
      await progressMsg.edit({
        embeds: [embeds.jobComplete({ url: modelId, userId, results })],
        components: [],
      });
      const outcome = jobOutcome(variants, results);
      let completionNote;
      if (outcome.allDone) {
        completionNote = `<@${userId}> Your quantization is done! 🎉`;
      } else if (outcome.status === 'completed') {
        completionNote = `<@${userId}> Partly done -- ${outcome.failed.join(', ')} bpw failed.`;
      } else {
        completionNote = `<@${userId}> The quantization you asked for failed (${outcome.failed.join(', ')} bpw).`;
      }
      if (outcome.reason) completionNote += ` Reason: ${String(outcome.reason).slice(0, 300)}`;
      if (!outcome.allDone) {
        // A self-calibrated retry finds the donor on the hub and the finished
        // stages in the cache, so it does not start over.
        completionNote += ' Running the same command again picks up what finished.';
      }
      if (outcome.extras.length) {
        completionNote += `\nAlso published: ${outcome.extras.join(', ')}`;
      }
      const actual = jobSpend(results);
      if (actual) {
        const q = job.quote;
        completionNote += `\nRunPod: $${actual.cost.toFixed(2)} on ${actual.cards}`
          + (q ? ` (quoted ~$${q.low.toFixed(2)}${q.high > q.low * 1.15 ? `, up to $${q.high.toFixed(2)}` : ''})` : '');
      }
      await thread.send(completionNote);

      // Catbench: if a smoke-test reply was an SVG (e.g. "draw a kitten"),
      // render it to PNG and post one per bpw so they can be compared. Static
      // render, best-effort: a malformed/truncated SVG from the model is skipped.
      try {
        const svgFiles = [];
        for (const r of results) {
          const svg = extractSvg(answerOf(r.sample));
          if (!svg) continue;
          try {
            svgFiles.push({ attachment: renderSvgToPng(svg), name: `sample-${r.bpw}bpw.png` });
          } catch (e) {
            log.debug(`SVG render failed for ${r.bpw} bpw: ${e.message}`);
          }
        }
        if (svgFiles.length) {
          await thread.send({ content: '🎨 Rendered SVG replies:', files: svgFiles.slice(0, 10) });
        }
      } catch (e) {
        log.warn(`SVG attach skipped: ${e.message}`);
      }

      const models = await db.loadModels();
      models[modelId] = {
        url: modelId,
        categories: [category],
        authors: [userId],
        quantizedBPWs: results.filter((r) => r.pushed && !r.drafter).map((r) => r.bpw ?? r.variant),
        results,
        completedAt: Date.now(),
      };
      await db.saveModels(models);

      await db.patchJob(jobId, {
        status: db.JOB_STATUS[outcome.status],
        // Against job.quote, to refit the quote from what jobs really cost.
        actualCost: jobSpend(results),
        completedAt: Date.now(),
        partialResults: results,
      });
    } catch (err) {
      log.error('Completion callback error', { error: err.message });
    }

    // Re-sync ALL of this model's EXL3 cards + collection, so variants done in
    // separate runs (or earlier) cross-reference each other. Run it for any
    // exl3 job regardless of what THIS run's result tracking says: the finalizer
    // discovers the actual uploaded repos from HF, so it cross-links whatever
    // actually landed even when a controller died/was killed but the quant had
    // already uploaded. Hoisted OUT of the try above (and the thread sends made
    // non-fatal) so a dead thread / deleted progress message -- common when a
    // job is force-cancelled, which is exactly when the cross-link was getting
    // skipped -- can't stop the cross-link from running. Best-effort.
    if (format === 'exl3') {
      try {
        await thread.send('Cross-linking model cards + collection...').catch(() => {});
        const fin = await finalizeCollection({ modelId, hfOrg: config.HF_ORG });
        await thread.send(
          fin.ok
            ? `Cards cross-linked${fin.collectionUrl ? ` · [collection](${fin.collectionUrl})` : ''}.`
            : 'Cross-link/collection step had an issue (any uploaded quants are still on HF).'
        ).catch(() => {});
      } catch (e) {
        log.debug(`finalize step failed: ${e.message}`);
      }
    }
  }

  async function handleError(err) {
    try {
      renderParallel?.cancel();
      await progressMsg.edit({
        embeds: [embeds.jobFailed({ url: modelId, userId, error: err.message })],
        components: [],
      });
      await thread.send(`<@${userId}> Quantization failed.`);
      await db.patchJob(jobId, {
        status: db.JOB_STATUS.failed,
        failedAt: Date.now(),
        error: sanitizeErrorText(err.message),
      });
    } catch (e) {
      log.error('Error callback error', { error: e.message });
    }
  }

  // RunPod EXL3: one pod per variant, in parallel. Each run_runpod_job grabs
  // its own pod (unique name tag -> safe orphan cleanup) and quantizes a single
  // bpw, so per-pod volume is small (easier to find stock) and the variants
  // finish concurrently instead of one after another. Progress from all pods is
  // aggregated into a single embed, one line per bpw.
  if (provider === 'runpod' && format === 'exl3') {
    await db.patchJob(jobId, { status: db.JOB_STATUS.running, startedAt: Date.now() });

    // Pod ids as the controllers announce them. /stop reads them back off the
    // job: the controller log is the other copy, and that one is only reachable
    // while the controller record still matches this job.
    const podIds = new Set(job.podIds ?? []);

    const pstate = {};
    variants.forEach((v) => {
      pstate[v] = { stage: 'Provisioning', overall: 0, message: 'waiting for a GPU', startedAt: Date.now() };
    });
    // The rows the embed draws. Normally the requested bitrates, but a job that
    // has to build its own donor first shows that too -- otherwise the embed
    // sits on "Queued" for the whole donor conversion, which is the frozen
    // embed this spent a long night getting rid of.
    let renderVariants = variants;
    renderParallel = throttle(async () => {
      try {
        await progressMsg.edit({
          embeds: [embeds.jobProgressParallel({
            url: modelId, userId, variants: renderVariants, state: pstate,
          })],
        });
      } catch (err) {
        log.debug(`parallel embed edit failed: ${err.message}`);
      }
    });
    renderParallel();
    // Heartbeat: re-render every 15s so elapsed time ticks even during silent
    // phases (the model download emits no events for ~10-15 min), so the embed
    // never looks frozen.
    const heartbeat = setInterval(() => renderParallel(), 15000);

    // Run one variant on its own pod, retrying the WHOLE thing on failure.
    // run_runpod_job can drop a variant to a transient post-create error
    // (SSH/API blip, rate limit) that spikes under concurrency; a fresh attempt
    // almost always succeeds, and it self-heals in the embed instead of leaving
    // a dead "Failed" line. Each attempt's pod self-terminates, so retries
    // never stack cost. The retry gate (err.retryable !== false) lives in
    // runVariantWithRetry: a signal-killed or pod-created failure is terminal,
    // so cancelling a broken model can't respawn a controller (the runaway).
    const MAX_ATTEMPTS = 3;

    // Progress handler shared by a fresh run and a resumed one, so a reattached
    // job draws the same embed the original was drawing.
    const onGroupProgress = (vs) => (d) => {
      if (d.podId && !podIds.has(d.podId)) {
        podIds.add(d.podId);
        db.patchJob(jobId, { podIds: [...podIds] }).catch((err) =>
          log.debug(`could not record pod ${d.podId}: ${err.message}`)
        );
      }
      // Provisioning, download and calibration are one pod's work for the whole
      // group, so every row in it shows them -- that is the truth, not a smear.
      // Once the converter starts the stream names a bitrate, and only that row
      // moves.
      const targets = d.currentBPW && vs.includes(d.currentBPW) ? [d.currentBPW] : vs;
      for (const v of targets) {
        pstate[v] = { ...pstate[v], stage: d.stage, overall: d.overall, message: d.message };
      }
      renderParallel();
    };

    // Each variant takes its own row's url/sample from the group's result.
    function markComplete(vs, res) {
      const byVariant = new Map((res || []).map((r) => [String(r.variant ?? r.bpw), r]));
      for (const x of vs) {
        const r = byVariant.get(String(x));
        pstate[x] = { ...pstate[x], stage: 'Complete', overall: 100, message: 'done',
                      url: r?.url ?? null, sample: r?.sample ?? null };
      }
      renderParallel();
    }

    async function runGroup(vs, over = {}) {
      const v = vs[0];
      // Resuming: the controller is already running and owns the pod. Attach to
      // its log instead of spawning a second one, and do not retry -- a retry
      // here would rent a pod alongside the one still working.
      const handle = resumeFrom?.get(v);
      if (handle) {
        try {
          const res = await attachToCli(handle, { variants: vs, onProgress: onGroupProgress(vs), sc });
          markComplete(vs, res);
          return res;
        } catch (err) {
          for (const x of vs) {
            pstate[x] = { ...pstate[x], stage: 'Failed', overall: 0, message: err.message };
          }
          renderParallel();
          return vs.map((x) => ({ bpw: x, variant: x, url: null, pushed: false, reused: false, duration: '', error: err.message }));
        }
      }
      // Every pod this group rented, retries included, so the completion can
      // say what the job really cost.
      const spent = [];
      const withSpend = (res) => {
        if (res?.length && spent.length) {
          res[0].podCost = { cost: spent.reduce((a, c) => a + c.cost, 0), pods: spent };
        }
        return res;
      };
      try {
        return withSpend(await runVariantWithRetry(v, {
          maxAttempts: MAX_ATTEMPTS,
          run: async () => {
            const res = await runViaCli({
              jobId,
              modelId,
              variants: vs,
              hfOrg: config.HF_ORG,
              testPrompt,
              codebook,
              visionBits,
              headBits,
              gpuCount,
              subfolder,
              sc,
              scLite,
              mtpBits,
              ngramBits,
              hq,
              donorRepo,
              maxRuntimeH: podRuntimeHours({
                sc: over.sc ?? sc, scLite: over.scLite ?? scLite, sizeGb, isMoe, nVariants: vs.length,
              }),
              ...over,
              onProgress: onGroupProgress(vs),
            }).catch((err) => {
              if (err.podCost) spent.push(err.podCost);
              throw err;
            });
            if (res[0]?.podCost) spent.push(res[0].podCost);
            markComplete(vs, res);
            return res;
          },
          onRetry: (next, max) => {
            for (const x of vs) {
              pstate[x] = { ...pstate[x], stage: 'Retrying', overall: 0, message: `attempt ${next}/${max}` };
            }
            renderParallel();
          },
        }));
      } catch (err) {
        for (const x of vs) {
          pstate[x] = { ...pstate[x], stage: 'Failed', overall: 0, message: err.message };
        }
        renderParallel();
        return withSpend(vs.map((x) => ({ bpw: x, variant: x, url: null, pushed: false, reused: false, duration: '', error: err.message })));
      }
    }

    // Self-calibration with no donor: build one first, on its own pod, and hand
    // it to the run that needs it. Two pods rather than one -- the saving from
    // merging them is about seven minutes on a job of several hours, and
    // splitting keeps each side clear of the 8h max_runtime, which is the
    // failure that actually costs something. The donor publishes either way, so
    // a later SC failure still leaves a usable quant behind.
    let effDonorRepo = donorRepo;
    let donorResult = null;
    if (sc && !effDonorRepo && donorBpw) {
      renderVariants = [donorBpw, ...variants];
      pstate[donorBpw] = { stage: 'Queued', overall: 0, message: 'donor for self-calibration' };
      renderParallel();
      // A plain build with exllamav3's defaults, whatever the job pinned: it is
      // published as a plain quant and found again by its plain name, and a
      // -HQ2 or -M3 donor would be neither.
      const res = await runGroup([donorBpw], {
        sc: false, scLite: false, donorRepo: null, mtpBits: null, ngramBits: null, hq: null,
      });
      effDonorRepo = res?.[0]?.url
        ? res[0].url.replace('https://huggingface.co/', '')
        : null;
      if (!effDonorRepo) {
        const why = res?.[0]?.error || 'no upload URL';
        await handleError(new Error(
          `Could not build the ${donorBpw}bpw donor self-calibration needs: ${why}`));
        return { thread };
      }
      // It is a real published quant, not scaffolding, so it belongs in the
      // completion embed alongside the variant that was asked for.
      donorResult = { ...res[0], donor: true };
      log.info(`built SC donor ${effDonorRepo} for ${modelId}`);
      await db.patchJob(jobId, { donorRepo: effDonorRepo })
        .catch((err) => log.debug(`could not record donor: ${err.message}`));
    }

    const groups = podGroups(variants, sc);
    let settled;
    try {
      settled = await Promise.all(
        // Stagger starts so 3 controllers don't hit the RunPod API in lockstep.
        groups.map((g, i) =>
          new Promise((r) => setTimeout(r, i * 4000))
            // The drafter rides on the first pod only; one copy is enough.
            .then(() => runGroup(g, { donorRepo: effDonorRepo, draftModel: i === 0 ? draftModel : null }))
        )
      );
    } finally {
      clearInterval(heartbeat);
    }
    // Killing the controllers fails every variant, so the normal completion
    // path would write "Complete" over the stop card and patch the status back
    // to completed. A stopped job is already final.
    const after = (await db.loadJobs().catch(() => ({})))[jobId];
    if (after?.status === db.JOB_STATUS.stopped) {
      renderParallel.cancel();
      await progressMsg
        .edit({ embeds: [stoppedEmbed(after)], components: [] })
        .catch((err) => log.debug(`stopped embed edit failed: ${err.message}`));
    } else {
      await handleComplete([...(donorResult ? [donorResult] : []), ...settled.flat()]);
    }
    return { thread };
  }

  const useApi = await isApiAvailable();

  if (useApi) {
    try {
      const apiJob = await submitJob({
        model_id: modelId,
        format,
        variants,
        provider,
        hf_org: config.HF_ORG,
        codebook,
      });

      await db.patchJob(jobId, {
        status: db.JOB_STATUS.running,
        startedAt: Date.now(),
        apiJobId: apiJob.job_id,
      });
      updateEmbed({ stage: 'Queued', progress: 0, overall: 0, message: `API job ${apiJob.job_id} queued` });

      const totalBPWs = bpws.length;
      const apiResult = await pollJob(apiJob.job_id, (status, result, error, progress) => {
        if (status === 'complete') {
          updateEmbed({ stage: 'Complete', progress: 100, overall: 100, message: 'Done' });
        } else if (status === 'failed') {
          updateEmbed({ stage: 'Failed', progress: 0, overall: 0, message: error || 'Failed' });
        } else {
          const stage = progress?.stage || status;
          const msg = progress?.message || status;
          const pct = progress?.percent ?? (status === 'running' ? 50 : 0);
          updateEmbed({ stage, progress: pct, overall: pct, message: msg, currentBPW: bpws[0], bpwIndex: 0, totalBPWs });
        }
      });

      const outputs = apiResult?.outputs || [];
      const results = outputs.map((o) => ({
        bpw: o.variant,
        variant: o.variant,
        duration: `${(apiResult?.total_wall_time ?? 0).toFixed(1)}s`,
        url: o.hf_url,
        treeUrl: o.hf_url,
        pushed: !!o.hf_url,
        reused: false,
        error: null,
      }));

      await handleComplete(results);
    } catch (err) {
      log.error('API job failed', { error: err.message });
      await handleError(err);
    }
  } else {
    queue.enqueue({
      jobId,
      url: urlInput,
      bpws,
      categories: [category],
      quantOptions: { headBits },
      precheckedRepos,
      userId,
      onProgress: (data) => updateEmbed(data),
      onComplete: handleComplete,
      onError: handleError,
    });
  }

  return { thread };
}
