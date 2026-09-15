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
import { costPreflightLine, getBalance, estimateCost } from '../services/runpod.js';
import { runViaCli, attachToCli, runVariantWithRetry, finalizeCollection } from '../services/runpodCli.js';
import { answerOf, extractSvg, renderSvgToPng } from '../utils/svg.js';
import { claimRunSlot, releaseRunSlot } from '../services/access.js';
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

export async function handleQuant(interaction) {
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });

  const urlInput = interaction.options.getString('url', true);
  const bpwInput = interaction.options.getString('bpw', true);
  // Optional smoke-test prompt: run on each finished quant so the requester
  // sees a real reply. Capped so it stays a quick check, not a chat session.
  const testPrompt = (interaction.options.getString('prompt') || '').trim().slice(0, 1000) || null;
  // Trellis codebook. Recorded in the quant itself, so it decides which
  // ExLlamaV3 builds can read what we publish; config.CODEBOOK is the default.
  const codebook = interaction.options.getString('codebook') || config.CODEBOOK;
  const sc = interaction.options.getBoolean('sc') ?? false;
  // Omitted means exllamav3 decides, and its default is NOT a flat 6: it is the
  // tower's own declared default_vision_bits, 6 on the arches with a validated
  // tower and 16 -- copy it whole -- everywhere else. 1.4.9 is where that split
  // appeared; older builds copied every tower, so the same request gives a
  // different artifact now, and 16 is the way back. preflight resolves which
  // off the generated table, so the name can say -V6 without asking.
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
  let variants = bpwInput.split(',').map((s) => s.trim()).filter(Boolean);
  if (variants.length === 0) {
    return interaction.editReply({
      embeds: [embeds.error('Invalid Variants', 'Provide comma-separated values, e.g. `3.0,4.0,5.0`')],
    });
  }

  const bpws = variants.map((s) => parseFloat(s)).filter((n) => !isNaN(n));
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
          headBits: sc ? effHeadBits : null,
          visionBits: sc ? effVisionBits : towerBits,
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
  if (sc) {
    const wanted = Math.max(...bpws);
    try {
      donorRepo = await hf.findDonorQuant(modelName, wanted);
    } catch (err) {
      return interaction.editReply({
        embeds: [embeds.error('Could not look for a donor', toUserMessage(err))],
      });
    }
    if (!donorRepo) {
      return interaction.editReply({
        embeds: [
          embeds.error(
            'No donor quant to calibrate from',
            `Self-calibration samples the model through one of its own quants, at least ` +
              `as many bits as the target. Nothing at \`${wanted}\`bpw or above exists for ` +
              `\`${modelName}\` yet — run a plain \`/quant\` first, then ask for SC.`
          ),
        ],
      });
    }
    log.info(`SC donor for ${modelId}: ${donorRepo}`);
  }

  // ── Cost gate: refuse if RunPod can't even afford the optimistic estimate ─
  // The work to actually do is the variants minus the ones already uploaded.
  const billable = Math.max(1, variants.length - alreadyUploaded.length);
  if (provider === 'runpod') {
    const bal = await getBalance();
    const est = estimateCost(billable);
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
      donorRepo,
      visionBits: towerBits,
      headBits: effHeadBits,
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

  const costLine = provider === 'runpod' ? await costPreflightLine(variants.length) : '';

  const requestEmbed = embeds.info(
    slot.quanter ? 'Quantization request · starting' : 'Quantization request · awaiting approval',
    [
      `**Model:** [\`${modelId}\`](https://huggingface.co/${modelId})`,
      `**Variants:** ${variants.join(', ')}  ·  **Format:** ${format.toUpperCase()}`,
      subfolder ? `**Subfolder:** \`${subfolder}\`` : '',
      `**Head bits:** ${effHeadBits ?? defaultHeadBits()}  ·  **Codebook:** \`${codebook}\`  ·  **Vision:** ${visionSummary(flight, effVisionBits)}`,
      `**Provider:** ${provider}`,
      costLine,
      `**Requested by:** <@${userId}>`,
      alreadyUploaded.length ? `**Reuses existing:** ${alreadyUploaded.join(', ')}` : '',
      `**Job:** \`${jobId.slice(0, 8)}\``,
      '',
      slot.quanter
        ? 'Starting now — the requester holds the quanter role. Stop it with the button or `/stop`.'
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
    subfolder = null,
    sc = false,
    donorRepo = null,
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
      const pushedCount = results.filter((r) => r.pushed).length;
      let completionNote;
      if (pushedCount === results.length && pushedCount > 0) {
        completionNote = `<@${userId}> Your quantization is done! 🎉`;
      } else if (pushedCount > 0) {
        completionNote = `<@${userId}> Quantization finished with partial upload success (${pushedCount}/${results.length}).`;
      } else {
        completionNote =
          `<@${userId}> Quantization finished, but upload failed for all outputs. ` +
          'Please retry upload or contact an admin.';
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
        quantizedBPWs: results.filter((r) => r.pushed).map((r) => r.bpw ?? r.variant),
        results,
        completedAt: Date.now(),
      };
      await db.saveModels(models);

      await db.patchJob(jobId, {
        status: db.JOB_STATUS.completed,
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
    renderParallel = throttle(async () => {
      try {
        await progressMsg.edit({
          embeds: [embeds.jobProgressParallel({ url: modelId, userId, variants, state: pstate })],
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
    const onVariantProgress = (v) => (d) => {
      if (d.podId && !podIds.has(d.podId)) {
        podIds.add(d.podId);
        db.patchJob(jobId, { podIds: [...podIds] }).catch((err) =>
          log.debug(`could not record pod ${d.podId}: ${err.message}`)
        );
      }
      pstate[v] = { ...pstate[v], stage: d.stage, overall: d.overall, message: d.message };
      renderParallel();
    };

    async function runVariant(v) {
      // Resuming: the controller is already running and owns the pod. Attach to
      // its log instead of spawning a second one, and do not retry -- a retry
      // here would rent a pod alongside the one still working.
      const handle = resumeFrom?.get(v);
      if (handle) {
        try {
          const res = await attachToCli(handle, { variants: [v], onProgress: onVariantProgress(v) });
          const url = res?.[0]?.url ?? null;
          pstate[v] = { ...pstate[v], stage: 'Complete', overall: 100, message: 'done', url,
                        sample: res?.[0]?.sample ?? null };
          renderParallel();
          return res;
        } catch (err) {
          pstate[v] = { ...pstate[v], stage: 'Failed', overall: 0, message: err.message };
          renderParallel();
          return [{ bpw: v, variant: v, url: null, pushed: false, reused: false, duration: '', error: err.message }];
        }
      }
      try {
        return await runVariantWithRetry(v, {
          maxAttempts: MAX_ATTEMPTS,
          run: async () => {
            const res = await runViaCli({
              jobId,
              modelId,
              variants: [v],
              hfOrg: config.HF_ORG,
              testPrompt,
              codebook,
              visionBits,
              headBits,
              subfolder,
              sc,
              donorRepo,
              onProgress: onVariantProgress(v),
            });
            const url = res && res[0] ? res[0].url : null;
            const sample = res && res[0] ? res[0].sample : null;
            pstate[v] = { ...pstate[v], stage: 'Complete', overall: 100, message: 'done', url, sample };
            renderParallel();
            return res;
          },
          onRetry: (next, max) => {
            pstate[v] = { ...pstate[v], stage: 'Retrying', overall: 0, message: `attempt ${next}/${max}` };
            renderParallel();
          },
        });
      } catch (err) {
        pstate[v] = { ...pstate[v], stage: 'Failed', overall: 0, message: err.message };
        renderParallel();
        return [{ bpw: v, variant: v, url: null, pushed: false, reused: false, duration: '', error: err.message }];
      }
    }

    let settled;
    try {
      settled = await Promise.all(
        // Stagger starts so 3 controllers don't hit the RunPod API in lockstep.
        variants.map((v, i) =>
          new Promise((r) => setTimeout(r, i * 4000)).then(() => runVariant(v))
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
      await handleComplete(settled.flat());
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
