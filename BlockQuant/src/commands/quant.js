import { getLogger } from '../logger.js';
import { MessageFlags } from 'discord.js';
import { randomUUID } from 'node:crypto';
import config from '../config.js';
import * as hf from '../services/huggingface.js';
import * as queue from '../services/queue.js';
import * as quantizer from '../services/quantizer.js';
import * as db from '../services/db.js';
import * as embeds from '../utils/embeds.js';
import { sanitizeErrorText, toUserMessage } from '../errors/taxonomy.js';
import { exl3RepoName } from '../utils/hfExl3.js';

const log = getLogger('cmd:quant');

/** Throttle progress embed edits to max once per 4 seconds. */
function throttle(fn, ms = 4000) {
  let last = 0;
  let pending = null;
  return (...args) => {
    const now = Date.now();
    if (now - last >= ms) {
      last = now;
      return fn(...args);
    }
    // Buffer the latest call
    clearTimeout(pending);
    pending = setTimeout(
      () => {
        last = Date.now();
        fn(...args);
      },
      ms - (now - last)
    );
  };
}

export async function handleQuant(interaction) {
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });

  const urlInput = interaction.options.getString('url', true);
  const bpwInput = interaction.options.getString('bpw', true);
  const profile = interaction.options.getString('profile') ?? 'auto';
  const headBitsOverride = interaction.options.getInteger('head_bits');
  const category = interaction.options.getString('category') ?? 'General';
  const userId = interaction.user.id;
  const isAdmin = config.ADMIN_IDS.includes(userId);

  if (!config.QUANT_PROFILES[profile]) {
    return interaction.editReply({
      embeds: [embeds.error('Invalid profile', `Unknown profile: \`${profile}\``)],
    });
  }

  if (headBitsOverride != null && !isAdmin) {
    return interaction.editReply({
      embeds: [embeds.error('Forbidden', 'Only admins can override `head_bits`.')],
    });
  }
  
  // Store profile and override for later calculation
  // headBits will be calculated per BPW if profile is 'auto'
  const quantOptions = {
    headBits: headBitsOverride ?? config.QUANT_PROFILES[profile].headBits,
    profile,
    isAuto: profile === 'auto' && headBitsOverride == null,
  };

  // ── Parse & validate BPW list ─────────────────────────────────────────────
  const bpws = [...new Set(
    bpwInput
      .split(',')
      .map((s) => s.trim())
      .map((s) => parseFloat(s))
      .filter((n) => !isNaN(n))
  )].sort((a, b) => a - b);
  if (bpws.length === 0) {
    return interaction.editReply({
      embeds: [embeds.error('Invalid BPW', 'Provide comma-separated numbers, e.g. `3.0,4.0,5.0`')],
    });
  }
  const invalid = bpws.filter((b) => b < 1.5 || b > 8.5);
  if (invalid.length) {
    return interaction.editReply({
      embeds: [
        embeds.error(
          'BPW Out of Range',
          `Values must be between 1.5–8.5. Got: ${invalid.join(', ')}`
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
    embeds: [
      embeds.info(
        '🔍 Running Pre-flight Checks…',
        'Verifying HuggingFace credentials and model access…'
      ),
    ],
  });

  let flight;
  try {
    flight = await hf.preflight(urlInput);
  } catch (err) {
    log.error('Preflight failed', { error: err.message, userId });
    return interaction.editReply({
      embeds: [embeds.error('Pre-flight Failed', toUserMessage(err))],
    });
  }

  if (!flight.modelExists) {
    return interaction.editReply({
      embeds: [
        embeds.error('Model Not Found', `\`${modelId}\` does not exist or is not accessible.`),
      ],
    });
  }

  if (!flight.canWrite) {
    return interaction.editReply({
      embeds: [
        embeds.error(
          'Token Error',
          'The HF token does not have write permissions. Update `HF_TOKEN` in .env.'
        ),
      ],
    });
  }

  if (flight.isGguf) {
    return interaction.editReply({
      embeds: [
        embeds.error(
          'Unsupported Model Format',
          'GGUF source models are disabled in this HF-only bot build. Use a standard Hugging Face model repo with `config.json` and safetensors/bin weights.'
        ),
      ],
    });
  }

  // ── Runtime setup checks ───────────────────────────────────────────────────
  try {
    await hf.validateSetup();
    await quantizer.validateSetup();
  } catch (err) {
    log.error('Runtime setup check failed', { error: err.message, userId });
    return interaction.editReply({
      embeds: [embeds.error('Quantizer Setup Error', toUserMessage(err) || err.message)],
    });
  }

  // ── Pre-check upload targets for idempotency ──────────────────────────────
  const modelName = modelId.split('/').pop();
  const precheckedRepos = {};
  const alreadyUploaded = [];
  try {
    for (const bpw of bpws) {
      const repoName = exl3RepoName(modelName, bpw);
      const state = await hf.inspectUploadRepo(repoName, {
        sourceModel: modelId,
        profile,
        bpw,
        quantOptions,
      });
      precheckedRepos[String(bpw)] = state;
      if (
        state.exists &&
        state.settingsMatch === false &&
        state.reason !== 'manifest_missing'
      ) {
        return interaction.editReply({
          embeds: [
            embeds.error(
              'Existing Repo Conflict',
              `\`${state.repoId}\` already has a manifest with different quant settings (${state.reason ?? 'manifest mismatch'}).`
            ),
          ],
        });
      }
      if (state.exists && state.settingsMatch) {
        alreadyUploaded.push(`${bpw} bpw`);
      }
    }
  } catch (err) {
    return interaction.editReply({
      embeds: [embeds.error('Repo Pre-check Failed', toUserMessage(err))],
    });
  }

  if (alreadyUploaded.length === bpws.length) {
    return interaction.editReply({
      embeds: [
        embeds.success(
          'Already Quantized',
          `Matching uploads already exist for all requested BPWs: ${alreadyUploaded.join(', ')}`
        ),
      ],
    });
  }

  // ── EXP check ─────────────────────────────────────────────────────────────
  const users = await db.loadUsers();
  const user = users[userId] ?? { exp: 0, lastQuant: 0 };
  const missingCount = bpws.length - alreadyUploaded.length;
  const cost = missingCount * 10; // charge only for bpw variants that still need work
  if (user.exp < cost) {
    return interaction.editReply({
      embeds: [
        embeds.warning(
          'Insufficient EXP',
          `You need **${cost}** EXP but have **${user.exp}**. Keep chatting to earn more!`
        ),
      ],
    });
  }

  const jobId = randomUUID();
  const createdAt = Date.now();

  // ── Create thread for live progress (fallback to channel if unavailable) ─
  const channel =
    interaction.channel ?? (await interaction.client.channels.fetch(interaction.channelId).catch(() => null));

  if (!channel) {
    return interaction.editReply({
      embeds: [embeds.error('Channel Error', 'Unable to resolve this channel for progress updates.')],
    });
  }

  let thread = null;
  if (channel.threads && typeof channel.threads.create === 'function') {
    thread = await channel.threads.create({
      name: `⚡ ${modelId.split('/').pop()} [${interaction.user.username}]`,
      autoArchiveDuration: 1440,
    });
  }

  const progressChannel = thread ?? channel;
  if (typeof progressChannel.send !== 'function') {
    return interaction.editReply({
      embeds: [
        embeds.error(
          'Channel Error',
          'This channel cannot receive progress messages. Use `/quant` in a server text channel.'
        ),
      ],
    });
  }

  const progressMsg = await progressChannel.send({
    embeds: [embeds.jobQueued({ url: modelId, bpws, categories: [category], userId })],
  });

  await db.upsertJob({
    id: jobId,
    status: db.JOB_STATUS.queued,
    createdAt,
    userId,
    modelId,
    url: urlInput,
    bpws,
    categories: [category],
    profile,
    quantOptions,
    cost,
    channelId: interaction.channelId,
    threadId: thread?.id ?? null,
    progressMessageId: progressMsg.id,
    chargedAt: null,
    refundedAt: null,
    partialResults: [],
  });

  const chargeResult = await db.chargeForJob({ jobId, userId, cost });
  if (!chargeResult.charged) {
    await db.patchJob(jobId, {
      status: db.JOB_STATUS.failed,
      error: 'Insufficient EXP at charge time',
      failedAt: Date.now(),
    });
    return interaction.editReply({
      embeds: [
        embeds.warning(
          'Insufficient EXP',
          `You need **${cost}** EXP but have **${chargeResult.balance ?? user.exp}**.`
        ),
      ],
    });
  }

  // Build auto configuration summary
  let autoSummary = '';
  if (quantOptions.isAuto && bpws.length > 0) {
    const sampleBpw = bpws[0];
    const autoParams = config.calculateAutoParams(sampleBpw);
    autoSummary = `\n🤖 **Auto mode** selected: head_bits=${autoParams.headBits}, cal=${autoParams.calRows}x${autoParams.calCols}`;
    if (bpws.length > 1) {
      autoSummary += ` (per BPW)`;
    }
  } else if (profile !== 'auto') {
    autoSummary = `\n⚙️ Profile: **${profile}** (head_bits=${quantOptions.headBits})`;
  }

  await interaction.editReply({
    embeds: [
      embeds.success(
        '✅ Job Submitted',
        `Follow progress in ${thread ?? progressChannel}.\nCost: **${cost}** EXP (balance: **${chargeResult.balance}**)${
          alreadyUploaded.length
            ? `\nReused existing uploads: **${alreadyUploaded.join(', ')}**`
            : ''
        }${autoSummary}`
      ),
    ],
  });

  // ── Throttled progress updater ────────────────────────────────────────────
  const updateEmbed = throttle(async (data) => {
    try {
      await progressMsg.edit({
        embeds: [embeds.jobProgress({ url: modelId, userId, ...data })],
      });
    } catch (err) {
      log.debug(`Failed to update embed: ${err.message}`);
    }
  });

  // ── Enqueue job ───────────────────────────────────────────────────────────
  queue.enqueue({
    jobId,
    url: urlInput,
    bpws,
    categories: [category],
    profile,
    quantOptions,
    precheckedRepos,
    userId,
    cost,

    onProgress: (data) => updateEmbed(data),

    onComplete: async (results) => {
      try {
        await progressMsg.edit({ embeds: [embeds.jobComplete({ url: modelId, userId, results })] });
        const pushedCount = results.filter((r) => r.pushed).length;
        let completionNote;
        if (pushedCount === results.length && pushedCount > 0) {
          completionNote = `<@${userId}> Your quantization is done! 🎉`;
        } else if (pushedCount > 0) {
          completionNote = `<@${userId}> Quantization finished with partial upload success (${pushedCount}/${results.length}).`;
        } else {
          completionNote =
            `<@${userId}> Quantization finished locally, but upload failed for all outputs. ` +
            'Please retry upload or contact an admin.';
        }
        if (typeof progressChannel.send === 'function') {
          await progressChannel.send(completionNote);
        }

        // Save model record
        const models = await db.loadModels();
        models[modelId] = {
          url: modelId,
          categories: [category],
          authors: [userId],
          quantizedBPWs: results.filter((r) => r.pushed).map((r) => r.bpw),
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
    },

    onError: async (err) => {
      try {
        const refund = await db.refundForJob({ jobId, userId, cost });
        await progressMsg.edit({
          embeds: [embeds.jobFailed({ url: modelId, userId, error: err.message })],
        });
        const refundMessage = refund.refunded
          ? `Your **${cost}** EXP has been refunded.`
          : 'No EXP refund was required for this job.';
        if (typeof progressChannel.send === 'function') {
          await progressChannel.send(`<@${userId}> Quantization failed. ${refundMessage}`);
        }
        await db.patchJob(jobId, {
          status: db.JOB_STATUS.failed,
          failedAt: Date.now(),
          error: sanitizeErrorText(err.message),
        });
      } catch (e) {
        log.error('Error callback error', { error: e.message });
      }
    },
  });
}
