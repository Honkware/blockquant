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
import { exl3RepoName, formatExl3Revision } from '../utils/hfExl3.js';
import { isApiAvailable, submitJob, pollJob } from '../services/api-client.js';

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
  const profile = interaction.options.getString('profile') ?? 'balanced';
  const headBitsOverride = interaction.options.getInteger('head_bits');
  const category = interaction.options.getString('category') ?? 'General';
  const format = interaction.options.getString('format') ?? 'exl3';
  const provider = interaction.options.getString('provider') ?? 'local';
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
  const quantOptions = {
    headBits: headBitsOverride ?? config.QUANT_PROFILES[profile].headBits,
    profile,
  };

  // ── Parse & validate variant list ─────────────────────────────────────────
  let variants = bpwInput.split(',').map((s) => s.trim()).filter(Boolean);
  if (variants.length === 0) {
    return interaction.editReply({
      embeds: [embeds.error('Invalid Variants', 'Provide comma-separated values, e.g. `3.0,4.0,5.0` or `q4_k_m,q5_k_m`')],
    });
  }

  let bpws;
  if (format === 'gguf') {
    const VALID_GGUF_VARIANTS = new Set([
      'q4_0', 'q4_1', 'q4_k_s', 'q4_k_m',
      'q5_0', 'q5_1', 'q5_k_s', 'q5_k_m',
      'q6_k', 'q8_0', 'f16', 'bf16',
    ]);
    const invalid = variants.filter((v) => !VALID_GGUF_VARIANTS.has(v.toLowerCase()));
    if (invalid.length) {
      return interaction.editReply({
        embeds: [
          embeds.error(
            'Invalid GGUF Variant',
            `Unknown variant(s): ${invalid.join(', ')}. Valid: ${Array.from(VALID_GGUF_VARIANTS).join(', ')}`
          ),
        ],
      });
    }
    variants = variants.map((v) => v.toLowerCase());
    bpws = []; // BPW concept doesn't apply to GGUF in the same way
  } else {
    // EXL3: parse as floats
    bpws = variants.map((s) => parseFloat(s)).filter((n) => !isNaN(n));
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

  // ── Runtime setup checks (local EXL3 only) ────────────────────────────────
  if (format === 'exl3') {
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
  const repoName = exl3RepoName(modelName);
  let precheckedRepos = {};
  let alreadyUploaded = [];
  if (format === 'exl3') {
    try {
      for (const bpw of bpws) {
        const revision = formatExl3Revision(bpw);
        const state = await hf.inspectUploadRepo(repoName, {
          sourceModel: modelId,
          profile,
          bpw,
          quantOptions,
          revision,
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
                `\`${state.repoId}\` (branch \`${revision}\`) already has a manifest with different quant settings (${state.reason ?? 'manifest mismatch'}).`
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
  }

  // ── EXP check ─────────────────────────────────────────────────────────────
  const users = await db.loadUsers();
  const user = users[userId] ?? { exp: 0, lastQuant: 0 };
  const missingCount = variants.length - alreadyUploaded.length;
  const cost = missingCount * 10; // charge only for variants that still need work
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

  // ── Create thread for live progress ───────────────────────────────────────
  const thread = await interaction.channel.threads.create({
    name: `⚡ ${modelId.split('/').pop()} [${interaction.user.username}]`,
    autoArchiveDuration: 1440,
  });

  const progressMsg = await thread.send({
    embeds: [embeds.jobQueued({ url: modelId, bpws: variants, categories: [category], userId })],
  });

  await db.upsertJob({
    id: jobId,
    status: db.JOB_STATUS.queued,
    createdAt,
    userId,
    modelId,
    url: urlInput,
    format,
    variants,
    bpws,
    categories: [category],
    profile,
    quantOptions,
    cost,
    channelId: interaction.channelId,
    threadId: thread.id,
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

  await interaction.editReply({
    embeds: [
      embeds.success(
        '✅ Job Submitted',
        `Follow progress in ${thread}.\nCost: **${cost}** EXP (balance: **${chargeResult.balance}**)${
          alreadyUploaded.length
            ? `\nReused existing uploads: **${alreadyUploaded.join(', ')}**`
            : ''
        }`
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

  // Shared completion handler
  async function handleComplete(results) {
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
          `<@${userId}> Quantization finished, but upload failed for all outputs. ` +
          'Please retry upload or contact an admin.';
      }
      await thread.send(completionNote);

      // Save model record
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
  }

  // Shared error handler
  async function handleError(err) {
    try {
      const refund = await db.refundForJob({ jobId, userId, cost });
      await progressMsg.edit({
        embeds: [embeds.jobFailed({ url: modelId, userId, error: err.message })],
      });
      const refundMessage = refund.refunded
        ? `Your **${cost}** EXP has been refunded.`
        : 'No EXP refund was required for this job.';
      await thread.send(`<@${userId}> Quantization failed. ${refundMessage}`);
      await db.patchJob(jobId, {
        status: db.JOB_STATUS.failed,
        failedAt: Date.now(),
        error: sanitizeErrorText(err.message),
      });
    } catch (e) {
      log.error('Error callback error', { error: e.message });
    }
  }

  // ── API vs Local branch ───────────────────────────────────────────────────
  const useApi = format === 'gguf' || await isApiAvailable();

  if (useApi) {
    // ── API mode ─────────────────────────────────────────────────────────────
    try {
      const apiJob = await submitJob({
        model_id: modelId,
        format,
        variants,
        provider,
        hf_org: config.HF_ORG,
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
          updateEmbed({
            stage,
            progress: pct,
            overall: pct,
            message: msg,
            currentBPW: bpws[0],
            bpwIndex: 0,
            totalBPWs,
          });
        }
      });

      // Map API result to existing result shape
      const outputs = apiResult?.outputs || [];
      const results = outputs.map((o) => ({
        bpw: o.variant,
        variant: o.variant,
        duration: `${(apiResult?.total_wall_time ?? 0).toFixed(1)}s`,
        url: o.hf_url,
        treeUrl: o.hf_url,
        revision: o.hf_revision || (format === 'exl3' ? formatExl3Revision(parseFloat(o.variant)) : o.variant),
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
    // ── Local mode (fallback) ────────────────────────────────────────────────
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
      onComplete: handleComplete,
      onError: handleError,
    });
  }
}
