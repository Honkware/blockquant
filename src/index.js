import { ActivityType, Client, GatewayIntentBits, REST, Routes } from 'discord.js';
import config from './config.js';
import { getLogger } from './logger.js';
import * as db from './services/db.js';
import * as jobQueue from './services/queue.js';
import { commands } from './commands/definitions.js';
import { routeCommand } from './commands/router.js';
import { getJobStatus, pollJob } from './services/api-client.js';
import * as embeds from './utils/embeds.js';
import { listPods, terminatePod } from './services/runpod.js';
import { reapOrphans as reap } from './services/reaper.js';
import { sweep as sweepControllers } from './services/detached.js';
import { registerChat } from './chat/index.js';

const log = getLogger('bot');

// Safety net: every controller names its pod `bq-<pid>-<ts%100000>-<ts>`. If
// that controller is gone, nobody will terminate the pod. Lives in
// services/reaper.js so it is testable; see backend/scripts/pod_watchdog.py for
// the same checks applied to pods this one does not match.
async function reapOrphans() {
  return reap({ listPods, terminatePod });
}

function trimPresence(text, max = 128) {
  return text.length <= max ? text : `${text.slice(0, max - 3)}...`;
}

function applyPresence(client, payload = {}) {
  if (!client.user) return;

  const waiting = payload.waiting ?? 0;
  const state = payload.state ?? 'idle';
  const stage = payload.stage ?? 'Working';
  const bpw = payload.currentBPW;

  let name = 'for /quant jobs';

  if (state === 'paused') {
    name = waiting > 0 ? `queue paused (${waiting} waiting)` : 'queue paused';
  } else if (state === 'active' && payload.modelId) {
    const stageText = bpw ? `${stage} ${bpw}bpw` : stage;
    name = `${payload.modelId} (${stageText})`;
  } else if (waiting > 0) {
    name = `${waiting} job${waiting === 1 ? '' : 's'} waiting`;
  }

  client.user.setPresence({
    activities: [{ type: ActivityType.Watching, name: trimPresence(name) }],
    status: 'online',
  });
}

// ── Discord Client ──────────────────────────────────────────────────────────

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

// ── Command Handling ────────────────────────────────────────────────────────

client.on('interactionCreate', routeCommand);

// Optional Kimi chat module — no-op unless CHAT_ENABLED + KIMI_API_KEY are set.
// Delete this line + src/chat/ to remove the feature entirely.
registerChat(client);

// ── Ready ───────────────────────────────────────────────────────────────────

client.once('clientReady', () => {
  applyPresence(client, { state: 'idle' });
  log.info(`Online as ${client.user.tag} - serving ${client.guilds.cache.size} guild(s)`);
});

// ── API Job Recovery ────────────────────────────────────────────────────────

async function recoverApiJobs() {
  const recoverable = await db.listRecoverableJobs();
  const apiJobs = recoverable.filter((j) => j.apiJobId && (j.status === db.JOB_STATUS.running || j.status === db.JOB_STATUS.queued));
  if (!apiJobs.length) return 0;

  for (const job of apiJobs) {
    try {
      // Check if API job is still alive
      const status = await getJobStatus(job.apiJobId);
      if (!status || status.status === 'complete' || status.status === 'failed') {
        // Already finished — patch local record and skip
        await db.patchJob(job.id, {
          status: status?.status === 'complete' ? db.JOB_STATUS.completed : db.JOB_STATUS.failed,
          completedAt: Date.now(),
          failedAt: status?.status === 'failed' ? Date.now() : undefined,
        });
        continue;
      }

      // Fetch Discord thread and progress message
      const channel = await client.channels.fetch(job.channelId).catch(() => null);
      const thread = channel?.threads?.cache?.get(job.threadId) || await channel?.threads?.fetch(job.threadId).catch(() => null);
      const progressMsg = thread
        ? await thread.messages.fetch(job.progressMessageId).catch(() => null)
        : null;

      if (!thread || !progressMsg) {
        log.warn(`Cannot recover API job ${job.id}: thread or message missing`);
        continue;
      }

      log.info(`Resuming polling for API job ${job.apiJobId} (local ${job.id})`);

      // Resume polling
      pollJob(job.apiJobId, (status, result, error, progress) => {
        const updateEmbed = async (data) => {
          try {
            await progressMsg.edit({
              embeds: [embeds.jobProgress({ url: job.modelId, userId: job.userId, ...data })],
            });
          } catch (err) {
            log.debug(`Failed to update recovered embed: ${err.message}`);
          }
        };

        if (status === 'complete') {
          updateEmbed({ stage: 'Complete', progress: 100, overall: 100, message: 'Done' });
          thread.send(`<@${job.userId}> Your quantization is done! 🎉`);
          db.patchJob(job.id, { status: db.JOB_STATUS.completed, completedAt: Date.now() });
        } else if (status === 'failed') {
          updateEmbed({ stage: 'Failed', progress: 0, overall: 0, message: error || 'Failed' });
          thread.send(`<@${job.userId}> Quantization failed.`);
          db.patchJob(job.id, { status: db.JOB_STATUS.failed, failedAt: Date.now() });
        } else {
          const stage = progress?.stage || status;
          const msg = progress?.message || status;
          const pct = progress?.percent ?? (status === 'running' ? 50 : 0);
          updateEmbed({ stage, progress: pct, overall: pct, message: msg });
        }
      }).catch((err) => {
        log.error(`Recovered API job ${job.id} failed`, { error: err.message });
        db.patchJob(job.id, { status: db.JOB_STATUS.failed, failedAt: Date.now() });
      });
    } catch (err) {
      log.error(`Failed to recover API job ${job.id}`, { error: err.message });
    }
  }

  return apiJobs.length;
}

// ── Bootstrap ───────────────────────────────────────────────────────────────

async function start() {
  log.info('Starting BlockQuant v2.0 (ExLlamaV3 / EXL3)');

  // 1. Register slash commands
  const rest = new REST({ version: '10' }).setToken(config.BOT_TOKEN);
  try {
    log.info('Registering slash commands...');
    await rest.put(Routes.applicationGuildCommands(config.CLIENT_ID, config.GUILD_ID), {
      body: commands.map((c) => c.toJSON()),
    });
    log.info(`Registered ${commands.length} commands`);
  } catch (err) {
    log.error('Failed to register commands', { error: err.message });
    process.exit(1);
  }

  // 2. Init job queue
  jobQueue.setPresenceUpdater((payload) => applyPresence(client, payload));
  jobQueue.init();
  // Controllers outlive the bot now (services/detached.js), so a restart no
  // longer kills the work. What it does lose is the Discord half: the progress
  // embed stops moving and nothing bot-side runs at the end. Say so, with the
  // log to watch, because otherwise nobody finds out.
  const { live, finished } = sweepControllers();
  for (const c of live) {
    const what = `${c.kind} ${c.meta?.modelId || c.id}`;
    log.warn(`${what} controller survived the restart (pid ${c.pid}); its embed is dead. Watch ${c.logPath}`);
  }
  if (live.some((c) => c.kind === 'quant')) {
    log.warn('Cards and the collection are written bot-side: run backend/scripts/publish_quant.py --base <model> once those land');
  }
  for (const c of finished) {
    const out = c.meta?.resultPath ? `, result ${c.meta.resultPath}` : '';
    log.warn(`${c.kind} ${c.meta?.modelId || c.id} finished while the bot was down: ${c.logPath}${out}`);
  }
  const liveModels = new Set(live.map((c) => c.meta?.modelId).filter(Boolean));

  // Do NOT auto-re-run persisted jobs on restart: the old recover path
  // re-enqueued them to the LOCAL quantizer (a 70GB download on the host). Mark
  // any leftover non-terminal job interrupted; the operator re-fires what they
  // want, or lets a surviving controller finish it.
  const stale = await db.listRecoverableJobs();
  for (const job of stale) {
    await db.patchJob(job.id, {
      status: db.JOB_STATUS.failed,
      error: liveModels.has(job.modelId)
        ? 'bot restarted; controller still running, quant will upload without the bot'
        : 'interrupted by bot restart',
      failedAt: Date.now(),
    });
  }
  if (stale.length) {
    log.info(`Marked ${stale.length} interrupted job(s) on startup (not re-run)`);
  }

  // 3. Recover API jobs (so restart doesn't orphan running Celery tasks)
  const apiRecovered = await recoverApiJobs();
  if (apiRecovered > 0) {
    log.info(`Recovered ${apiRecovered} API job(s) after startup`);
  }

  // 4. Login
  await client.login(config.BOT_TOKEN);

  // 5. Orphan reaper: sweep on startup (catches pods leaked by a crashed
  // previous session) and every 2 min thereafter.
  reapOrphans().catch(() => {});
  setInterval(() => reapOrphans().catch(() => {}), 120_000);
}

// ── Graceful Shutdown ───────────────────────────────────────────────────────

async function shutdown(signal) {
  log.info(`Received ${signal}, shutting down...`);
  await jobQueue.shutdown();
  client.destroy();
  process.exit(0);
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('unhandledRejection', (err) => {
  log.error('Unhandled rejection', { error: err?.message, stack: err?.stack });
});

// ── Go ──────────────────────────────────────────────────────────────────────

start().catch((err) => {
  log.error('Fatal startup error', { error: err.message, stack: err.stack });
  process.exit(1);
});
