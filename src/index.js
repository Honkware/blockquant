import { ActivityType, Client, GatewayIntentBits, REST, Routes } from 'discord.js';
import config from './config.js';
import { getLogger } from './logger.js';
import * as db from './services/db.js';
import * as jobQueue from './services/queue.js';
import { commands } from './commands/definitions.js';
import { routeCommand } from './commands/router.js';
import { getJobStatus, pollJob } from './services/api-client.js';
import * as embeds from './utils/embeds.js';

const log = getLogger('bot');

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

// ── EXP on Message ──────────────────────────────────────────────────────────

client.on('messageCreate', async (message) => {
  if (message.author.bot) return;
  try {
    const users = await db.loadUsers();
    const id = message.author.id;
    if (!users[id]) users[id] = { exp: 0, lastQuant: 0 };
    users[id].exp += 1;
    await db.saveUsers(users);
  } catch (err) {
    log.debug(`EXP save failed: ${err.message}`);
  }
});

// ── Command Handling ────────────────────────────────────────────────────────

client.on('interactionCreate', routeCommand);

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
  const recoveredCount = await jobQueue.recoverPersistedJobs();
  if (recoveredCount > 0) {
    log.info(`Recovered ${recoveredCount} persisted job(s) after startup`);
  }

  // 3. Recover API jobs (so restart doesn't orphan running Celery tasks)
  const apiRecovered = await recoverApiJobs();
  if (apiRecovered > 0) {
    log.info(`Recovered ${apiRecovered} API job(s) after startup`);
  }

  // 4. Login
  await client.login(config.BOT_TOKEN);
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
