import { ActivityType, Client, GatewayIntentBits, REST, Routes } from 'discord.js';
import config from './config.js';
import { getLogger } from './logger.js';
import * as db from './services/db.js';
import * as jobQueue from './services/queue.js';
import { commands } from './commands/definitions.js';
import { routeCommand } from './commands/router.js';

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

  // 3. Login
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
