import { Client, GatewayIntentBits, Collection } from 'discord.js';
import { config } from './config.js';
import { logger } from './utils/logger.js';
import * as workspace from './services/workspace.js';
import * as quantCmd from './commands/quant.js';
import * as statusCmd from './commands/status.js';

const client = new Client({ intents: [GatewayIntentBits.Guilds] });
client.commands = new Collection();

client.once('ready', () => {
  logger.info(`logged in as ${client.user.tag}`);
});

client.on('interactionCreate', async interaction => {
  if (!interaction.isChatInputCommand()) return;
  const cmd = client.commands.get(interaction.commandName);
  if (!cmd) return;
  try {
    await cmd.execute(interaction);
  } catch (e) {
    logger.error(e);
    await interaction.reply({ content: 'error', ephemeral: true });
  }
});

async function main() {
  await workspace.init();
  client.commands.set(quantCmd.data.name, quantCmd);
  client.commands.set(statusCmd.data.name, statusCmd);
  await client.login(config.discord.token);
}

main();
