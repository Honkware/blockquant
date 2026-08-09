import { getLogger } from '../logger.js';
import { MessageFlags } from 'discord.js';
import { handleQuant } from './quant.js';
import { handleCatbench, autocompleteCatbench } from './catbench.js';
import { handleApproval } from './approval.js';
import { handleHealth, handleHistory, handleQueueStatus } from './info.js';
import { handleCache, handleDiag, handlePause, handleResume } from './admin.js';
import { toUserMessage } from '../errors/taxonomy.js';

const log = getLogger('commands');

const handlers = {
  quant: handleQuant,
  catbench: handleCatbench,
  queue: handleQueueStatus,
  health: handleHealth,
  history: handleHistory,
  pause: handlePause,
  resume: handleResume,
  diag: handleDiag,
  cache: handleCache,
};

/**
 * Route an incoming interaction to the correct handler.
 * Catches all errors so the bot never crashes from a command.
 */
export async function routeCommand(interaction) {
  // Approve / Deny buttons on pending quant requests.
  if (interaction.isButton()) {
    try {
      await handleApproval(interaction);
    } catch (err) {
      log.error('Error handling button', { error: err.message, stack: err.stack });
      try {
        if (!interaction.replied && !interaction.deferred) {
          await interaction.reply({ content: toUserMessage(err), flags: MessageFlags.Ephemeral });
        }
      } catch {
        // interaction expired
      }
    }
    return;
  }

  // Autocomplete on /catbench's model option. Must answer within 3s and can
  // never reply with anything but choices, so it gets its own short path.
  if (interaction.isAutocomplete()) {
    if (interaction.commandName !== 'catbench') return;
    try {
      await autocompleteCatbench(interaction);
    } catch (err) {
      log.debug(`autocomplete failed: ${err.message}`);
      await interaction.respond([]).catch(() => {});
    }
    return;
  }

  if (!interaction.isChatInputCommand()) return;

  const name = interaction.commandName;
  const handler = handlers[name];

  if (!handler) {
    log.warn(`No handler for command: ${name}`);
    return interaction.reply({ content: 'Unknown command.', flags: MessageFlags.Ephemeral });
  }

  log.info(`/${name} by ${interaction.user.tag} (${interaction.user.id})`);

  try {
    await handler(interaction);
  } catch (err) {
    log.error(`Error in /${name}`, {
      error: err.message,
      stack: err.stack,
      userId: interaction.user.id,
    });

    const msg = toUserMessage(err);
    try {
      if (interaction.deferred || interaction.replied) {
        await interaction.editReply({ content: msg, embeds: [], components: [] });
      } else {
        await interaction.reply({ content: msg, flags: MessageFlags.Ephemeral });
      }
    } catch {
      // interaction may have expired — nothing we can do
    }
  }
}
