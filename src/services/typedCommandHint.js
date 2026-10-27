// A slash command typed as plain text goes out as a message and nothing runs.
// ccobaltdev posted two full /quant requests that way on 10-17 -- SC, head and
// vision bits, three bitrates -- and no job ever existed. A real slash command
// never arrives as a message, so anything that looks like one is this mistake.
import { getLogger } from '../logger.js';

const log = getLogger('typed-command');

const TYPED = /^\s*\.?\/?(quant|quants|catbench)\s+\w+:/i;

export function looksLikeTypedCommand(content) {
  return TYPED.test(content || '');
}

export function registerTypedCommandHint(client) {
  client.on('messageCreate', (msg) => {
    if (msg.author?.bot || !looksLikeTypedCommand(msg.content)) return;
    const name = msg.content.match(TYPED)[1].toLowerCase();
    log.info(`typed /${name} as text from ${msg.author.username}`);
    msg.reply({
      content: `That went out as a message, so nothing ran. Type \`/${name}\` and pick it from `
        + 'the menu that pops up -- the options become fields you fill in.',
      allowedMentions: { repliedUser: false },
    }).catch((err) => log.debug(`hint reply failed: ${err.message}`));
  });
}
