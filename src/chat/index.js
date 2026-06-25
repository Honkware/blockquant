import { Events } from 'discord.js';
import config from '../config.js';
import { getLogger } from '../logger.js';
import { kimiChat } from './kimi.js';

// Self-contained, optional chat module. Wire it up with one line in index.js
// (`registerChat(client)`). To remove the feature entirely: delete this folder,
// that one line, and the CHAT_*/KIMI_* keys from .env + config.js. It never
// touches the quant pipeline and is a no-op unless CHAT_ENABLED + a key are set.

const log = getLogger('chat');

const TURNS = 6; // user/assistant exchanges kept per channel for context
const MAX_INPUT = 2000; // ignore pasted walls of text
const DEFAULT_SYSTEM =
  'You are a friendly, concise assistant in the ExLlama Discord — ' +
  "turboderp's community around exllamav3 and EXL3 quantization, running " +
  'LLMs locally, and related topics. Answer briefly and helpfully. EXL3 ' +
  'quant jobs are run by this same bot via the /quant command, so point ' +
  'users there for that.';

// channelId -> [{ role, content }] (recent turns only, in memory)
const histories = new Map();

export function registerChat(client) {
  if (!config.CHAT_ENABLED) {
    log.info('chat module disabled (CHAT_ENABLED not set)');
    return;
  }
  if (!config.KIMI_API_KEY) {
    log.warn('CHAT_ENABLED but KIMI_API_KEY is missing — chat stays off');
    return;
  }

  const system = config.CHAT_SYSTEM_PROMPT || DEFAULT_SYSTEM;
  const allow = new Set(
    (config.CHAT_CHANNELS || '').split(',').map((s) => s.trim()).filter(Boolean)
  );

  client.on(Events.MessageCreate, async (msg) => {
    try {
      if (msg.author.bot || !client.user) return;
      if (allow.size && !allow.has(msg.channelId)) return;

      // Trigger on an @mention of the bot, or a reply to one of its messages.
      const mentioned = msg.mentions.users.has(client.user.id);
      let isReplyToBot = false;
      if (!mentioned && msg.reference?.messageId) {
        const ref = await msg.fetchReference().catch(() => null);
        isReplyToBot = ref?.author?.id === client.user.id;
      }
      if (!mentioned && !isReplyToBot) return;

      const content = msg.content
        .replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '')
        .trim();
      if (!content || content.length > MAX_INPUT) return;

      const hist = histories.get(msg.channelId) || [];
      const messages = [
        { role: 'system', content: system },
        ...hist,
        { role: 'user', content },
      ];

      await msg.channel.sendTyping().catch(() => {});
      const reply = await kimiChat(messages);
      if (!reply) return;

      // Keep only the last TURNS exchanges so context (and cost) stay bounded.
      histories.set(
        msg.channelId,
        [...hist, { role: 'user', content }, { role: 'assistant', content: reply }].slice(-TURNS * 2)
      );

      // Discord caps a message at 2000 chars.
      await msg.reply(reply.length > 1900 ? reply.slice(0, 1900) + '…' : reply);
    } catch (e) {
      log.warn(`chat error: ${e.message}`);
      await msg.reply('⚠️ chat hiccup — try again in a moment.').catch(() => {});
    }
  });

  log.info(`chat module enabled (Kimi: ${config.KIMI_MODEL})`);
}
