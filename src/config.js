import dotenv from 'dotenv';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: join(__dirname, '../.env') });

export const config = {
  discord: {
    token: process.env.DISCORD_TOKEN,
    clientId: process.env.DISCORD_CLIENT_ID,
    guildId: process.env.DISCORD_GUILD_ID
  },
  huggingface: {
    token: process.env.HF_TOKEN
  },
  workspace: {
    path: process.env.WORKSPACE_PATH || './.tmp',
    cache: process.env.HF_CACHE_PATH || './.tmp/cache'
  }
};
