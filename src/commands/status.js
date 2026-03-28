import { SlashCommandBuilder } from 'discord.js';
import { db } from '../services/db.js';

export const data = new SlashCommandBuilder()
  .setName('status')
  .setDescription('Show queue status');

export async function execute(interaction) {
  const pending = db.data.queue.filter(j => j.status === 'pending').length;
  const running = db.data.queue.filter(j => j.status === 'running').length;
  await interaction.reply(`${pending} pending, ${running} running`);
}
