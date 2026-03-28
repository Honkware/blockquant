import { SlashCommandBuilder } from 'discord.js';
import { db } from '../services/db.js';

export const data = new SlashCommandBuilder()
  .setName('info')
  .setDescription('Your stats');

export async function execute(interaction) {
  const user = db.data.users[interaction.user.id] || { exp: 0, jobs: 0 };
  await interaction.reply(`${interaction.user.username}: ${user.exp} XP, ${user.jobs} jobs`);
}
