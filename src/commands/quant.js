import { SlashCommandBuilder } from 'discord.js';
import { queue } from '../services/queue.js';

export const data = new SlashCommandBuilder()
  .setName('quant')
  .setDescription('Queue a model for quantization')
  .addStringOption(o => o.setName('model').setDescription('HF model id').setRequired(true))
  .addIntegerOption(o =>
    o.setName('bits')
      .setDescription('Bits per weight')
      .addChoices({ name: '4', value: 4 }, { name: '6', value: 6 }, { name: '8', value: 8 })
  );

export async function execute(interaction) {
  const model = interaction.options.getString('model');
  const bits = interaction.options.getInteger('bits') || 8;
  const id = await queue.add({ type: 'quant', userId: interaction.user.id, model, bits });
  await interaction.reply(`queued \`${model}\` (${bits}bit) - id: \`${id.slice(0, 8)}\``);
}
