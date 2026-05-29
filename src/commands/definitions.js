import { SlashCommandBuilder } from 'discord.js';

export const commands = [
  new SlashCommandBuilder()
    .setName('quant')
    .setDescription('Request an EXL3 quantization (an admin approves it)')
    .addStringOption((opt) =>
      opt
        .setName('url')
        .setDescription('HuggingFace model URL or ID (e.g. meta-llama/Llama-3.1-8B-Instruct)')
        .setRequired(true)
    )
    .addStringOption((opt) =>
      opt
        .setName('bpw')
        .setDescription('Bits per weight, comma-separated (e.g. 3.0,4.0,5.0)')
        .setRequired(true)
    ),

  new SlashCommandBuilder().setName('queue').setDescription('Check the quantization queue status'),
  new SlashCommandBuilder().setName('health').setDescription('Check bot/service health summary'),

  new SlashCommandBuilder()
    .setName('history')
    .setDescription('Show recent quantization jobs'),

  new SlashCommandBuilder().setName('pause').setDescription('(Admin) Pause the job queue'),

  new SlashCommandBuilder().setName('resume').setDescription('(Admin) Resume the job queue'),
  new SlashCommandBuilder().setName('diag').setDescription('(Admin) Detailed diagnostics'),
  new SlashCommandBuilder()
    .setName('cache')
    .setDescription('(Admin) Cache operations')
    .addStringOption((opt) =>
      opt
        .setName('action')
        .setDescription('Operation to run')
        .setRequired(false)
        .addChoices(
          { name: 'Status', value: 'status' },
          { name: 'Prune', value: 'prune' },
          { name: 'Clear', value: 'clear' }
        )
    ),
];
