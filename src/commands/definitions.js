import { SlashCommandBuilder } from 'discord.js';

export const commands = [
  new SlashCommandBuilder()
    .setName('quant')
    .setDescription('Quantize a HuggingFace model to EXL3 format')
    .addStringOption((opt) =>
      opt
        .setName('url')
        .setDescription('HuggingFace model URL or ID (e.g. meta-llama/Llama-3.1-8B-Instruct)')
        .setRequired(true)
    )
    .addStringOption((opt) => {
      opt
        .setName('bpw')
        .setDescription('Bits per weight — comma-separated (e.g. 3.0,4.0,5.0)')
        .setRequired(true);
      return opt;
    })
    .addStringOption((opt) =>
      opt
        .setName('profile')
        .setDescription('Quantization profile (default: Auto - scales head_bits with BPW)')
        .setRequired(false)
        .addChoices(
          { name: '🤖 Auto (recommended)', value: 'auto' },
          { name: '⚡ Fast', value: 'fast' },
          { name: '⚖️ Balanced', value: 'balanced' },
          { name: '✨ Quality', value: 'quality' }
        )
    )
    .addIntegerOption((opt) =>
      opt
        .setName('head_bits')
        .setDescription('Advanced: override head bits (admin only)')
        .setRequired(false)
        .setMinValue(2)
        .setMaxValue(16)
    )
    .addStringOption((opt) =>
      opt
        .setName('category')
        .setDescription('Model category (optional)')
        .setRequired(false)
        .addChoices(
          { name: '💬 Chat / Assistant', value: 'Chat' },
          { name: '✍️ Writing', value: 'Writing' },
          { name: '💻 Code', value: 'Code' },
          { name: '🔬 Science / Research', value: 'Science' },
          { name: '🎭 Roleplay', value: 'Roleplay' },
          { name: '📊 General Purpose', value: 'General' }
        )
    ),

  new SlashCommandBuilder().setName('queue').setDescription('Check the quantization queue status'),
  new SlashCommandBuilder().setName('health').setDescription('Check bot/service health summary'),

  new SlashCommandBuilder()
    .setName('score')
    .setDescription("Check a user's EXP balance")
    .addUserOption((opt) =>
      opt.setName('user').setDescription('User to check (defaults to you)').setRequired(false)
    ),

  new SlashCommandBuilder().setName('leaderboard').setDescription('Show the top EXP earners'),

  new SlashCommandBuilder()
    .setName('give')
    .setDescription('(Admin) Give EXP to a user')
    .addUserOption((opt) => opt.setName('user').setDescription('Recipient').setRequired(true))
    .addNumberOption((opt) => opt.setName('amount').setDescription('EXP amount').setRequired(true)),

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
