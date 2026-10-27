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
        .setDescription('Bits per weight, comma-separated (e.g. 3.0,4.0,5.0), or low / mid / high / ladder')
        .setRequired(true)
    )
    .addStringOption((opt) =>
      opt
        .setName('prompt')
        .setDescription('Optional: a test prompt run on each finished quant; the reply shows in the result')
        .setRequired(false)
    )
    .addBooleanOption((opt) =>
      opt
        .setName('sc')
        .setDescription('Self-calibrated: measure this model and convert from a per-tensor recipe. Needs head_bits')
        .setRequired(false)
    )
    .addStringOption((opt) =>
      opt
        .setName('sc_mode')
        .setDescription('How much self-calibration to do. Implies sc. Default full')
        .setRequired(false)
        .addChoices(
          { name: 'full — measure every tensor, per-tensor recipe', value: 'full' },
          { name: 'trace — own-output calibration + -hq 2, no measure (big models)', value: 'trace' }
        )
    )
    .addStringOption((opt) =>
      opt
        .setName('subfolder')
        .setDescription('Subdirectory holding the model, for repos that ship several formats (e.g. BF16)')
        .setRequired(false)
    )
    .addIntegerOption((opt) =>
      opt
        .setName('gpu_count')
        .setDescription('GPUs on the pod. The converter runs one worker thread per device. Default 1.')
        .setRequired(false)
        .setMinValue(1)
        .setMaxValue(4)
    )
    .addIntegerOption((opt) =>
      opt
        .setName('vision_bits')
        .setDescription('Bits for the vision tower. 1-8, or 16 to copy it unquantized. Default depends on the arch.')
        .setRequired(false)
        .setMinValue(1)
        .setMaxValue(16)
    )
    .addIntegerOption((opt) =>
      opt
        .setName('head_bits')
        .setDescription('Bits for the output head. 1-8, or 16 to leave it unquantized. Default 6.')
        .setRequired(false)
        .setMinValue(1)
        .setMaxValue(16)
    )
    .addStringOption((opt) =>
      opt
        .setName('codebook')
        .setDescription('EXL3 trellis codebook (default auto)')
        .setRequired(false)
        .addChoices(
          { name: 'auto — mul1 for dense, mcg for MoE (default)', value: 'auto' },
          { name: 'mul1', value: 'mul1' },
          { name: 'mcg', value: 'mcg' },
          { name: '3inst', value: '3inst' }
        )
    )
    .addNumberOption((opt) =>
      opt
        .setName('mtp_bits')
        .setDescription('Bits for the MTP head: 1-8, 1.5/2.5/3.5, or 16. Default 4. Ignored without one')
        .setRequired(false)
        .setMinValue(1)
        .setMaxValue(16)
    )
    .addIntegerOption((opt) =>
      opt
        .setName('ngram_bits')
        .setDescription('Bits for n-gram embedding tables (PLE models like Qwen3.8-Flash-Next). Default: bpw rounded')
        .setRequired(false)
        .setMinValue(1)
        .setMaxValue(8)
    )
    .addStringOption((opt) =>
      opt
        .setName('draft_model')
        .setDescription('A drafter (e.g. a DFlash2 repo) to quantize at 4bpw beside the model, on the same pod')
        .setRequired(false)
    )
    .addIntegerOption((opt) =>
      opt
        .setName('hq')
        .setDescription('Promote select layers N times (MoE mostly). Default 0, or 2 with sc_mode:trace')
        .setRequired(false)
        .setMinValue(0)
        .setMaxValue(4)
    ),

  new SlashCommandBuilder()
    .setName('catbench')
    .setDescription('Ask a model for a cute kitten twice, and look at what comes out')
    .addStringOption((opt) =>
      opt
        .setName('model')
        .setDescription('A benched model, or a HuggingFace ID to run. Leave empty to list what is benched.')
        .setRequired(false)
        .setAutocomplete(true)
    )
    .addBooleanOption((opt) =>
      opt
        .setName('refresh')
        .setDescription('(Admin) Ignore the stored result and bench it again on a fresh pod')
        .setRequired(false)
    )
    .addStringOption((opt) =>
      opt
        .setName('revision')
        .setDescription('Branch to bench, e.g. 4.00bpw. For repos that keep one quant per branch.')
        .setRequired(false)
    )
    .addBooleanOption((opt) =>
      opt
        .setName('contribute')
        .setDescription('(Admin) Offer this result to the upstream CatBench gallery')
        .setRequired(false)
    ),

  new SlashCommandBuilder()
    .setName('stop')
    .setDescription('Stop a running quantization and terminate its pods')
    .addStringOption((opt) =>
      opt
        .setName('job')
        .setDescription('Job ID from the request card. Leave empty to stop your own running job.')
        .setRequired(false)
    ),

  new SlashCommandBuilder()
    .setName('config')
    .setDescription('(Admin) Bot settings')
    .addSubcommand((sub) =>
      sub
        .setName('quanter-role')
        .setDescription('Role whose members run /quant without approval')
        .addRoleOption((opt) =>
          opt
            .setName('role')
            .setDescription('The role. Leave empty to clear it and require approval from everyone.')
            .setRequired(false)
        )
    ),

  new SlashCommandBuilder()
    .setName('quants')
    .setDescription('Every EXL3 quant of a model on Hugging Face, ours and the community\'s')
    .addStringOption((opt) =>
      opt
        .setName('model')
        .setDescription('HuggingFace URL or ID. Leave empty for the bot\'s latest quants.')
        .setRequired(false)
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
