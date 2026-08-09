import { AttachmentBuilder, EmbedBuilder, MessageFlags } from 'discord.js';
import { getLogger } from '../logger.js';
import config from '../config.js';
import * as hf from '../services/huggingface.js';
import * as gallery from '../services/catbench.js';
import { preflight, runCatbench } from '../services/catbenchCli.js';
import { sanitizeSvg, renderSvgToPng } from '../utils/svg.js';
import { truncate } from '../utils/format.js';
import { toUserMessage } from '../errors/taxonomy.js';

const log = getLogger('cmd:catbench');

const COLOR = 0xff6b35; // upstream CatBench accent

// /catbench has no approval gate, so the throttles ARE the gate. One pod at a
// time across the whole bot, and one run per user per window.
let running = null; // modelId of the in-flight run
const lastRun = new Map(); // userId -> ms

function cooldownLeft(userId) {
  if (config.ADMIN_IDS.includes(userId)) return 0;
  const prev = lastRun.get(userId);
  if (!prev) return 0;
  return Math.max(0, config.CATBENCH_COOLDOWN_MS - (Date.now() - prev));
}

/**
 * Both halves as images, nothing else. The raw SVG markup and the generated
 * Python are deliberately not posted: they are long, they are model output, and
 * the benchmark is about what the kitten LOOKS like.
 */
function resultEmbeds({ name, svgImage, pythonImage, footer }) {
  const head = new EmbedBuilder()
    .setTitle(`🐱 CatBench · ${truncate(name, 80)}`)
    .setColor(COLOR)
    .setURL(gallery.GALLERY_URL);
  const embeds = [];
  if (svgImage) embeds.push(head.setImage(svgImage).setDescription('**SVG** · asked for one directly'));
  else embeds.push(head.setDescription('**SVG** · nothing renderable came back'));
  if (pythonImage) {
    // No .setURL here on purpose: Discord merges consecutive embeds that share
    // a url into one, which would swallow this caption and leave both kittens
    // unlabelled.
    embeds.push(
      new EmbedBuilder()
        .setColor(COLOR)
        .setDescription('**Python** · matplotlib, run in a sandbox')
        .setImage(pythonImage)
    );
  }
  if (footer) embeds[embeds.length - 1].setFooter({ text: truncate(footer, 200) });
  return embeds;
}

/** Show an already-benched model. Upstream entries are URLs, ours are files. */
async function showCached(interaction, entry) {
  const files = [];
  let svgImage = null;
  let pythonImage = null;

  if (entry.source === 'upstream') {
    svgImage = entry.svg;
    pythonImage = entry.python;
  } else {
    if (entry.svgFile) {
      files.push(new AttachmentBuilder(entry.svgFile, { name: 'svg.png' }));
      svgImage = 'attachment://svg.png';
    }
    if (entry.pythonFile) {
      files.push(new AttachmentBuilder(entry.pythonFile, { name: 'python.png' }));
      pythonImage = 'attachment://python.png';
    }
  }

  await interaction.editReply({
    embeds: resultEmbeds({
      name: entry.name,
      svgImage,
      pythonImage,
      footer: entry.source === 'upstream' ? 'already in the CatBench gallery' : 'benched here earlier',
    }),
    files,
  });
}

async function showList(interaction) {
  const all = await gallery.listAll();
  if (!all.length) {
    return interaction.editReply({
      content: `Could not reach the CatBench gallery. ${gallery.GALLERY_URL}`,
    });
  }
  // Discord caps a description at 4096; the roster is short but grow-proof it.
  const names = all.map((e) => `\`${e.name}\``);
  let body = names.join(' · ');
  if (body.length > 3900) body = `${body.slice(0, 3900)}…`;

  const embed = new EmbedBuilder()
    .setTitle(`🐱 Benched already · ${all.length}`)
    .setColor(COLOR)
    .setURL(gallery.GALLERY_URL)
    .setDescription(body)
    .setFooter({ text: 'Any of these come back instantly. Anything else runs on a pod.' });
  return interaction.editReply({ embeds: [embed] });
}

/**
 * Autocomplete the model argument with what is already benched, so the free,
 * instant path is the one that surfaces while you type. Anything not in the
 * list is still accepted — the option is a plain string, the choices are only
 * a hint.
 */
export async function autocompleteCatbench(interaction) {
  const typed = (interaction.options.getFocused() || '').toLowerCase();
  let all = [];
  try {
    all = await gallery.listAll();
  } catch {
    /* offline: no suggestions, the command still works */
  }
  const hits = all
    .filter((e) => !typed || e.name.toLowerCase().includes(typed) || e.key.includes(typed))
    .slice(0, 25)
    .map((e) => ({ name: `${e.name} (benched)`.slice(0, 100), value: e.name.slice(0, 100) }));
  return interaction.respond(hits);
}

export async function handleCatbench(interaction) {
  const input = (interaction.options.getString('model') || '').trim();

  // No model: the roster. Keeps `/catbench <model>` exactly as specified while
  // still giving a zero-argument way to browse.
  if (!input) {
    await interaction.deferReply();
    return showList(interaction);
  }

  await interaction.deferReply();
  const userId = interaction.user.id;

  // Cached first, before any validation — an upstream hit needs no HF lookup,
  // no pod, no GPU.
  let entry = null;
  try {
    entry = await gallery.lookup(input);
  } catch (err) {
    log.warn(`gallery lookup failed: ${err.message}`);
  }
  if (entry) return showCached(interaction, entry);

  // Not benched: this will cost money, so everything below is a gate.
  let modelId;
  try {
    modelId = hf.parseModelId(input);
  } catch {
    return interaction.editReply({
      content:
        `\`${truncate(input, 60)}\` is not in the gallery and is not a HuggingFace model ID. ` +
        'Give me `org/model`, or run `/catbench` with no model to see what is benched.',
    });
  }

  const wait = cooldownLeft(userId);
  if (wait) {
    return interaction.editReply({
      content: `You benched a model recently. Try again in ${Math.ceil(wait / 60000)} min.`,
    });
  }
  if (running) {
    return interaction.editReply({
      content: `A CatBench run is already going (\`${running}\`). One pod at a time — try again when it lands.`,
    });
  }
  // Claim the slot before the size check, not after: the check takes seconds
  // and two clicks inside that window would otherwise both get through.
  running = modelId;
  const started = Date.now();
  try {
    // Size gate. Runs against the HF API, before a pod exists.
    const gate = await preflight(modelId);
    if (!gate.ok) {
      return interaction.editReply({ content: `❌ ${gate.error}` });
    }
    lastRun.set(userId, Date.now());
    const status = (stage, message) =>
      interaction
        .editReply({
          embeds: [
            new EmbedBuilder()
              .setTitle(`🐱 Benching \`${truncate(modelId, 60)}\``)
              .setColor(COLOR)
              .setDescription(
                `**${stage}** · ${truncate(message || '…', 90)}\n` +
                  `~${gate.gb} GB of weights · ${Math.floor((Date.now() - started) / 60000)}m elapsed`
              ),
          ],
        })
        .catch(() => {});

    await status('Provisioning', 'looking for the cheapest card that fits');
    let lastEdit = 0;
    const result = await runCatbench({
      modelId,
      onProgress: ({ stage, message }) => {
        // Discord rate-limits edits; a slow cadence is plenty for a short run.
        if (Date.now() - lastEdit < 8000) return;
        lastEdit = Date.now();
        status(stage, message);
      },
    });

    // Rasterize the SVG here, with resvg: a static renderer with no script
    // engine and no network, fed sanitized markup. Posting the .svg itself is
    // not an option — Discord will not render an SVG attachment inline, it
    // shows a file stub — so the embed gets a PNG.
    const files = [];
    let svgImage = null;
    let svgPng = null;
    if (result.svg) {
      try {
        svgPng = renderSvgToPng(sanitizeSvg(result.svg), 640);
        files.push(new AttachmentBuilder(svgPng, { name: 'svg.png' }));
        svgImage = 'attachment://svg.png';
      } catch (e) {
        log.debug(`svg render failed: ${e.message}`);
      }
    }

    // The matplotlib half was already rendered on the pod, inside the sandbox.
    let pythonImage = null;
    let pythonPng = null;
    if (result.python_png_b64) {
      pythonPng = Buffer.from(result.python_png_b64, 'base64');
      files.push(new AttachmentBuilder(pythonPng, { name: 'python.png' }));
      pythonImage = 'attachment://python.png';
    }

    if (!svgImage && !pythonImage) {
      return interaction.editReply({
        embeds: [],
        content:
          `\`${modelId}\` drew nothing usable. ` +
          `SVG: ${result.svg_error || 'no render'} · Python: ${result.python_error || 'no render'}`,
      });
    }

    const mins = Math.round((result.wall_seconds || (Date.now() - started) / 1000) / 60);
    const cost = result.cost_usd != null ? ` · ~$${result.cost_usd.toFixed(2)}` : '';
    await interaction.editReply({
      embeds: resultEmbeds({
        name: modelId,
        svgImage,
        pythonImage,
        footer: `fresh run · ${mins}m${cost}`,
      }),
      files,
    });

    // Remember it, so the next request for this model is free.
    await gallery
      .saveRun(modelId, { svgPng, pythonPng, note: result.python_error || result.svg_error })
      .catch((e) => log.warn(`could not cache result: ${e.message}`));
  } catch (err) {
    log.error(`catbench failed for ${modelId}`, { error: err.message });
    await interaction.editReply({ embeds: [], content: `❌ ${toUserMessage(err)}` }).catch(() => {});
  } finally {
    running = null;
  }
}
