import { existsSync, readFileSync } from 'node:fs';
import { AttachmentBuilder, EmbedBuilder, MessageFlags } from 'discord.js';
import { getLogger } from '../logger.js';
import config from '../config.js';
import * as hf from '../services/huggingface.js';
import * as gallery from '../services/catbench.js';
import * as upstream from '../services/contribute.js';
import { preflight, runCatbench } from '../services/catbenchCli.js';
import { list as listControllers } from '../services/detached.js';
import { sanitizeSvg, renderSvgToPng } from '../utils/svg.js';
import { truncate } from '../utils/format.js';
import { toUserMessage } from '../errors/taxonomy.js';

const log = getLogger('cmd:catbench');

const COLOR = 0xff6b35; // upstream CatBench accent

// /catbench has no approval gate, so the throttles ARE the gate. One pod at a
// time across the whole bot, and one run per user per window.
let running = null; // modelId of the in-flight run
const lastRun = new Map(); // userId -> ms

/**
 * The model a run is already burning a pod on, or null. Controllers survive a
 * bot restart, so the in-memory flag alone would let the first /catbench after
 * a deploy boot a second pod next to the first.
 */
function inFlight() {
  if (running) return running;
  const c = listControllers().find((r) => r.kind === 'catbench' && r.running);
  return c ? c.meta?.modelId || c.id : null;
}

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

// A fresh run takes 12-25 minutes and an interaction token dies at 15, so the
// result usually lands after editReply has stopped working -- it throws and the
// user sits on the last progress state forever. Fall back to the channel, which
// is where someone who wandered off will look anyway.
//
// It logs `what` and where it landed, every time. The good endings used to log
// nothing at all, so a run that finished, posted "drew nothing usable" and
// stopped left a bot log holding one `spawn` line and nothing after it, which
// is indistinguishable from a run that disappeared and was read as one.
async function deliver(interaction, what, payload) {
  try {
    await interaction.editReply(payload);
    log.info(`${what} -> reply`);
    return true;
  } catch (err) {
    const ch = interaction.channel
      || (await interaction.client.channels.fetch(interaction.channelId).catch(() => null));
    if (!ch) {
      log.error(`${what} -> nowhere: token gone (${err.message}) and no channel`);
      return false;
    }
    const at = `<@${interaction.user.id}>`;
    try {
      await ch.send({ ...payload, content: payload.content ? `${at} ${payload.content}` : at });
      log.info(`${what} -> channel`);
      return true;
    } catch (e) {
      log.error(`${what} -> nowhere: ${e.message}`);
      return false;
    }
  }
}

/**
 * A cached hit's footer. Ours carries the date and the engine that made it: an
 * old run is still worth showing, but it should read as an old run rather than
 * as the current answer, and `refresh` is how you replace it.
 */
function cachedFooter(entry) {
  if (entry.source === 'upstream') return 'already in the CatBench gallery';
  const day = entry.at ? String(entry.at).slice(0, 10) : 'earlier';
  const engine = [entry.loader, entry.engine].filter(Boolean).join(' ');
  return `benched here ${day}${engine ? ` · ${engine}` : ''}`;
}

/**
 * Show an already-benched model. Upstream is a URL; ours attaches the mirrored
 * file when this box still has it and falls back to the dataset URL when it
 * doesn't, so a rebuilt VPS still answers from the runs it used to have.
 */
async function showCached(interaction, entry) {
  const files = [];
  const image = (file, url, name) => {
    if (file && existsSync(file)) {
      files.push(new AttachmentBuilder(file, { name }));
      return `attachment://${name}`;
    }
    return url || null;
  };
  const svgImage = image(entry.svgFile, entry.svg, 'svg.jpg');
  const pythonImage = image(entry.pythonFile, entry.python, 'python.jpg');

  await interaction.editReply({
    embeds: resultEmbeds({
      name: entry.name,
      svgImage,
      pythonImage,
      footer: cachedFooter(entry),
    }),
    files,
  });
}

/**
 * The two files upstream wants, for a run we already have. Local mirror first,
 * dataset URL when this box no longer holds it.
 */
async function storedSources(entry) {
  const load = async (file, url) => {
    if (file && existsSync(file)) return readFileSync(file, 'utf8');
    if (!url) return '';
    const res = await fetch(url, { signal: AbortSignal.timeout(20_000) });
    return res.ok ? res.text() : '';
  };
  return {
    svgSource: await load(entry.svgSourceFile, entry.svgSourceUrl),
    pythonSource: await load(entry.pythonSourceFile, entry.pythonSourceUrl),
    upstreamRenderOk: entry.upstreamRenderOk,
  };
}

/**
 * Offer one run to Katehuuh's gallery and say what happened, as a follow-up
 * rather than as part of the result: the kitten is the answer to /catbench,
 * and a contribution is a separate thing that can fail on its own.
 */
async function announce(interaction, modelId, sources) {
  let res;
  try {
    res = await upstream.contribute(modelId, sources);
  } catch (err) {
    log.warn(`contribute ${modelId} failed: ${err.message}`);
    res = { ok: false, why: err.message };
  }
  const line = !res.ok
    ? `📮 Not contributed: ${truncate(res.why, 160)}`
    : res.skipped
      ? `📮 \`${res.stem}\` is already in the gallery, nothing to send.`
      : `📮 ${res.added ? 'Pushed' : 'Already on the branch'} \`${res.stem}\` · ` +
        `${res.pending} waiting · open the PR when you want: <${res.url}>`;
  log.info(`contribute ${modelId}: ${line}`);
  await interaction.followUp({ content: line }).catch(async () => {
    const ch = interaction.channel
      || (await interaction.client.channels.fetch(interaction.channelId).catch(() => null));
    await ch?.send({ content: `<@${interaction.user.id}> ${line}` }).catch(() => {});
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
  const refresh = interaction.options.getBoolean('refresh') ?? false;
  const wantsPr = interaction.options.getBoolean('contribute') ?? false;
  // A repo can keep one quant per branch with a README on main, so the branch
  // is part of which model this is -- not a detail of how to fetch it.
  const optRevision = (interaction.options.getString('revision') || '').trim();
  // Resolved before the cache is consulted, not after: a pasted /tree/<branch>
  // URL names a different model than the same repo's default, and looking the
  // unpinned name up first would answer 4bpw with whatever branch we benched
  // last. The option wins over the URL -- someone who typed a branch meant it.
  let revision = optRevision;
  let parsed = '';
  try {
    const ref = hf.parseModelRef(input);
    parsed = ref.modelId;
    revision = optRevision || ref.revision;
  } catch {
    /* a bare gallery name rather than a model id; the roster can still serve it */
  }

  // No model: the roster. Keeps `/catbench <model>` exactly as specified while
  // still giving a zero-argument way to browse.
  if (!input) {
    await interaction.deferReply();
    return showList(interaction);
  }

  await interaction.deferReply();
  const userId = interaction.user.id;
  const isAdmin = config.ADMIN_IDS.includes(userId);

  // `refresh` skips the cache and re-benches. Admin only: the cache is the only
  // thing standing between /catbench and paying for the same kitten twice, so a
  // switch that turns it off is a switch that spends money.
  if (refresh && !isAdmin) {
    return interaction.editReply({
      content: 'Only an admin can force a re-run. A cached bench is a pod nobody has to pay for.',
    });
  }

  // Contributing opens a pull request against someone else's repository under
  // our account. That is not a thing a passer-by gets to do with a slash
  // command, so it is admin-only and off by default.
  if (wantsPr && !isAdmin) {
    return interaction.editReply({
      content: 'Only an admin can contribute a run to the CatBench gallery.',
    });
  }
  if (wantsPr && !upstream.enabled()) {
    return interaction.editReply({
      content: 'Contributing is not configured on this box (needs GITHUB_TOKEN and CATBENCH_FORK).',
    });
  }

  // Cached first, before any validation — an upstream hit needs no HF lookup,
  // no pod, no GPU.
  let entry = null;
  if (!refresh) {
    try {
      // The pinned identity when we have one, so a 4bpw request cannot be
      // answered with the 6bpw run.
      entry = await gallery.lookup(parsed ? gallery.modelRef(parsed, revision) : input);
    } catch (err) {
      log.warn(`gallery lookup failed: ${err.message}`);
    }
  }
  if (entry) {
    await showCached(interaction, entry);
    // A run we already have is contributable without a pod. Upstream's own
    // entries are not ours to send back, so they are skipped here rather than
    // rejected by the API three calls later.
    if (wantsPr && entry.source === 'ours') {
      await announce(interaction, entry.modelId || input, {
        ...(await storedSources(entry)),
        name: entry.name,
      });
    } else if (wantsPr) {
      await interaction
        .followUp({ content: '📮 That one is upstream\'s own entry, nothing to contribute.' })
        .catch(() => {});
    }
    return;
  }

  // Not benched: this will cost money, so everything below is a gate.
  const modelId = parsed;
  if (!modelId) {
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
  const busy = inFlight();
  if (busy) {
    return interaction.editReply({
      content: `A CatBench run is already going (\`${busy}\`). One pod at a time — try again when it lands.`,
    });
  }
  // Claim the slot before the size check, not after: the check takes seconds
  // and two clicks inside that window would otherwise both get through.
  const name = gallery.benchName(modelId, revision);
  running = gallery.modelRef(modelId, revision);
  const started = Date.now();
  try {
    // Size gate. Runs against the HF API, before a pod exists.
    const gate = await preflight(modelId, revision);
    if (!gate.ok) {
      return interaction.editReply({ content: `❌ ${gate.error}` });
    }
    lastRun.set(userId, Date.now());
    const status = (stage, message) =>
      interaction
        .editReply({
          embeds: [
            new EmbedBuilder()
              .setTitle(`🐱 Benching \`${truncate(name, 60)}\``)
              .setColor(COLOR)
              .setDescription(
                `**${stage}** · ${truncate(message || '…', 90)}\n` +
                  `~${gate.gb} GB of weights · ${Math.floor((Date.now() - started) / 60000)}m elapsed`
              ),
          ],
        })
        .catch(() => {});

    let stage = 'Provisioning';
    let detail = 'looking for the cheapest card that fits';
    await status(stage, detail);
    let lastEdit = Date.now();

    // The log only speaks when something matches, and the download is minutes
    // of `Fetching 42 files: 12%`, which matches nothing. Without a beat the
    // embed sits on "Provisioning ... 0m elapsed" for the whole run and looks
    // hung. Re-render on a timer so the minutes move even when the pod is quiet.
    const beat = setInterval(() => {
      if (Date.now() - lastEdit < 45_000) return;
      lastEdit = Date.now();
      status(stage, detail);
    }, 15_000);

    let result;
    try {
      result = await runCatbench({
        modelId,
        revision,
        onProgress: (p) => {
          stage = p.stage;
          detail = p.message;
          // Discord rate-limits edits; a slow cadence is plenty for a short run.
          if (Date.now() - lastEdit < 8000) return;
          lastEdit = Date.now();
          status(stage, detail);
        },
      });
    } finally {
      clearInterval(beat);
    }

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
        // warn, not debug: production runs at info, and an SVG that came back
        // and would not rasterize reports as "no render" with no trace of why.
        log.warn(`svg render failed: ${e.message}`);
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

    const mins = Math.round((result.wall_seconds || (Date.now() - started) / 1000) / 60);
    const cost = result.cost_usd != null ? ` · ~$${result.cost_usd.toFixed(2)}` : '';

    // A pod that answered with no kitten in it. Still an embed with the run's
    // cost and minutes on it: a bare line of text replacing the progress embed
    // reads as the bot losing its place, which is how a run that did say what
    // went wrong still counted as "got nothing".
    if (!svgImage && !pythonImage) {
      const why =
        `SVG: ${result.svg_error || 'no render'} · ` +
        `Python: ${result.python_error || 'no render'}`;
      return deliver(interaction, `${modelId} drew nothing usable (${why})`, {
        embeds: [
          new EmbedBuilder()
            .setTitle(`🐱 CatBench · ${truncate(name, 80)}`)
            .setColor(COLOR)
            .setURL(gallery.GALLERY_URL)
            .setDescription(`Nothing renderable came back.\n${why}`)
            .setFooter({ text: truncate(`fresh run · ${mins}m${cost} · not saved`, 200) }),
        ],
      });
    }

    // Only a clean both-halves run is worth keeping. A half-run cached is a
    // half-run nobody ever retries, and those are the ones a fix would fix.
    const grade = gallery.gradeRun(result, { svgPng, pythonPng });
    await deliver(interaction, `${modelId} benched in ${mins}m${cost}`, {
      embeds: resultEmbeds({
        name,
        svgImage,
        pythonImage,
        footer: grade.ok
          ? `fresh run · ${mins}m${cost}`
          : `fresh run · ${mins}m${cost} · not saved (${truncate(grade.why, 90)})`,
      }),
      files,
    });

    if (!grade.ok) {
      log.info(`not caching ${modelId}: ${grade.why}`);
    } else {
      // Remember it, so the next request for this model is free.
      await gallery
        .saveRun(modelId, {
          svgPng,
          pythonPng,
          svgSource: result.svg,
          pythonSource: result.python_source,
          loader: result.loader,
          engine: result.engine,
          format: result.format,
          prompts: result.prompts,
          revision,
          displayName: name,
          upstreamRenderOk: result.upstream_render_ok,
        })
        .catch((e) => log.warn(`could not cache result: ${e.message}`));
      if (wantsPr) {
        await announce(interaction, modelId, {
          svgSource: result.svg,
          pythonSource: result.python_source,
          upstreamRenderOk: result.upstream_render_ok,
          name,
        });
      }
    }
  } catch (err) {
    log.error(`catbench failed for ${modelId}`, { error: err.message });
    await deliver(interaction, `${modelId} failed`, {
      embeds: [],
      content: `❌ ${toUserMessage(err)}`,
    });
  } finally {
    running = null;
  }
}
