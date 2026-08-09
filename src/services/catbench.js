import { readFile, stat } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import config from '../config.js';
import { getLogger } from '../logger.js';
import { storeRun } from './catbenchCli.js';

const log = getLogger('catbench');

const BASE = 'https://katehuuh.github.io/demos/CatBench';
const CONTENTS_API =
  'https://api.github.com/repos/Katehuuh/Katehuuh.github.io/contents/demos/CatBench/assets';

const dataDir = path.join(config.ROOT_DIR, 'data');
export const MIRROR_DIR = path.join(dataDir, 'catbench');
const MIRROR_FILE = path.join(MIRROR_DIR, 'manifest.json');

/** Raw-file base for our dataset, the URL Discord embeds from. */
export function datasetBase(repo = config.CATBENCH_DATASET) {
  return repo ? `https://huggingface.co/datasets/${repo}/resolve/main` : '';
}

// Upstream's own key rule (demos/CatBench/index.html:normKey). Match it exactly
// or our lookups miss entries the gallery shows.
export function normKey(stem) {
  return String(stem).toLowerCase().replace(/[_\s]+/g, '-');
}

/** Key a HF id the way upstream keys a file stem: the bit after the slash. */
export function modelKey(modelId) {
  return normKey(String(modelId).split('/').pop());
}

// ── Upstream gallery ────────────────────────────────────────────────────────

let cache = { at: 0, entries: null };
const TTL_MS = 10 * 60 * 1000;

async function getJson(url, timeoutMs = 8000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ctl.signal, headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(`${url} -> ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Upstream publishes `demos/CatBench/manifest.json`: one object per model with
 * `display_name` plus asset paths for `svg` (a rasterized jpg/gif), `python_render`
 * (jpg), and the `.svg` / `.py` sources. That is the index — one request, no
 * rate limit — so we read it rather than listing the directory.
 *
 * The contents API is the fallback the demo page itself uses when the manifest
 * is missing or stale. Filenames there encode the model and the half:
 * `<Stem>-svg.jpg`, `<Stem>-python.jpg`, `<Stem>.svg`, `<Stem>.py`.
 */
async function fetchUpstream() {
  const entries = new Map();
  try {
    const manifest = await getJson(`${BASE}/manifest.json`);
    for (const [key, m] of Object.entries(manifest.models || {})) {
      entries.set(key, {
        key,
        name: m.display_name || key,
        svg: m.svg ? `${BASE}/${m.svg}` : null,
        python: m.python_render ? `${BASE}/${m.python_render}` : null,
        source: 'upstream',
      });
    }
  } catch (err) {
    log.warn(`CatBench manifest unavailable (${err.message}); listing assets instead`);
  }

  if (entries.size) return entries;

  const files = await getJson(CONTENTS_API);
  if (!Array.isArray(files)) return entries;
  for (const f of files) {
    if (f.type !== 'file') continue;
    const dot = f.name.lastIndexOf('.');
    if (dot < 1) continue;
    const stem = f.name.slice(0, dot);
    const ext = f.name.slice(dot + 1).toLowerCase();
    if (!['png', 'jpg', 'jpeg', 'gif'].includes(ext)) continue; // sources aren't postable
    const half = normKey(stem).endsWith('-svg') ? 'svg' : normKey(stem).endsWith('-python') ? 'python' : null;
    if (!half) continue;
    const base = stem.replace(/-(svg|python)$/i, '');
    const key = normKey(base);
    const e = entries.get(key) || { key, name: base.replace(/[_\s]+/g, ' '), svg: null, python: null, source: 'upstream' };
    e[half] ||= `${BASE}/assets/${f.name}`;
    entries.set(key, e);
  }
  return entries;
}

async function upstream() {
  if (cache.entries && Date.now() - cache.at < TTL_MS) return cache.entries;
  try {
    const entries = await fetchUpstream();
    if (entries.size) cache = { at: Date.now(), entries };
  } catch (err) {
    log.warn(`CatBench gallery fetch failed: ${err.message}`);
  }
  return cache.entries || new Map();
}

// ── Our own results ─────────────────────────────────────────────────────────
// Runs this bot did aren't in Katehuuh's repo, so they go in our own dataset,
// `Honkware/catbench-results`, written by backend/scripts/catbench_store.py.
// The dataset is the durable copy; data/catbench is a mirror of it on the VPS,
// so the common lookup is a disk read and costs nothing. Without either, every
// repeat of the same model burns another pod, and /catbench has no approval
// gate to stop that.

/** One manifest entry, ours, as the shape the command renders. */
function ourEntry(key, m, urlBase, dir) {
  const at = m.run_date || m._first_seen || null;
  const file = (rel) => (dir && rel && existsSync(path.join(dir, rel)) ? path.join(dir, rel) : null);
  const url = (rel) => (urlBase && rel ? `${urlBase}/${rel}` : null);
  return {
    key,
    name: m.display_name || m.model_id || key,
    modelId: m.model_id || null,
    // Local copy when the box still has it, the dataset URL when it doesn't.
    svgFile: file(m.svg),
    pythonFile: file(m.python_render),
    svg: url(m.svg),
    python: url(m.python_render),
    loader: m.loader || '',
    engine: m.engine || '',
    at,
    source: 'ours',
  };
}

/** Index a manifest by key and, so a re-run of an exact repo always wins, by id. */
function indexOurs(manifest, urlBase, dir) {
  const byKey = new Map();
  const byModel = new Map();
  for (const [key, m] of Object.entries(manifest?.models || {})) {
    if (!m || (!m.svg && !m.python_render)) continue;
    const entry = ourEntry(key, m, urlBase, dir);
    if (!entry.svgFile && !entry.pythonFile && !entry.svg && !entry.python) continue;
    byKey.set(key, entry);
    if (entry.modelId) byModel.set(entry.modelId.toLowerCase(), entry);
  }
  return { byKey, byModel };
}

const EMPTY = { byKey: new Map(), byModel: new Map() };

// Memoized on the manifest's mtime: a hit re-reads nothing, a write is picked
// up on the next lookup without anyone having to invalidate anything.
let mirrorCache = { mtime: -1, index: EMPTY };

async function mirror() {
  let mtime;
  try {
    mtime = (await stat(MIRROR_FILE)).mtimeMs;
  } catch {
    return EMPTY;
  }
  if (mtime === mirrorCache.mtime) return mirrorCache.index;
  try {
    const manifest = JSON.parse(await readFile(MIRROR_FILE, 'utf8'));
    mirrorCache = { mtime, index: indexOurs(manifest, datasetBase(), MIRROR_DIR) };
  } catch (err) {
    log.warn(`local CatBench manifest unreadable: ${err.message}`);
    mirrorCache = { mtime, index: EMPTY };
  }
  return mirrorCache.index;
}

let dsCache = { at: 0, index: null };

/**
 * The dataset's own manifest. The mirror covers this in normal operation; this
 * is what makes a rebuilt VPS, or a second bot, see runs it never did itself.
 */
async function dataset() {
  const base = datasetBase();
  if (!base) return EMPTY;
  if (dsCache.index && Date.now() - dsCache.at < TTL_MS) return dsCache.index;
  try {
    const index = indexOurs(await getJson(`${base}/manifest.json`), base, null);
    dsCache = { at: Date.now(), index };
  } catch (err) {
    // A dataset with no manifest.json yet 404s. That is the normal state until
    // the first run lands, so it is a debug line, not a warning.
    log.debug(`CatBench dataset manifest unavailable: ${err.message}`);
    // Stamp it anyway: until the first run lands there is no manifest to read,
    // and retrying that 404 on every lookup would put a round trip back on the
    // fast path we are trying to keep free.
    dsCache = { at: Date.now(), index: dsCache.index || EMPTY };
  }
  return dsCache.index;
}

/**
 * Our entry for this input, or null.
 *
 * An exact model id always wins. A bare stem falls back to the upstream key
 * rule, because that is what someone typing `glm-5.2` means. A *different*
 * org's fully-qualified id does not: `someone/Qwen3-8B` is a fine-tune, not
 * Qwen's Qwen3-8B, and showing Qwen's kitten for it would be a lie. Upstream
 * keeps the loose rule; its manifest records no repo to be strict about.
 */
function pick(index, input) {
  const raw = String(input);
  const exact = index.byModel.get(raw.toLowerCase());
  if (exact) return exact;
  if (raw.includes('/')) return null;
  return index.byKey.get(modelKey(raw)) || index.byKey.get(normKey(raw)) || null;
}

/**
 * Find an existing bench for `modelId`, or null when nothing is benched.
 *
 * Ours first, then upstream. Ours is a disk read on the hot path, and it has
 * to win outright: an admin who forces a re-run of a model upstream also has
 * would otherwise never see their own result. The two remote reads run
 * together, so an upstream hit still costs one round trip, not two.
 */
export async function lookup(modelId) {
  const local = pick(await mirror(), modelId);
  if (local) return local;
  const [ds, up] = await Promise.all([dataset(), upstream()]);
  const mine = pick(ds, modelId);
  if (mine) return mine;
  const hit = up.get(modelKey(modelId)) || up.get(normKey(modelId));
  return hit && (hit.svg || hit.python) ? hit : null;
}

// ── Grading ─────────────────────────────────────────────────────────────────

// A blank 640px canvas is a few hundred bytes of PNG; anything with a kitten
// in it clears this by an order of magnitude.
const MIN_RASTER_BYTES = 1024;
// <g> and <defs> are not marks. One of these is.
const DRAWS = /<(path|circle|ellipse|rect|polygon|polyline|line|text|image)\b/i;

/**
 * Is this run a result, or a thing that went wrong?
 *
 * Only a clean both-halves run is storable. A crashed script, an SVG that
 * renders blank and prose where the code should be all look like results once
 * they are in a cache, and nobody re-runs a cache hit. Those are exactly the
 * runs someone will want to retry after a fix, so they stay uncached.
 */
export function gradeRun(result = {}, { svgPng, pythonPng } = {}) {
  const why = [];
  if (!result.svg) why.push(result.svg_error || 'no SVG in the reply');
  else if (!DRAWS.test(result.svg)) why.push('the SVG has no shapes in it');
  else if (!svgPng) why.push('the SVG would not rasterize');
  else if (svgPng.length < MIN_RASTER_BYTES) why.push('the SVG rendered blank');

  if (!result.python_source) why.push(result.python_error || 'no python in the reply');
  else if (!pythonPng) why.push(result.python_error || 'the script drew nothing');
  else if (result.python_error) why.push(`the script crashed: ${String(result.python_error).slice(0, 120)}`);

  return { ok: !why.length, why: why.join('; ') };
}

// ── Writing ─────────────────────────────────────────────────────────────────

/**
 * Hand a graded run to catbench_store.py: mirror on disk, dataset on HF.
 * The caller has already run gradeRun; the store re-checks, because the one
 * thing worse than not caching a good run is caching a bad one.
 */
export async function saveRun(modelId, artifacts) {
  const rec = await storeRun({ ...artifacts, model_id: modelId });
  // Drop the memo: the store just rewrote the manifest and mtime resolution
  // is coarse enough that a same-millisecond write could otherwise be missed.
  mirrorCache = { mtime: -1, index: EMPTY };
  if (config.CATBENCH_DATASET && !rec.uploaded) {
    // Cached here, not published. The next run that reaches HF carries it.
    log.warn(`${modelId} stored locally but not pushed: ${rec.note || 'no reason given'}`);
  }
  return rec;
}

/** Everything benched, ours plus upstream, alphabetical. Ours wins a tie. */
export async function listAll() {
  const [up, ds, mine] = await Promise.all([upstream(), dataset(), mirror()]);
  const out = new Map(up);
  for (const index of [ds, mine]) for (const [k, v] of index.byKey) out.set(k, v);
  return [...out.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export const GALLERY_URL = `${BASE}/`;
export const DATASET_URL = config.CATBENCH_DATASET
  ? `https://huggingface.co/datasets/${config.CATBENCH_DATASET}`
  : '';
