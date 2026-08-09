import { readFile, writeFile, mkdir, rename } from 'fs/promises';
import path from 'path';
import config from '../config.js';
import { getLogger } from '../logger.js';

const log = getLogger('catbench');

const BASE = 'https://katehuuh.github.io/demos/CatBench';
const CONTENTS_API =
  'https://api.github.com/repos/Katehuuh/Katehuuh.github.io/contents/demos/CatBench/assets';

const dataDir = path.join(config.ROOT_DIR, 'data');
const imgDir = path.join(dataDir, 'catbench');
const LOCAL_FILE = path.join(dataDir, 'catbench.json');

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
// Runs this bot did aren't in Katehuuh's repo, so keep them here. Without this
// every repeat of the same model would burn another pod, and /catbench has no
// approval gate to stop that.

async function loadLocal() {
  try {
    return JSON.parse(await readFile(LOCAL_FILE, 'utf8'));
  } catch {
    return {};
  }
}

async function saveLocal(data) {
  await mkdir(dataDir, { recursive: true });
  const tmp = `${LOCAL_FILE}.tmp`;
  await writeFile(tmp, JSON.stringify(data, null, 2));
  await rename(tmp, LOCAL_FILE);
}

/** Persist a fresh run's two PNGs and index them under the model's key. */
export async function saveRun(modelId, { svgPng, pythonPng, note }) {
  const key = modelKey(modelId);
  await mkdir(imgDir, { recursive: true });
  const rec = { key, name: modelId, modelId, at: Date.now(), source: 'local', note: note || null };
  if (svgPng) {
    rec.svgFile = path.join(imgDir, `${key}-svg.png`);
    await writeFile(rec.svgFile, svgPng);
  }
  if (pythonPng) {
    rec.pythonFile = path.join(imgDir, `${key}-python.png`);
    await writeFile(rec.pythonFile, pythonPng);
  }
  const all = await loadLocal();
  all[key] = rec;
  await saveLocal(all);
  return rec;
}

/**
 * Find an existing bench for `modelId`. Upstream wins: it is the published
 * result and costs nothing to show. Returns null when nothing is benched.
 */
export async function lookup(modelId) {
  const key = modelKey(modelId);
  const up = await upstream();
  // Also accept the raw input as a key, so `/catbench glm-5.2` works alongside
  // `/catbench zai-org/GLM-5.2`.
  const hit = up.get(key) || up.get(normKey(modelId));
  if (hit && (hit.svg || hit.python)) return hit;
  const local = await loadLocal();
  return local[key] ?? null;
}

/** Everything benched, upstream plus ours, newest-looking first. */
export async function listAll() {
  const out = new Map();
  for (const [k, v] of await upstream()) out.set(k, v);
  for (const [k, v] of Object.entries(await loadLocal())) if (!out.has(k)) out.set(k, v);
  return [...out.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export const GALLERY_URL = `${BASE}/`;
