// Who has already quantized what, across HF -- ours and everyone else's.
// Asked for in #quant-bot as "a public ledger" and "a centralised list ... of
// all the quants done by bot and/or community", by someone about to spend six
// hours uploading a quant that might already exist.
import config from '../config.js';

const HF = 'https://huggingface.co/api';

async function hfGet(path) {
  const resp = await fetch(`${HF}${path}`, {
    headers: config.HF_TOKEN ? { Authorization: `Bearer ${config.HF_TOKEN}` } : {},
  });
  if (!resp.ok) throw new Error(`HF ${resp.status} on ${path.split('?')[0]}`);
  return resp.json();
}

// The part of a quant's name before "exl3", lowercased.
const stem = (name) => name.split(/[-_.]exl3/i)[0].toLowerCase();

/**
 * Is this repo, found by name, a quant of `modelName`? Only when the part
 * before "exl3" is the model name exactly. Allowing a prefix (TandemLLM-Ornith-
 * 1.5-122B-EXL3) also let in every fine-tune named after its base --
 * Swift-Qwen3.8-27b listed as a Qwen3.8-27B quant -- so a differently named
 * repo has to come in through its base_model link instead.
 */
export function isQuantOf(repoId, modelName) {
  return stem(repoId.split('/').pop()) === modelName.toLowerCase();
}

/** What a quant's name says about it, as far as anyone's names say anything. */
export function describeQuant(repoId) {
  const name = repoId.split('/').pop();
  // "4.0bpw", "4-bpw", or a bare number right after exl3 (TandemLLM-...-EXL3-4.0).
  const bpw = name.match(/(\d+(?:\.\d+)?)\s*[-_]?bpw/i)?.[1]
    ?? name.match(/exl3[-_](\d+(?:\.\d+)?)(?:$|[-_])/i)?.[1] ?? null;
  const kind = /(^|[-_])SCT([-_]|$)/.test(name) ? 'SCT'
    : /(^|[-_])SC([-_]|$)|self[-_]?cal/i.test(name) ? 'SC' : null;
  return { id: repoId, bpw: bpw != null ? Number(bpw) : null, kind };
}

/**
 * Every EXL3 quant of `modelId` on HF. Two queries, because neither is enough
 * alone: the base_model link catches repos named anything, the name search
 * catches the many that never set the link.
 */
export async function findExl3Quants(modelId, { maxBranchLookups = 8, orgs = config.LEDGER_ORGS } = {}) {
  const modelName = modelId.split('/').pop();
  // The name search ranks by popularity and a base model can have 200
  // fine-tunes ahead of our own quants, so our orgs are asked by name too.
  const [linked, named, ...mine] = await Promise.all([
    hfGet(`/models?filter=${encodeURIComponent(`base_model:quantized:${modelId}`)}&limit=200&full=false`)
      .catch(() => []),
    hfGet(`/models?search=${encodeURIComponent(`${modelName}-exl3`)}&limit=200&full=false`).catch(() => []),
    ...orgs.map((o) => hfGet(`/models?author=${encodeURIComponent(o)}&search=${encodeURIComponent(modelName)}&limit=100&full=false`)
      .catch(() => [])),
  ]);
  const byId = new Map();
  for (const m of linked) if (/exl3/i.test(m.id)) byId.set(m.id, { ...m, linked: true });
  for (const m of [...named, ...mine.flat()]) {
    if (/exl3/i.test(m.id) && isQuantOf(m.id, modelName) && !byId.has(m.id)) byId.set(m.id, m);
  }
  const out = [...byId.values()].map((m) => ({
    ...describeQuant(m.id),
    downloads: m.downloads ?? 0,
    likes: m.likes ?? 0,
    updated: m.lastModified || m.createdAt || null,
    branches: null,
  }));
  // One repo per model with a branch per bitrate (turboderp's layout): the
  // name has no bpw, the branches do.
  for (const q of out.filter((x) => x.bpw == null).slice(0, maxBranchLookups)) {
    try {
      const refs = await hfGet(`/models/${q.id}/refs`);
      q.branches = (refs.branches || []).map((b) => b.name).filter((n) => n !== 'main');
    } catch {
      q.branches = null;
    }
  }
  return out.sort((a, b) => (a.bpw ?? 99) - (b.bpw ?? 99) || b.downloads - a.downloads);
}

/** Our own most recent quants, newest first, across every org we publish under. */
export async function recentBotQuants(orgs, limit = 15) {
  const lists = await Promise.all(orgs.map((org) =>
    hfGet(`/models?author=${encodeURIComponent(org)}&search=exl3&sort=lastModified&direction=-1&limit=${limit}`)
      .catch(() => [])));
  return lists.flat()
    .map((m) => ({ ...describeQuant(m.id), updated: m.lastModified || null, downloads: m.downloads ?? 0 }))
    .sort((a, b) => String(b.updated).localeCompare(String(a.updated)))
    .slice(0, limit);
}
