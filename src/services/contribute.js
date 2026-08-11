import config from '../config.js';
import { getLogger } from '../logger.js';
import { normKey } from './catbench.js';

const log = getLogger('contribute');

const API = 'https://api.github.com';

/**
 * Push benched models back to Katehuuh's CatBench gallery.
 *
 * Upstream's build Action does the rest: on a push to its default branch it
 * renders every `<model>.py`, rasterizes every `<model>.svg`, regenerates
 * manifest.json and commits the lot back. So a contribution is two source
 * files and nothing else -- shipping our own rasters would register phantom
 * models (`<model>-python.jpg` keys as a model named `<model>-python`) and a
 * hand-written manifest entry is overwritten on the next push anyway.
 *
 * Everything accumulates on one branch in our fork. The maintainer gets one
 * pull request that grows, rather than one per bench.
 */

/** Is contributing configured at all? */
export function enabled() {
  return Boolean(config.GITHUB_TOKEN && config.CATBENCH_FORK && config.CATBENCH_UPSTREAM);
}

/**
 * The filename upstream should carry for a model.
 *
 * Upstream keys a model by its filename stem, so the owner has to go: they
 * have no field for it and `Honkware/X` would key as a model called
 * `honkware`. BF16 goes too -- it describes what we quantized *from*, which
 * says nothing about the model that drew the kitten. The quant level stays,
 * because at 4.5bpw that is the thing being benchmarked.
 */
export function upstreamStem(modelId, name = '') {
  // `name` is the benched name, which already carries the branch for a repo
  // that keeps one quant per branch. Falling back to the repo stem would send
  // 4bpw and 6bpw upstream under one filename.
  const stem = name || String(modelId).split('/').pop();
  return stem.replace(/[-_]bf16(?=[-_]|$)/i, '').replace(/^bf16[-_]/i, '');
}

// ── GitHub ──────────────────────────────────────────────────────────────────

async function gh(path, { method = 'GET', body, allow = [] } = {}) {
  const res = await fetch(path.startsWith('http') ? path : `${API}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${config.GITHUB_TOKEN}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'blockquant-catbench',
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(30_000),
  });
  if (allow.includes(res.status)) return null;
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    let detail = text.slice(0, 200);
    try {
      const j = JSON.parse(text);
      detail = j.message + (j.errors ? ` (${JSON.stringify(j.errors).slice(0, 120)})` : '');
    } catch {
      /* not JSON */
    }
    throw new Error(`${method} ${path} -> ${res.status}: ${detail}`);
  }
  return res.status === 204 ? null : res.json();
}

/** The upstream branch we base on and target. Never assume `main`. */
async function upstreamBase() {
  const repo = await gh(`/repos/${config.CATBENCH_UPSTREAM}`);
  const ref = await gh(`/repos/${config.CATBENCH_UPSTREAM}/git/ref/heads/${repo.default_branch}`);
  return { branch: repo.default_branch, sha: ref.object.sha };
}

/**
 * Our contribution branch, created off upstream's head if it does not exist.
 *
 * An existing branch is left where it is rather than reset: it holds models
 * from earlier benches that the open pull request is already showing.
 */
async function ensureBranch(base) {
  const ref = `heads/${config.CATBENCH_BRANCH}`;
  const have = await gh(`/repos/${config.CATBENCH_FORK}/git/ref/${ref}`, { allow: [404] });
  if (have) return false;
  await gh(`/repos/${config.CATBENCH_FORK}/git/refs`, {
    method: 'POST',
    body: { ref: `refs/${ref}`, sha: base.sha },
  });
  log.info(`created ${config.CATBENCH_FORK}#${config.CATBENCH_BRANCH} at ${base.sha.slice(0, 7)}`);
  return true;
}

function assetPath(name) {
  return `${config.CATBENCH_ASSET_DIR}/${name}`;
}

/** Names directly under the asset dir of one ref, lowercased stem -> name. */
async function assetsOn(repo, ref) {
  const list = await gh(
    `/repos/${repo}/contents/${encodeURI(config.CATBENCH_ASSET_DIR)}?ref=${encodeURIComponent(ref)}`,
    { allow: [404] }
  );
  const out = new Map();
  for (const f of Array.isArray(list) ? list : []) {
    if (f.type !== 'file') continue;
    const dot = f.name.lastIndexOf('.');
    if (dot < 1) continue;
    out.set(normKey(f.name.slice(0, dot)), f.name);
  }
  return out;
}

/**
 * Write one file to the contribution branch, or report that it was already
 * there byte for byte. Re-uploading identical content would still make a
 * commit, which turns a no-op contribution into churn on an open PR.
 */
async function putFile(path, text, message) {
  const url = `/repos/${config.CATBENCH_FORK}/contents/${encodeURI(path)}`;
  const branch = config.CATBENCH_BRANCH;
  const content = Buffer.from(text, 'utf8').toString('base64');
  const have = await gh(`${url}?ref=${encodeURIComponent(branch)}`, { allow: [404] });
  if (have && have.content && have.content.replace(/\s/g, '') === content) return false;
  await gh(url, {
    method: 'PUT',
    body: { message, content, branch, ...(have ? { sha: have.sha } : {}) },
  });
  return true;
}

/**
 * A title in upstream's voice, listing what the branch adds. Their own history
 * reads `Fix visit counter tooltip text` -- imperative, sentence case, no
 * prefix -- so a batch says what it adds and stops.
 */
export function prTitle(stems) {
  const names = [...stems].sort((a, b) => a.localeCompare(b));
  if (!names.length) return 'Add CatBench results';
  if (names.length === 1) return `Add ${names[0]}`;
  // Long quant names blow past GitHub's title field fast; past three, count.
  const joined = names.length <= 3 && names.join(', ').length < 120;
  if (!joined) return `Add ${names[0]} and ${names.length - 1} more to CatBench`;
  return `Add ${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

/** The open PR from our branch, or null. */
async function openPr() {
  const owner = config.CATBENCH_FORK.split('/')[0];
  const prs = await gh(
    `/repos/${config.CATBENCH_UPSTREAM}/pulls` +
      `?state=open&head=${encodeURIComponent(`${owner}:${config.CATBENCH_BRANCH}`)}`
  );
  return Array.isArray(prs) && prs.length ? prs[0] : null;
}

// ── The one entry point ─────────────────────────────────────────────────────

/**
 * Stage one benched model onto the contribution branch and open or update the
 * pull request.
 *
 * `sources` is the SVG the model wrote and the script it wrote, exactly as
 * stored -- the same two files whether they came from a fresh pod or from the
 * mirror of an earlier run.
 *
 * Resolves `{ ok, skipped, why, stem, url, added }`. `skipped` is the ordinary
 * outcome for a model upstream already has; it is not an error.
 */
export async function contribute(modelId, { svgSource, pythonSource, upstreamRenderOk, name }) {
  if (!enabled()) {
    return { ok: false, why: 'contributing is not configured (GITHUB_TOKEN and CATBENCH_FORK)' };
  }
  if (!String(svgSource || '').trim() || !String(pythonSource || '').trim()) {
    return { ok: false, why: 'the run has no stored sources to contribute' };
  }
  // Their Action renders the .py itself, and a script that fails there lands in
  // the gallery as `⚠ render failed` with our name on it. The pod answers this
  // in its own sandbox, under their renderer's rules, before anything is staged.
  if (upstreamRenderOk !== true) {
    return {
      ok: false,
      why:
        upstreamRenderOk === false
          ? "their renderer gets no picture from this script"
          : 'no upstream render check on record for this run — re-run it with refresh',
    };
  }

  const stem = upstreamStem(modelId, name);
  const key = normKey(stem);
  const base = await upstreamBase();

  // Already theirs. The gallery is the point, so a model that is in it needs
  // nothing from us -- and /catbench serves it from upstream anyway.
  const theirs = await assetsOn(config.CATBENCH_UPSTREAM, base.branch);
  if (theirs.has(key)) {
    return { ok: true, skipped: true, why: 'already in the gallery', stem };
  }

  await ensureBranch(base);
  const message = `Add ${stem} to CatBench`;
  const wrote = [
    await putFile(assetPath(`${stem}.svg`), svgSource, message),
    await putFile(assetPath(`${stem}.py`), pythonSource, message),
  ];
  const added = wrote.some(Boolean);

  // Everything on our branch that upstream does not have yet: what the PR is
  // actually offering, recomputed rather than remembered, so a bot restart or
  // a second box still titles it correctly.
  // Stems, not normalized keys: the key is for matching, the stem is what a
  // human reads. Seeded with this run's, since a listing can lag a write.
  const offering = new Set([stem]);
  const ours = await assetsOn(config.CATBENCH_FORK, config.CATBENCH_BRANCH);
  for (const [k, name] of ours) {
    if (!theirs.has(k)) offering.add(name.replace(/\.(svg|py|png|jpe?g|gif)$/i, ''));
  }
  const title = prTitle(offering);

  let pr = await openPr();
  if (!pr) {
    // Body stays empty on purpose. The diff says what this is, and upstream's
    // own commits do not pad.
    pr = await gh(`/repos/${config.CATBENCH_UPSTREAM}/pulls`, {
      method: 'POST',
      body: {
        title,
        body: '',
        head: `${config.CATBENCH_FORK.split('/')[0]}:${config.CATBENCH_BRANCH}`,
        base: base.branch,
      },
    });
    log.info(`opened ${pr.html_url} for ${stem}`);
  } else if (pr.title !== title) {
    await gh(`/repos/${config.CATBENCH_UPSTREAM}/pulls/${pr.number}`, {
      method: 'PATCH',
      body: { title },
    });
    log.info(`updated ${pr.html_url}: ${title}`);
  }

  return { ok: true, added, stem, url: pr.html_url, pending: offering.size };
}

/** What the branch is offering right now, for a status line. */
export async function pending() {
  if (!enabled()) return null;
  const base = await upstreamBase();
  const theirs = await assetsOn(config.CATBENCH_UPSTREAM, base.branch);
  const ours = await assetsOn(config.CATBENCH_FORK, config.CATBENCH_BRANCH);
  const stems = new Set();
  for (const [k, name] of ours) {
    if (!theirs.has(k)) stems.add(name.replace(/\.(svg|py|png|jpe?g|gif)$/i, ''));
  }
  const pr = await openPr();
  return { stems: [...stems].sort(), url: pr?.html_url || '' };
}
