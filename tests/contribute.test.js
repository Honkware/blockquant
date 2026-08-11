import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// config.js hard-exits on missing required env vars.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';
process.env.GITHUB_TOKEN = 'gh_test';
process.env.CATBENCH_UPSTREAM = 'Kate/site';
process.env.CATBENCH_FORK = 'Us/site';
process.env.CATBENCH_BRANCH = 'catbench';
process.env.CATBENCH_ASSET_DIR = 'demos/CatBench/assets';

const { contribute, upstreamStem, compareUrl, enabled, whyDisabled } = await import(
  '../src/services/contribute.js'
);

/**
 * A GitHub stub that answers by route. `assets` is what each ref holds, so a
 * test says what the world looks like rather than scripting a call order --
 * the code is free to reorder its reads without rewriting every test.
 */
function github({ theirs = [], ours = null } = {}) {
  const calls = [];
  const files = (names) =>
    names.map((n) => ({ type: 'file', name: n, sha: `sha-${n}` }));

  const handler = vi.fn(async (url, opts = {}) => {
    const method = opts.method || 'GET';
    const path = String(url).replace('https://api.github.com', '');
    calls.push({ method, path, body: opts.body ? JSON.parse(opts.body) : null });
    const json = (body, status = 200) => ({
      ok: status < 400,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body),
    });

    if (path === '/repos/Kate/site') return json({ default_branch: 'main' });
    if (path === '/repos/Kate/site/git/ref/heads/main') return json({ object: { sha: 'base1' } });
    if (path.startsWith('/repos/Kate/site/contents/') && path.includes('ref=main')) {
      return json(files(theirs));
    }
    if (path === '/repos/Us/site/git/ref/heads/catbench') {
      return ours ? json({ object: { sha: 'b1' } }) : json({ message: 'Not Found' }, 404);
    }
    if (path === '/repos/Us/site/git/refs' && method === 'POST') return json({});
    if (path.startsWith('/repos/Us/site/contents/') && path.includes('ref=catbench')) {
      // A directory listing, or a single file probe.
      if (path.includes('assets?')) return json(files(ours || []));
      const name = decodeURI(path.split('/').pop().split('?')[0]);
      return (ours || []).includes(name)
        ? json({ sha: `sha-${name}`, content: 'AAAA' })
        : json({ message: 'Not Found' }, 404);
    }
    if (path.startsWith('/repos/Us/site/contents/') && method === 'PUT') return json({});
    // Deliberately still answered, as a trap. If the code ever goes back to
    // opening pull requests these routes succeed quietly rather than throwing
    // "unstubbed", so the assertion below is what catches it -- and it names
    // the actual problem instead of a missing stub.
    if (path.includes('/pulls')) return json([]);
    throw new Error(`unstubbed ${method} ${path}`);
  });
  return { handler, calls };
}

const GOOD = { svgSource: '<svg/>', pythonSource: 'print(1)', upstreamRenderOk: true };

let restore;
beforeEach(() => {
  restore = globalThis.fetch;
});
afterEach(() => {
  globalThis.fetch = restore;
  vi.restoreAllMocks();
});

describe('the name upstream carries', () => {
  it('drops the owner, because upstream keys a model by its filename', () => {
    expect(upstreamStem('Honkware/Qwopus3.6-27B-Fusion-exl3-4.5bpw')).toBe(
      'Qwopus3.6-27B-Fusion-exl3-4.5bpw'
    );
  });

  it('drops BF16 and keeps the quant level', () => {
    expect(upstreamStem('Honkware/Qwopus3.6-27B-Fusion-BF16-exl3-4.5bpw')).toBe(
      'Qwopus3.6-27B-Fusion-exl3-4.5bpw'
    );
  });

  it('leaves a name that only looks like it has BF16 in it alone', () => {
    expect(upstreamStem('org/Model-BF16Bit-exl3')).toBe('Model-BF16Bit-exl3');
    expect(upstreamStem('org/NotBF16-exl3')).toBe('NotBF16-exl3');
  });

  it('keeps the case, which is what the gallery displays', () => {
    expect(upstreamStem('a/GLM-5.2-Air')).toBe('GLM-5.2-Air');
  });
});

describe('the compare link', () => {
  it('points a human at the pull request they open themselves', () => {
    // The bot deliberately stops at the push, so this URL is the handoff.
    expect(compareUrl('main')).toBe(
      'https://github.com/Kate/site/compare/main...Us:catbench?expand=1'
    );
  });

  it('uses upstream\'s real default branch, not an assumed main', () => {
    expect(compareUrl('gh-pages')).toContain('/compare/gh-pages...Us:catbench');
  });
});

describe('contributing a run', () => {
  it('refuses a run their renderer could not draw', async () => {
    globalThis.fetch = vi.fn();
    const res = await contribute('a/B', { ...GOOD, upstreamRenderOk: false });
    expect(res.ok).toBe(false);
    expect(res.why).toMatch(/renderer/);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it('refuses a run from before the check existed rather than guessing', async () => {
    globalThis.fetch = vi.fn();
    const res = await contribute('a/B', { ...GOOD, upstreamRenderOk: undefined });
    expect(res.ok).toBe(false);
    expect(res.why).toMatch(/refresh/);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it('refuses a run with no sources stored', async () => {
    globalThis.fetch = vi.fn();
    const res = await contribute('a/B', { ...GOOD, pythonSource: '  ' });
    expect((await contribute('a/B', { ...GOOD, svgSource: '' })).ok).toBe(false);
    expect(res.ok).toBe(false);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it('skips a model the gallery already has, without touching the fork', async () => {
    const { handler, calls } = github({ theirs: ['GLM-5.2.svg', 'GLM-5.2.py'] });
    globalThis.fetch = handler;
    const res = await contribute('org/GLM-5.2', GOOD);
    expect(res).toMatchObject({ ok: true, skipped: true });
    expect(calls.some((c) => c.path.includes('/Us/site'))).toBe(false);
  });

  it('matches upstream loosely enough to catch a case difference', async () => {
    const { handler } = github({ theirs: ['glm_5.2.py'] });
    globalThis.fetch = handler;
    expect(await contribute('org/GLM-5.2', GOOD)).toMatchObject({ skipped: true });
  });

  it('sends sources only, never a raster or a manifest', async () => {
    const { handler, calls } = github();
    globalThis.fetch = handler;
    const res = await contribute('Honkware/Kit-BF16-exl3-4.5bpw', GOOD);
    expect(res.ok).toBe(true);
    const puts = calls.filter((c) => c.method === 'PUT').map((c) => decodeURI(c.path));
    expect(puts).toHaveLength(2);
    expect(puts.some((p) => p.endsWith('/demos/CatBench/assets/Kit-exl3-4.5bpw.svg'))).toBe(true);
    expect(puts.some((p) => p.endsWith('/demos/CatBench/assets/Kit-exl3-4.5bpw.py'))).toBe(true);
    expect(calls.some((c) => /manifest\.json/.test(c.path))).toBe(false);
    expect(calls.some((c) => /\.(jpg|png)/.test(c.path))).toBe(false);
  });

  it('opens the branch off upstream head when it does not exist yet', async () => {
    const { handler, calls } = github();
    globalThis.fetch = handler;
    await contribute('org/Kit', GOOD);
    const mk = calls.find((c) => c.path === '/repos/Us/site/git/refs');
    expect(mk.body).toEqual({ ref: 'refs/heads/catbench', sha: 'base1' });
  });

  it('never touches upstream beyond reading it', async () => {
    const { handler, calls } = github();
    globalThis.fetch = handler;
    await contribute('org/Kit', GOOD);
    const writes = calls.filter((c) => c.method !== 'GET');
    expect(writes.length).toBeGreaterThan(0);
    // Every write lands on our own fork. That is what keeps the token small.
    for (const c of writes) expect(c.path).toContain('/repos/Us/site');
    expect(calls.some((c) => c.path.includes('/pulls'))).toBe(false);
  });

  it('hands back the compare link so a person can open the PR', async () => {
    const { handler } = github();
    globalThis.fetch = handler;
    const res = await contribute('org/Kit', GOOD);
    expect(res.url).toBe('https://github.com/Kate/site/compare/main...Us:catbench?expand=1');
  });

  it('adds to the same branch rather than starting a new one', async () => {
    const { handler, calls } = github({ ours: ['Old.svg', 'Old.py'] });
    globalThis.fetch = handler;
    const res = await contribute('org/New', GOOD);
    expect(calls.some((c) => c.path === '/repos/Us/site/git/refs')).toBe(false);
    expect(res.pending).toBe(2);
  });

  it('reports a GitHub failure instead of claiming a contribution', async () => {
    globalThis.fetch = vi.fn(async () => ({
      ok: false,
      status: 403,
      text: async () => JSON.stringify({ message: 'Resource not accessible by personal access token' }),
    }));
    await expect(contribute('org/Kit', GOOD)).rejects.toThrow(/403.*not accessible/);
  });
});

describe('configuration', () => {
  it('is on when a token and a fork are both set', () => {
    expect(enabled()).toBe(true);
    expect(whyDisabled()).toBe('');
  });

  it('names only what is missing', async () => {
    // Listing every requirement when one is unset reads as though none are,
    // and sends whoever hit it looking in the wrong place.
    const real = process.env.GITHUB_TOKEN;
    process.env.GITHUB_TOKEN = '';
    vi.resetModules();
    const fresh = await import('../src/services/contribute.js?nocache=1');
    expect(fresh.whyDisabled()).toBe('GITHUB_TOKEN is not set');
    expect(fresh.whyDisabled()).not.toContain('CATBENCH_FORK');
    process.env.GITHUB_TOKEN = real;
  });
});
