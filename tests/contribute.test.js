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

const { contribute, upstreamStem, prTitle, enabled } = await import(
  '../src/services/contribute.js'
);

/**
 * A GitHub stub that answers by route. `assets` is what each ref holds, so a
 * test says what the world looks like rather than scripting a call order --
 * the code is free to reorder its reads without rewriting every test.
 */
function github({ theirs = [], ours = null, pr = null } = {}) {
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
    if (path.startsWith('/repos/Kate/site/pulls?')) return json(pr ? [pr] : []);
    if (path === '/repos/Kate/site/pulls' && method === 'POST') {
      return json({ number: 1, html_url: 'https://gh/pr/1', title: opts.body && JSON.parse(opts.body).title });
    }
    if (/\/repos\/Kate\/site\/pulls\/\d+$/.test(path) && method === 'PATCH') return json({});
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

describe('the pull request title', () => {
  it('names one model plainly', () => {
    expect(prTitle(['GLM-5.2'])).toBe('Add GLM-5.2');
  });

  it('joins a short batch', () => {
    expect(prTitle(['B', 'A'])).toBe('Add A and B');
  });

  it('counts instead of listing once the names get long', () => {
    const long = ['a', 'b', 'c', 'd'].map((c) => c.repeat(40));
    expect(prTitle(long)).toMatch(/and 3 more to CatBench$/);
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

  it('opens a pull request with an empty body', async () => {
    const { handler, calls } = github();
    globalThis.fetch = handler;
    const res = await contribute('org/Kit', GOOD);
    const open = calls.find((c) => c.path === '/repos/Kate/site/pulls' && c.method === 'POST');
    expect(open.body).toMatchObject({ title: 'Add Kit', body: '', base: 'main', head: 'Us:catbench' });
    expect(res.url).toBe('https://gh/pr/1');
  });

  it('adds to the open pull request instead of opening a second one', async () => {
    const { handler, calls } = github({
      ours: ['Old.svg', 'Old.py'],
      pr: { number: 7, html_url: 'https://gh/pr/7', title: 'Add Old' },
    });
    globalThis.fetch = handler;
    const res = await contribute('org/New', GOOD);
    expect(calls.some((c) => c.path === '/repos/Kate/site/pulls' && c.method === 'POST')).toBe(false);
    const retitle = calls.find((c) => c.method === 'PATCH');
    expect(retitle.body.title).toBe('Add New and Old');
    expect(res).toMatchObject({ url: 'https://gh/pr/7', pending: 2 });
  });

  it('leaves the title alone when the batch has not changed', async () => {
    const { handler, calls } = github({
      ours: ['Kit.svg', 'Kit.py'],
      pr: { number: 7, html_url: 'https://gh/pr/7', title: 'Add Kit' },
    });
    globalThis.fetch = handler;
    await contribute('org/Kit', GOOD);
    expect(calls.some((c) => c.method === 'PATCH')).toBe(false);
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
  });
});
