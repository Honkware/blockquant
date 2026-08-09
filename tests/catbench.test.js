import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// catbench.js pulls in config.js, which hard-exits on missing required env vars.
process.env.BOT_TOKEN ??= 'test';
process.env.CLIENT_ID ??= 'test';
process.env.GUILD_ID ??= 'test';
process.env.HF_TOKEN ??= 'test';

const { normKey, modelKey, lookup, listAll } = await import('../src/services/catbench.js');
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
  },
};

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
