#!/usr/bin/env node
/**
 * Store a CatBench run from a controller result file.
 *
 * A controller outlives the bot -- it is reparented to init so a restart
 * cannot orphan its pod -- which means a run can finish with nobody listening.
 * The kittens are in the result JSON either way; what is missing is the half
 * of /catbench that happens after runCatbench resolves. This is that half:
 * rasterize the SVG, grade both halves, and put the run in the store.
 *
 * It deliberately does not post to Discord. Once the run is stored, asking for
 * the same model serves it from cache instantly, with the same embeds a live
 * run would have produced -- so the delivery path stays in one place.
 *
 *   node scripts/catbench-recover.mjs backend/logs/catbench-<model>-<ts>.json
 *   node scripts/catbench-recover.mjs --latest
 *   node scripts/catbench-recover.mjs --latest --dry-run
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import config from '../src/config.js';
import * as gallery from '../src/services/catbench.js';
import { sanitizeSvg, renderSvgToPng } from '../src/utils/svg.js';

const LOG_DIR = path.join(config.ROOT_DIR, 'backend', 'logs');

function newestResult() {
  const files = readdirSync(LOG_DIR)
    .filter((n) => /^catbench-.+-\d+\.json$/.test(n))
    .map((n) => path.join(LOG_DIR, n))
    .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
  if (!files.length) throw new Error(`no catbench result files in ${LOG_DIR}`);
  return files[0];
}

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const file = args.includes('--latest')
  ? newestResult()
  : args.find((a) => !a.startsWith('--'));
if (!file) {
  console.error('usage: catbench-recover.mjs <result.json> | --latest [--dry-run]');
  process.exit(2);
}

const result = JSON.parse(readFileSync(file, 'utf8'));
const modelId = result.model_id;
const revision = result.revision || '';
const name = gallery.benchName(modelId, revision);
console.log(`${path.basename(file)}\n  model    ${modelId}${revision ? `@${revision}` : ''}`);
console.log(`  status   ${result.status}${result.error ? ` (${result.error})` : ''}`);

// Exactly what the command does with a live result: resvg on sanitized markup,
// and the matplotlib PNG the pod already rendered in its sandbox.
let svgPng = null;
if (result.svg) {
  try {
    svgPng = renderSvgToPng(sanitizeSvg(result.svg), 640);
  } catch (e) {
    console.log(`  svg      would not rasterize: ${e.message}`);
  }
}
const pythonPng = result.python_png_b64
  ? Buffer.from(result.python_png_b64, 'base64')
  : null;

const grade = gallery.gradeRun(result, { svgPng, pythonPng });
console.log(`  svg      ${svgPng ? `${svgPng.length} B` : 'none'}`);
console.log(`  python   ${pythonPng ? `${pythonPng.length} B` : 'none'}`);
console.log(`  upstream ${result.upstream_render_ok === undefined
  ? 'not checked' : result.upstream_render_ok ? 'renders' : 'would fail'}`);
console.log(`  storable ${grade.ok ? 'yes' : `no -- ${grade.why}`}`);

if (!grade.ok) process.exit(1);
if (dryRun) {
  console.log('\ndry run, nothing written');
  process.exit(0);
}

const rec = await gallery.saveRun(modelId, {
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
});
console.log(`\nstored as \`${rec.key}\`${rec.uploaded ? ' and pushed' : ' locally only'}`);
console.log(`ask for it with:  /catbench model:${modelId}` +
  `${revision ? ` revision:${revision}` : ''}`);
