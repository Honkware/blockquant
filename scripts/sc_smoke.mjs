#!/usr/bin/env node
/**
 * Drive one self-calibrated quant through the bot's own service layer, without
 * Discord.
 *
 * This is everything /quant does after the slash command: donor resolution,
 * runViaCli, the detached controller, the pod. It exists because an SC job is
 * ~45 minutes and the only other way to exercise it is to type in a channel and
 * wait -- and because the first four times it was run it failed in a different
 * place each time.
 *
 * It is NOT a substitute for the real path. The command layer has its own bugs
 * (head bits went unpinned there and every SC job died at launch), so a green
 * run here still needs /quant sc:true before anything is trusted.
 *
 *   node scripts/sc_smoke.mjs Qwen/Qwen3.5-0.8B 3.0 [--gpus N] [--keep-pod]
 *
 * --gpus N is worth having on a big model: sc_trace loads the donor
 * tensor-parallel and the converter runs a worker thread per device, so the
 * two longest stages both scale. sc_measure is single-device whatever you do.
 *
 * --keep-pod leaves the machine up on failure, so a crash after the conversion
 * is rescuable with backend/scripts/rescue_upload.py instead of reaped.
 */
import { fileURLToPath } from 'url';
import path from 'path';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { runViaCli } = await import(path.join(ROOT, 'src/services/runpodCli.js'));
const hf = await import(path.join(ROOT, 'src/services/huggingface.js'));
const { defaultHeadBits } = await import(path.join(ROOT, 'src/utils/archSupport.js'));

const argv = process.argv.slice(2);
const positional = argv.filter((a, i) => a[0] !== '-' && argv[i - 1] !== '--gpus');
const [modelId = 'Qwen/Qwen3.5-0.8B', bpw = '3.0'] = positional;
const keepPod = process.argv.includes('--keep-pod');
const gi = process.argv.indexOf('--gpus');
const gpuCount = gi === -1 ? null : Number(process.argv[gi + 1]);

const flight = await hf.preflight(modelId);
if (!flight.modelExists) {
  console.error(`${modelId}: not found or not accessible`);
  process.exit(1);
}

// SC samples the model through one of its own quants at >= the target bitrate.
const donor = await hf.findDonorQuant(modelId.split('/').pop(), parseFloat(bpw));
if (!donor) {
  console.error(`no donor quant at >= ${bpw}bpw for ${modelId}; run a plain quant first`);
  process.exit(1);
}

// Head bits are pinned rather than left to the converter: the published name
// states them, and run_runpod_job refuses --sc without them.
const headBits = defaultHeadBits();

console.log(`model    : ${modelId} (${flight.sizeGb} GB, ${flight.architecture})`);
console.log(`vision   : ${flight.hasVision ? `auto -> ${flight.visionBitsAuto}` : 'none'}`);
console.log(`donor    : ${donor}`);
console.log(`gpus     : ${gpuCount ?? 1}`);
console.log(`keep-pod : ${keepPod}`);

const res = await runViaCli({
  jobId: 'sc-smoke',
  modelId,
  variants: [bpw],
  hfOrg: process.env.HF_ORG || 'Honkware',
  sc: true,
  donorRepo: donor,
  headBits,
  gpuCount,
  keepPod,
  onProgress: (d) => process.stdout.write(
    `[${String(d.overall).padStart(3)}%] ${String(d.stage).padEnd(12)} ${d.message ?? ''}\n`
  ),
});
console.log('RESULT', JSON.stringify(res, null, 2));
