#!/usr/bin/env node
/**
 * Time a plain conversion on N GPUs, to find out whether multi-GPU is worth
 * the multiplied price cap. Nothing has ever measured this.
 *
 *   node scripts/gpu_scale_probe.mjs <model> <bpw> <gpuCount>
 *
 * Read [quantize] <bpw> complete (Xm) out of the controller log and compare
 * against the same model on one card. Deliberately a plain quant: the
 * conversion is what parallelizes, and SC would bury it under an hour of
 * calibration.
 */
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { runViaCli } = await import(path.join(ROOT, "src/services/runpodCli.js"));

const [modelId = "Qwen/Qwen3.5-0.8B", bpw = "4.0", gpus = "1"] =
  process.argv.slice(2).filter((a) => a[0] !== "-");

console.log(`probe: ${modelId} @ ${bpw}bpw on ${gpus} GPU(s)`);
const t0 = Date.now();
const res = await runViaCli({
  jobId: `gpuprobe-${gpus}`,
  modelId,
  variants: [bpw],
  hfOrg: process.env.HF_ORG || "Honkware",
  gpuCount: Number(gpus),
  onProgress: (d) => {
    if (d.stage === "Quantizing") process.stdout.write(`\r[${d.overall}%] ${d.message ?? ""}          `);
  },
});
console.log(`\nwall ${(Date.now() - t0) / 60000} min`);
console.log("RESULT", JSON.stringify(res, null, 2));
