/**
 * Hugging Face layout: one repo per (model × bpw) — `{modelName}-exl3-{bpw}bpw`.
 * This matches the RunPod backend convention: one upload target per variant,
 * no branch differentiation required.
 */

// One decimal, because that is what actually gets published: quant.js
// normalizes the request to "4.0"/"4.5" and the pod uploads that string
// verbatim (remote/quant.py). This used to be toFixed(2), so the pre-flight
// asked HF about `-exl3-4.00bpw` while every real repo was `-exl3-4.0bpw` --
// the lookup missed every time and "already quantized" could never fire.
export function exl3RepoName(modelName, bpw) {
  const n = Number(bpw);
  if (!Number.isFinite(n)) throw new TypeError(`Invalid bpw value: ${bpw} (expected finite number)`);
  return `${modelName}-exl3-${Number.isInteger(n) ? n.toFixed(1) : String(n)}bpw`;
}

export function exl3TreeUrl(repoUrl) {
  if (!repoUrl) return repoUrl;
  return String(repoUrl).replace(/\/$/, '');
}
