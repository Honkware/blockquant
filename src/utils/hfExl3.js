/**
 * Hugging Face layout: one repo per (model × bpw) — `{modelName}-exl3-{bpw}bpw`.
 * This matches the RunPod backend convention: one upload target per variant,
 * no branch differentiation required.
 */

export function formatExl3Revision(bpw) {
  const n = Number(bpw);
  if (!Number.isFinite(n)) throw new TypeError(`Invalid bpw value: ${bpw} (expected finite number)`);
  return `${n.toFixed(2)}bpw`;
}

export function exl3RepoName(modelName, bpw) {
  const n = Number(bpw);
  if (!Number.isFinite(n)) throw new TypeError(`Invalid bpw value: ${bpw} (expected finite number)`);
  return `${modelName}-exl3-${n.toFixed(2)}bpw`;
}

export function exl3TreeUrl(repoUrl) {
  if (!repoUrl) return repoUrl;
  return String(repoUrl).replace(/\/$/, '');
}
