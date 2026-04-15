/**
 * Hugging Face layout: one model repo per BPW, e.g. `{modelName}-8bpw-exl3`.
 */

function formatBpwLabel(bpw) {
  const n = Number(bpw);
  if (!Number.isFinite(n)) throw new TypeError(`Invalid bpw: ${bpw}`);
  return Number.isInteger(n) ? String(n) : String(n).replace(/\.0+$/, '');
}

export function exl3RepoName(modelName, bpw) {
  if (bpw == null) return `${modelName}-exl3`;
  return `${modelName}-${formatBpwLabel(bpw)}bpw-exl3`;
}

export function exl3TreeUrl(repoUrl) {
  return repoUrl || null;
}
