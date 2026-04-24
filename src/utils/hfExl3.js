/**
 * Hugging Face layout: one model repo `{modelName}-exl3`, one git branch per BPW (`X.XXbpw`).
 */

export function formatExl3Revision(bpw) {
  const n = Number(bpw);
  if (!Number.isFinite(n)) throw new TypeError(`Invalid bpw: ${bpw}`);
  return `${n.toFixed(2)}bpw`;
}

export function exl3RepoName(modelName) {
  return `${modelName}-exl3`;
}

export function exl3TreeUrl(repoUrl, revision) {
  if (!revision || !repoUrl) return repoUrl;
  const base = String(repoUrl).replace(/\/$/, '');
  return `${base}/tree/${revision}`;
}
