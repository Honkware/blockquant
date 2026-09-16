/**
 * Hugging Face layout: one repo per (model × bpw) — `{modelName}-exl3-{bpw}bpw`.
 * This matches the RunPod backend convention: one upload target per variant,
 * no branch differentiation required.
 */

// Keep in step with exl3_repo_slug in backend/src/blockquant/cards.py -- the
// pod builds the upload name there and the bot looks it up here, so a name
// only one of them knows about is a lookup that always misses. That already
// happened once: this was toFixed(2) while every published repo is one
// decimal, so "already quantized" could never fire.
//
// headBits is required for an SC build, because under SC the recipe picks
// head bits per budget and the name is incomplete without it. visionBits is
// set only when the tower was quantized; absent means it was copied at fp16.
// 16 is how a request spells "copy the tower whole", and a copied tower is not
// named -- so it is not a -V16, it is nothing. Mirrors quantized_vision_bits in
// backend/src/blockquant/cards.py.
export function quantizedVisionBits(visionBits) {
  if (visionBits == null) return null;
  const n = Number(visionBits);
  return n >= 1 && n <= 8 ? n : null;
}

export function exl3RepoName(modelName, bpw, { sc = false, headBits = null, visionBits = null } = {}) {
  const n = Number(bpw);
  if (!Number.isFinite(n)) throw new TypeError(`Invalid bpw value: ${bpw} (expected finite number)`);
  if (sc && headBits == null) throw new TypeError('A self-calibrated quant must name its head bits');
  const variant = Number.isInteger(n) ? n.toFixed(1) : String(n);
  const core = sc ? `SC-${variant}bpw-H${headBits}` : `${variant}bpw`;
  const vb = quantizedVisionBits(visionBits);
  return `${modelName}-exl3-${core}${vb ? `-V${vb}` : ''}`;
}

export function exl3TreeUrl(repoUrl) {
  if (!repoUrl) return repoUrl;
  return String(repoUrl).replace(/\/$/, '');
}
