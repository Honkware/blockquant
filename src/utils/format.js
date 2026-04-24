/** Format milliseconds into "Xh Ym Zs" string. */
export function formatDuration(ms) {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;

  const parts = [];
  if (h) parts.push(`${h}h`);
  if (m) parts.push(`${m}m`);
  parts.push(`${sec}s`);
  return parts.join(' ');
}

/** Build a text progress bar. */
export function progressBar(pct, width = 20) {
  const clampedPct = Math.max(0, Math.min(100, pct));
  const safeWidth = Math.max(0, Math.floor(width));
  const filled = Math.round((clampedPct / 100) * safeWidth);
  const empty = safeWidth - filled;
  return `${'█'.repeat(filled)}${'░'.repeat(empty)} ${Math.round(clampedPct)}%`;
}

/** Truncate a string with ellipsis. */
export function truncate(str, max = 100) {
  return str.length > max ? str.slice(0, max - 1) + '…' : str;
}
