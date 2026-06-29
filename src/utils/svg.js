import { Resvg } from '@resvg/resvg-js';

const SVG_RE = /<svg[\s\S]*?<\/svg>/i;

/** Pull the first complete <svg>...</svg> out of a model reply, or null. */
export function extractSvg(text) {
  if (!text) return null;
  const m = SVG_RE.exec(text);
  return m ? m[0] : null;
}

/**
 * Render an SVG string to a PNG buffer. Resvg is a static renderer — it does
 * not execute scripts or fetch remote resources — so it's safe on untrusted
 * model output. Throws on malformed SVG; callers should catch and skip.
 */
export function renderSvgToPng(svg, width = 512) {
  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: width },
    background: 'white',
  });
  return Buffer.from(resvg.render().asPng());
}
