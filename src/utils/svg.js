import { Resvg } from '@resvg/resvg-js';

const SVG_RE = /<svg[\s\S]*?<\/svg>/i;

/** Pull the first complete <svg>...</svg> out of a model reply, or null. */
export function extractSvg(text) {
  if (!text) return null;
  const m = SVG_RE.exec(text);
  return m ? m[0] : null;
}

/**
 * Strip the active parts of an SVG: <script>, event handlers, <foreignObject>
 * (arbitrary HTML), and any href that isn't an in-document fragment or an
 * inline data: image. Resvg ignores all of this already — it has no script
 * engine and no network — but a model reply is untrusted input and the SVG may
 * also be handed to something less careful later, so strip it at the door.
 */
export function sanitizeSvg(svg) {
  return String(svg)
    .replace(/<script[\s\S]*?<\/script\s*>/gi, '')
    .replace(/<foreignObject[\s\S]*?<\/foreignObject\s*>/gi, '')
    .replace(/\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '')
    .replace(
      /\s(?:xlink:)?href\s*=\s*("([^"]*)"|'([^']*)')/gi,
      (full, _q, dq, sq) => {
        const v = (dq ?? sq ?? '').trim();
        return /^(#|data:image\/)/i.test(v) ? full : '';
      }
    );
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
