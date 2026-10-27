/**
 * The generated exllamav3 facts table, read from the same JSON the backend and
 * preflight read. Regenerate with backend/scripts/gen_arch_support.py; nothing
 * here is meant to be edited by hand.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const FILE = path.join(path.dirname(fileURLToPath(import.meta.url)),
                       '..', '..', 'backend', 'arch_support.json');

let cached = null;

function table() {
  if (cached) return cached;
  try {
    cached = JSON.parse(fs.readFileSync(FILE, 'utf8'));
  } catch {
    // The bot names quants off this, so an empty table is better than a crash
    // at command time: every default falls back to the unquantized answer and
    // the requester can still pin the numbers themselves.
    cached = {};
  }
  return cached;
}

// exllamav3's own --head_bits default, scraped from its convert_model.py.
export function defaultHeadBits() {
  return table().default_head_bits ?? 6;
}

// What --vision_bits `auto` gives this architecture. Absent means 16 -- copy the
// tower whole -- and that is a fact, not a gap: convert_model.py reads the cap
// off the vision model with a literal `.get("default_vision_bits", 16)`, with no
// base-class default behind it, so an arch that declares nothing gets 16.
export function defaultVisionBits(arch) {
  return Number((table().default_vision_bits || {})[arch] ?? 16);
}
