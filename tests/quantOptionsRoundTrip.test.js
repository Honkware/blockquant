import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';

const SRC = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'src', 'commands', 'quant.js');

// Every /quant option has to survive three hops: read off the interaction,
// written to the job record, read back in runApprovedJob. subfolder was read,
// shown on the approval card and passed to the controller, but never written --
// so the default in the destructure won and the pod downloaded whole repos it
// had been told to take one directory from. Nothing errored.
describe('/quant options round-trip through the job record', () => {
  const src = fs.readFileSync(SRC, 'utf8');
  const from = src.indexOf('await db.upsertJob({');
  const persisted = src.slice(from, src.indexOf('} finally {', from));
  const rj = src.indexOf('export async function runApprovedJob');
  const destructured = src.slice(rj, src.indexOf('const category =', rj));

  for (const field of ['codebook', 'visionBits', 'headBits', 'subfolder', 'testPrompt']) {
    it(`persists and reads back ${field}`, () => {
      expect(persisted).toContain(field);
      expect(destructured).toContain(field);
    });
  }
});
