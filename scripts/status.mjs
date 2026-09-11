#!/usr/bin/env node
// What is in flight right now. Exits 1 if anything is, so a deploy can gate on
// it:  npm run --silent status && pm2 restart blockquant
//
// A restart no longer kills a controller (services/detached.js), but it still
// kills the Discord side of it: the progress embed freezes, and for /quant the
// cards and the collection never get written. Worth knowing before you deploy.
import { list } from '../src/services/detached.js';

const runs = list();
const live = runs.filter((r) => r.running);
const done = runs.filter((r) => !r.running);

const age = (ms) => {
  const m = Math.floor((Date.now() - ms) / 60000);
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, '0')}`;
};

for (const r of done) {
  const st = r.status?.signal || (r.status ? `exit ${r.status.code}` : 'no status');
  console.log(`done     ${r.kind} ${r.meta?.modelId || r.id} (${st}) ${r.logPath}`);
}
for (const r of live) {
  const extra = r.meta?.variants ? ` [${r.meta.variants.join(', ')}]` : '';
  console.log(`RUNNING  ${r.kind} ${r.meta?.modelId || r.id}${extra} pid ${r.pid}, ${age(r.startedAt)} in`);
  console.log(`         ${r.logPath}`);
}

if (!live.length) {
  console.log('nothing in flight; safe to restart');
  process.exit(0);
}
console.log(`\n${live.length} controller(s) in flight. A restart leaves them running but freezes their embeds.`);
process.exit(1);
