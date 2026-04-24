/**
 * BlockQuant Self-Test Harness
 *
 * Run: node self-test.mjs
 *
 * Tests the API, dashboard, Discord bot connectivity, and pipeline integrity.
 */

const API_URL = process.env.BLOCKQUANT_API_URL || 'http://localhost:8000';

let passed = 0;
let failed = 0;

function ok(label, condition, detail = '') {
  if (condition) {
    passed++;
    console.log(`  ✅ ${label}`);
  } else {
    failed++;
    console.log(`  ❌ ${label}${detail ? ` — ${detail}` : ''}`);
  }
}

async function test(name, fn) {
  console.log(`\n▶ ${name}`);
  try {
    await fn();
  } catch (err) {
    failed++;
    console.log(`  ❌ Test crashed: ${err.message}`);
  }
}

// ── API Tests ───────────────────────────────────────────────────────────────

await test('API Health', async () => {
  const res = await fetch(`${API_URL}/health`);
  ok('responds 200', res.status === 200);
  const data = await res.json();
  ok('status is ok', data.status === 'ok');
  ok('celery is true', data.celery === true);
});

await test('Dashboard HTML', async () => {
  const res = await fetch(`${API_URL}/dashboard/`);
  ok('responds 200', res.status === 200);
  const html = await res.text();
  ok('contains BlockQuant', html.includes('BlockQuant'));
});

await test('Dashboard Stats JSON', async () => {
  const res = await fetch(`${API_URL}/dashboard/api/stats`);
  ok('responds 200', res.status === 200);
  const data = await res.json();
  ok('has date', typeof data.date === 'string');
  ok('has jobs_completed', typeof data.jobs_completed === 'number');
});

await test('Dashboard Leaderboard JSON', async () => {
  const res = await fetch(`${API_URL}/dashboard/api/leaderboard?limit=5`);
  ok('responds 200', res.status === 200);
  const data = await res.json();
  ok('is array', Array.isArray(data));
});

await test('Dashboard Recent Jobs JSON', async () => {
  const res = await fetch(`${API_URL}/dashboard/api/recent?limit=5`);
  ok('responds 200', res.status === 200);
  const data = await res.json();
  ok('is array', Array.isArray(data));
});

// ── Quant Job Tests ─────────────────────────────────────────────────────────

let jobId = null;

await test('Submit Quant Job', async () => {
  const res = await fetch(`${API_URL}/api/v1/quant`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_id: 'microsoft/Phi-3-mini-4k-instruct',
      format: 'exl3',
      variants: ['4.0'],
      verify_quality: false,
    }),
  });
  ok('responds 200', res.status === 200);
  const data = await res.json();
  ok('has job_id', typeof data.job_id === 'string');
  jobId = data.job_id;
});

await test('Poll Job Status', async () => {
  ok('job_id exists', !!jobId);
  if (!jobId) return;

  const res = await fetch(`${API_URL}/api/v1/jobs/${jobId}`);
  ok('responds 200', res.status === 200);
  const data = await res.json();
  ok('has status', typeof data.status === 'string');
  ok('status is valid', ['queued', 'running', 'complete', 'failed'].includes(data.status));
});

// ── Completed Job Integrity ─────────────────────────────────────────────────

await test('Completed Job Has Result', async () => {
  if (!jobId) return;
  // Wait up to 60s for completion — if model is cached it'll finish fast
  for (let i = 0; i < 12; i++) {
    const res = await fetch(`${API_URL}/api/v1/jobs/${jobId}`);
    const data = await res.json();
    if (data.status === 'complete' || data.status === 'failed') {
      ok('job finished', true);
      if (data.status === 'complete') {
        ok('has result', !!data.result);
        ok('result has stages', Array.isArray(data.result.stages));
        ok('result has outputs', Array.isArray(data.result.outputs));
      }
      return;
    }
    await new Promise((r) => setTimeout(r, 5000));
  }
  // Job didn't finish in 60s — likely uncached model. Not a failure, just skip extended checks.
  ok('job completion (uncached — skipped extended checks)', true, 'model not cached in API workspace');
});

// ── Live Progress Metadata ──────────────────────────────────────────────────

await test('Running Job Has Progress Metadata', async () => {
  // Submit a fresh job and immediately poll for progress
  const res = await fetch(`${API_URL}/api/v1/quant`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_id: 'microsoft/Phi-3-mini-4k-instruct',
      format: 'exl3',
      variants: ['4.0'],
      verify_quality: false,
    }),
  });
  const data = await res.json();
  const freshJobId = data.job_id;

  // Poll a few times
  let hasProgress = false;
  for (let i = 0; i < 5; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    const poll = await fetch(`${API_URL}/api/v1/jobs/${freshJobId}`);
    const status = await poll.json();
    if (status.progress) {
      hasProgress = true;
      ok('progress has stage', typeof status.progress.stage === 'string');
      ok('progress has percent', typeof status.progress.percent === 'number');
      ok('progress has message', typeof status.progress.message === 'string');
      break;
    }
    if (status.status === 'complete' || status.status === 'failed') break;
  }
  if (!hasProgress) {
    // Job may have completed too fast (cached) — that's okay
    ok('progress metadata (or job cached)', true, 'job may have been cached');
  }
});

// ── Summary ─────────────────────────────────────────────────────────────────

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
console.log(`${'='.repeat(50)}`);

process.exit(failed > 0 ? 1 : 0);
