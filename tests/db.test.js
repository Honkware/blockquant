import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// The job store rewrites its whole file on every patch, so a file it cannot
// parse must survive the next write.
const h = await vi.hoisted(async () => {
  const { mkdtempSync } = await import('node:fs');
  const { join } = await import('node:path');
  const { tmpdir } = await import('node:os');
  return { root: mkdtempSync(join(tmpdir(), 'bq-db-')) };
});

vi.mock('../src/config.js', () => ({ default: { ROOT_DIR: h.root, LOG_LEVEL: 'error' } }));

const db = await import('../src/services/db.js');
const data = path.join(h.root, 'data');

beforeEach(() => {
  fs.rmSync(data, { recursive: true, force: true });
  fs.mkdirSync(data, { recursive: true });
});

describe('the job store', () => {
  it('keeps a corrupt file when the next patch writes', async () => {
    const torn = '{"job-1": {"id": "job-1", "status": "running"}, "job-2": {"id": "jo';
    fs.writeFileSync(path.join(data, 'jobs.json'), torn);
    await db.patchJob('job-3', { status: 'queued' });
    const kept = fs.readdirSync(data).filter((n) => n.startsWith('jobs.json.corrupt-'));
    expect(kept).toHaveLength(1);
    expect(fs.readFileSync(path.join(data, kept[0]), 'utf8')).toBe(torn);
  });

  it('refuses to write over a file it cannot read', async () => {
    const file = path.join(data, 'jobs.json');
    fs.writeFileSync(file, '{"job-1": {"id": "job-1"}}');
    fs.chmodSync(file, 0o000);
    await expect(db.patchJob('job-2', { status: 'queued' })).rejects.toThrow();
    fs.chmodSync(file, 0o600);
    expect(JSON.parse(fs.readFileSync(file, 'utf8'))).toEqual({ 'job-1': { id: 'job-1' } });
  });

  it('starts empty when there is no file yet', async () => {
    expect(await db.loadJobs()).toEqual({});
  });
});
