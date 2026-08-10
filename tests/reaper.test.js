import { describe, it, expect, vi } from 'vitest';
import { parseRunTag, controllerAlive, reapOrphans } from '../src/services/reaper.js';

// A pod name carries the controller pid and the second it was created, so these
// tests can drive the exact ambiguity the reaper has to resolve: a pid that
// exists but is not the process that launched the pod.
const TS = 1786310305;
const PID = 4119069;
const NAME = `bq-${PID}-${TS % 100000}-${TS}`;

/** /proc for one pid, with a controlled cmdline and start time. */
function fakeProc({ pid = PID, cmdline = `python run_catbench_job.py --model x`, startedAt = TS - 30 } = {}) {
  const btime = 1000000;
  return (path) => {
    if (path === '/proc/stat') return `cpu 1 2 3\nbtime ${btime}\n`;
    if (path === `/proc/${pid}/cmdline`) return cmdline.replace(/ /g, '\0');
    if (path === `/proc/${pid}/stat`) {
      const ticks = (startedAt - btime) * 100;
      // pid (comm) state, then 19 fields before starttime at index 19.
      return `${pid} (python3) S ${Array(18).fill(0).join(' ')} ${ticks} rest`;
    }
    throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
  };
}

const alive = () => undefined;
const gone = () => { throw Object.assign(new Error('ESRCH'), { code: 'ESRCH' }); };
const notOurs = () => { throw Object.assign(new Error('EPERM'), { code: 'EPERM' }); };

describe('parseRunTag', () => {
  it('reads the controller pid and creation time out of the name', () => {
    expect(parseRunTag(NAME)).toEqual({ pid: PID, ts: TS });
  });

  it('accepts a real pod whose short field does not equal ts % 100000', () => {
    // run_tag is stamped when the controller starts; the pod's timestamp is
    // appended when it is actually rented, which is later. A real name seen in
    // production: bq-4130775-20914-1786320953, where 1786320953 % 100000 is
    // 20953, not 20914. Requiring them to match reaped nothing, ever.
    expect(parseRunTag('bq-4130775-20914-1786320953')).toEqual({ pid: 4130775, ts: 1786320953 });
  });

  it('rejects a name with a nonsense timestamp', () => {
    expect(parseRunTag(`bq-${PID}-20914-42`)).toBeNull();
  });

  it('ignores pods that are not ours', () => {
    for (const n of ['exliberate7', 'btc-node-1', 'klverify', 'bq-phase3', '', null]) {
      expect(parseRunTag(n)).toBeNull();
    }
  });
});

describe('controllerAlive', () => {
  it('keeps a pod whose controller is running and predates it', () => {
    expect(controllerAlive(PID, TS, { kill: alive, readFile: fakeProc() })).toBe(true);
  });

  it('reaps when the pid is gone', () => {
    expect(controllerAlive(PID, TS, { kill: gone, readFile: fakeProc() })).toBe(false);
  });

  it('treats EPERM as alive, not dead', () => {
    // A live process owned by another user. Reading this as dead would kill a
    // job that is running perfectly well.
    expect(controllerAlive(PID, TS, { kill: notOurs, readFile: fakeProc() })).toBe(true);
  });

  it('reaps a recycled pid running something that is not a controller', () => {
    const proc = fakeProc({ cmdline: '/usr/sbin/sshd -D' });
    expect(controllerAlive(PID, TS, { kill: alive, readFile: proc })).toBe(false);
  });

  it('reaps a controller that started after the pod, which cannot have launched it', () => {
    // The pid-reuse case that matters: right name, right script, wrong process.
    const proc = fakeProc({ startedAt: TS + 3600 });
    expect(controllerAlive(PID, TS, { kill: alive, readFile: proc })).toBe(false);
  });

  it('allows clock skew rather than reaping on a few seconds', () => {
    const proc = fakeProc({ startedAt: TS + 30 });
    expect(controllerAlive(PID, TS, { kill: alive, readFile: proc })).toBe(true);
  });

  it('keeps the pod when /proc cannot be read at all', () => {
    const blind = () => { throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' }); };
    expect(controllerAlive(PID, TS, { kill: alive, readFile: blind })).toBe(true);
  });
});

describe('reapOrphans', () => {
  const pod = { id: 'abc123', name: NAME };
  const nowAfter = (mins) => () => (TS + mins * 60) * 1000;

  it('terminates a pod whose controller is gone and which is old enough', async () => {
    const terminatePod = vi.fn().mockResolvedValue(true);
    vi.spyOn(process, 'kill').mockImplementation(gone);
    const out = await reapOrphans({
      listPods: async () => [pod], terminatePod, now: nowAfter(60),
    });
    expect(out).toEqual(['abc123']);
    expect(terminatePod).toHaveBeenCalledWith('abc123');
    vi.restoreAllMocks();
  });

  it('leaves a young pod alone even with a dead pid', async () => {
    // Provisioning and bootstrap take minutes; a retried controller inside that
    // window is a job starting, not a leak.
    const terminatePod = vi.fn();
    vi.spyOn(process, 'kill').mockImplementation(gone);
    const out = await reapOrphans({
      listPods: async () => [pod], terminatePod, now: nowAfter(2),
    });
    expect(out).toEqual([]);
    expect(terminatePod).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });

  it('never touches pods it cannot parse', async () => {
    const terminatePod = vi.fn();
    vi.spyOn(process, 'kill').mockImplementation(gone);
    const out = await reapOrphans({
      listPods: async () => [{ id: 'x', name: 'exliberate7' }, { id: 'y', name: 'btc-node-1' }],
      terminatePod,
      now: nowAfter(600),
    });
    expect(out).toEqual([]);
    expect(terminatePod).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });

  it('reports without terminating in dry run', async () => {
    const terminatePod = vi.fn();
    vi.spyOn(process, 'kill').mockImplementation(gone);
    const out = await reapOrphans({
      listPods: async () => [pod], terminatePod, now: nowAfter(60), dryRun: true,
    });
    expect(out).toEqual([]);
    expect(terminatePod).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });

  it('says nothing and reaps nothing when the pod list is unavailable', async () => {
    const terminatePod = vi.fn();
    const out = await reapOrphans({
      listPods: async () => { throw new Error('runpod down'); }, terminatePod,
    });
    expect(out).toEqual([]);
    expect(terminatePod).not.toHaveBeenCalled();
  });
});
