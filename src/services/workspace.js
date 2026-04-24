import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'fs/promises';
import path from 'path';
import config from '../config.js';
import { getLogger } from '../logger.js';
import { AppError } from '../errors/taxonomy.js';

const log = getLogger('workspace');

const DIRS = {
  root: config.WORKSPACE_DIR,
  model: path.join(config.WORKSPACE_DIR, 'model'),
  work: path.join(config.WORKSPACE_DIR, 'work'),
  output: path.join(config.WORKSPACE_DIR, 'output'),
  cache: path.join(config.WORKSPACE_DIR, 'cache'),
};
const CACHE_INDEX_FILE = path.join(DIRS.cache, 'cache-index.json');

function cacheKey(modelId) {
  return modelId.replace(/[^\w.-]/g, '_');
}

function cacheDir(modelId) {
  return path.join(DIRS.cache, cacheKey(modelId));
}

async function validateDir(dirPath) {
  const files = await readdir(dirPath);
  const hasWeights = files.some(
    (f) => f.endsWith('.safetensors') || f.endsWith('.bin') || f.endsWith('.gguf')
  );
  if (!hasWeights) {
    throw new AppError(
      'WORKSPACE_INVALID',
      `No model weight files found in ${dirPath}. Download may have failed.`
    );
  }
  const hasConfig = files.some((f) => f === 'config.json');
  if (!hasConfig) {
    throw new AppError(
      'WORKSPACE_INVALID',
      `No config.json found in ${dirPath}. Is this a valid HuggingFace model?`
    );
  }
  return files;
}

async function readCacheIndex() {
  try {
    const raw = await readFile(CACHE_INDEX_FILE, 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    if (err.code === 'ENOENT') return {};
    return {};
  }
}

async function writeCacheIndex(index) {
  await mkdir(DIRS.cache, { recursive: true });
  await writeFile(CACHE_INDEX_FILE, JSON.stringify(index, null, 2));
}

async function dirSizeBytes(dirPath) {
  let total = 0;
  let entries = [];
  try {
    entries = await readdir(dirPath, { withFileTypes: true });
  } catch {
    return 0;
  }

  for (const entry of entries) {
    const full = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      total += await dirSizeBytes(full);
    } else {
      const info = await stat(full).catch(() => null);
      total += info?.size ?? 0;
    }
  }
  return total;
}

/** Clear work + output only (fast). Does not touch model/ — avoid slow full-tree deletes on WSL + NTFS before cache restore. */
export async function prepareWorkAndOutput() {
  const t0 = Date.now();
  log.info('Preparing workspace (work + output)...');
  await rm(DIRS.work, { recursive: true, force: true });
  await rm(DIRS.output, { recursive: true, force: true });
  for (const dir of [DIRS.root, DIRS.model, DIRS.work, DIRS.output, DIRS.cache]) {
    await mkdir(dir, { recursive: true });
  }
  log.info(`Workspace work/output ready (${Date.now() - t0} ms)`);
}

/** Remove model/ entirely. Can take many minutes for large trees on /mnt/<drive>/ (NTFS via WSL). */
export async function clearModelDir() {
  const t0 = Date.now();
  log.info('Clearing model directory...');
  await rm(DIRS.model, { recursive: true, force: true });
  await mkdir(DIRS.model, { recursive: true });
  log.info(`Model directory cleared (${Date.now() - t0} ms)`);
}

/**
 * Reset all active job directories (including model). Prefer prepareWorkAndOutput + clearModelDir
 * only when needed — avoids double-deleting model before cache restore.
 */
export async function prepare() {
  log.info('Preparing workspace (full reset including model)...');
  const t0 = Date.now();
  await clearModelDir();
  await rm(DIRS.work, { recursive: true, force: true });
  await rm(DIRS.output, { recursive: true, force: true });
  for (const dir of [DIRS.root, DIRS.model, DIRS.work, DIRS.output, DIRS.cache]) {
    await mkdir(dir, { recursive: true });
  }
  log.info(`Workspace ready (${Date.now() - t0} ms)`);
}

/** Clean up active job directories and keep cache for reuse. */
export async function cleanup() {
  await rm(DIRS.model, { recursive: true, force: true }).catch(() => {});
  await rm(DIRS.work, { recursive: true, force: true }).catch(() => {});
  await rm(DIRS.output, { recursive: true, force: true }).catch(() => {});
  log.debug('Workspace cleaned');
}

/** Verify the model directory contains expected files after download. */
export async function validateModelDir() {
  const files = await validateDir(DIRS.model);
  log.info(`Model directory validated: ${files.length} files`);
  return files;
}

/**
 * @param {string} modelId
 * @param {{ onWillCopy?: () => void }} [hooks] Called right before rm+copy from cache (can take a long time on slow disks).
 */
export async function restoreCachedModel(modelId, hooks = {}) {
  if (!config.MODEL_CACHE_ENABLED) return false;
  const src = cacheDir(modelId);
  try {
    if (config.MODEL_CACHE_VALIDATE_ON_RESTORE) {
      await validateDir(src);
    }
  } catch {
    await rm(src, { recursive: true, force: true }).catch(() => {});
    return false;
  }

  hooks.onWillCopy?.();

  await rm(DIRS.model, { recursive: true, force: true });
  await mkdir(DIRS.model, { recursive: true });
  await cp(src, DIRS.model, { recursive: true, force: true });
  const index = await readCacheIndex();
  index[modelId] = {
    ...(index[modelId] ?? {}),
    modelId,
    path: src,
    lastUsedAt: Date.now(),
    sizeBytes: await dirSizeBytes(src),
  };
  await writeCacheIndex(index);
  log.info(`Cache hit: restored ${modelId}`);
  return true;
}

export async function cacheCurrentModel(modelId) {
  if (!config.MODEL_CACHE_ENABLED) return;
  const target = cacheDir(modelId);
  await validateDir(DIRS.model);
  await rm(target, { recursive: true, force: true });
  await mkdir(target, { recursive: true });
  await cp(DIRS.model, target, { recursive: true, force: true });
  const sizeBytes = await dirSizeBytes(target);
  const index = await readCacheIndex();
  index[modelId] = {
    modelId,
    path: target,
    cachedAt: Date.now(),
    lastUsedAt: Date.now(),
    sizeBytes,
  };
  await writeCacheIndex(index);
  await pruneCache('post-cache');
  log.info(`Cached model: ${modelId}`);
}

export async function getCacheStats() {
  const index = await readCacheIndex();
  const entries = Object.values(index);
  const totalBytes = entries.reduce((sum, e) => sum + (e.sizeBytes ?? 0), 0);
  return {
    entries: entries.length,
    totalBytes,
    models: entries.sort((a, b) => (b.lastUsedAt ?? 0) - (a.lastUsedAt ?? 0)),
  };
}

export async function clearCache() {
  await rm(DIRS.cache, { recursive: true, force: true });
  await mkdir(DIRS.cache, { recursive: true });
  await writeCacheIndex({});
}

export async function pruneCache(reason = 'manual') {
  if (!config.MODEL_CACHE_ENABLED) return { removed: 0, reason };
  const index = await readCacheIndex();
  const models = Object.values(index).sort((a, b) => (a.lastUsedAt ?? 0) - (b.lastUsedAt ?? 0));
  const now = Date.now();
  const ttlMs = config.MODEL_CACHE_TTL_HOURS <= 0 ? 0 : config.MODEL_CACHE_TTL_HOURS * 3600 * 1000;
  const keep = {};
  const removed = [];
  let totalBytes = 0;

  for (const model of models) {
    const expired = ttlMs > 0 && now - (model.lastUsedAt ?? model.cachedAt ?? 0) > ttlMs;
    if (expired) {
      removed.push(model.modelId);
      await rm(model.path, { recursive: true, force: true }).catch(() => {});
      continue;
    }
    keep[model.modelId] = model;
    totalBytes += model.sizeBytes ?? 0;
  }

  let ordered = Object.values(keep).sort((a, b) => (a.lastUsedAt ?? 0) - (b.lastUsedAt ?? 0));
  while (
    ordered.length > config.MODEL_CACHE_MAX_ENTRIES ||
    totalBytes > config.MODEL_CACHE_MAX_BYTES
  ) {
    const victim = ordered.shift();
    if (!victim) break;
    removed.push(victim.modelId);
    totalBytes -= victim.sizeBytes ?? 0;
    delete keep[victim.modelId];
    await rm(victim.path, { recursive: true, force: true }).catch(() => {});
  }

  await writeCacheIndex(keep);
  if (removed.length > 0) {
    log.info(
      `Pruned ${removed.length} cache entr${removed.length === 1 ? 'y' : 'ies'} (${reason})`
    );
  }
  return { removed: removed.length, reason };
}

export { DIRS };
