import PQueue from 'p-queue';
import { writeFile } from 'fs/promises';
import path from 'path';
import config from '../config.js';
import { getLogger } from '../logger.js';
import * as db from './db.js';
import * as workspace from './workspace.js';
import * as hf from './huggingface.js';
import * as quantizer from './quantizer.js';
import { formatDuration } from '../utils/format.js';
import { AppError } from '../errors/taxonomy.js';
import { exl3RepoName, formatExl3Revision, exl3TreeUrl } from '../utils/hfExl3.js';

const log = getLogger('queue');

let queue;
let paused = false;
let presenceUpdater = () => {};
const activeJobs = new Map();

async function writeManifest(outputDir, payload) {
  const manifestPath = path.join(outputDir, 'blockquant-manifest.json');
  await writeFile(manifestPath, JSON.stringify(payload, null, 2));
}

function updatePresence(payload) {
  try {
    presenceUpdater(payload);
  } catch (err) {
    log.debug(`Presence update failed: ${err.message}`);
  }
}

export function setPresenceUpdater(fn) {
  presenceUpdater = typeof fn === 'function' ? fn : () => {};
}

// ── Init ────────────────────────────────────────────────────────────────────

export function init() {
  queue = new PQueue({ concurrency: config.CONCURRENT_JOBS });
  log.info(`Job queue initialized (concurrency: ${config.CONCURRENT_JOBS})`);
  if (config.MODEL_CACHE_PRUNE_ON_STARTUP) {
    workspace
      .pruneCache('startup')
      .catch((err) => log.debug(`Cache prune failed at startup: ${err.message}`));
  }
  updatePresence({ state: 'idle', waiting: 0, active: 0 });
}

export function status() {
  return { waiting: queue?.size ?? 0, active: queue?.pending ?? 0, paused };
}

export function diagnostics() {
  return {
    waiting: queue?.size ?? 0,
    active: queue?.pending ?? 0,
    paused,
    activeJobs: Array.from(activeJobs.values()),
  };
}

export function pause() {
  queue.pause();
  paused = true;
  log.info('Queue paused');
  updatePresence({ state: 'paused', waiting: queue?.size ?? 0, active: queue?.pending ?? 0 });
}
export function resume() {
  queue.start();
  paused = false;
  log.info('Queue resumed');
  updatePresence({ state: 'active', waiting: queue?.size ?? 0, active: queue?.pending ?? 0 });
}

// ── Job ─────────────────────────────────────────────────────────────────────

/**
 * @typedef {Object} JobConfig
 * @property {string}   url
 * @property {number[]} bpws
 * @property {string[]} categories
 * @property {string}   userId
 * @property {string}   [jobId]
 * @property {string}   [profile]
 * @property {Object}   [quantOptions]
 * @property {Object}   [precheckedRepos]
 * @property {(data: object) => void} onProgress
 * @property {(results: object[]) => void} onComplete
 * @property {(error: Error) => void} onError
 */

/**
 * Enqueue a full quantization pipeline:
 *   1. Prepare workspace
 *   2. Download model
 *   3. Validate model files
 *   4. For each bpw: quantize → upload
 *   5. Cleanup
 */
export function enqueue(jobConfig) {
  const controller = new AbortController();

  const promise = queue.add(async () => {
    const results = [];
    const modelId = hf.parseModelId(jobConfig.url);
    const modelName = path.basename(modelId.split('/').pop());
    const jobId = jobConfig.jobId ?? null;
    let lastPersistAt = 0;
    if (jobId) {
      activeJobs.set(jobId, {
        jobId,
        modelId,
        userId: jobConfig.userId,
        startedAt: Date.now(),
      });
    }

    const reportProgress = (data) => {
      jobConfig.onProgress(data);
      updatePresence({
        state: 'active',
        modelId,
        stage: data.stage,
        currentBPW: data.currentBPW,
        waiting: queue?.size ?? 0,
        active: queue?.pending ?? 0,
      });

      if (!jobId) return;
      const now = Date.now();
      if (now - lastPersistAt < 1200 && data.stage === 'Quantizing') return;
      lastPersistAt = now;

      db.patchJob(jobId, (job) => ({
        ...job,
        status: db.JOB_STATUS.running,
        progress: {
          stage: data.stage,
          progress: data.progress ?? 0,
          overall: data.overall ?? 0,
          message: data.message ?? '',
          currentBPW: data.currentBPW ?? null,
          bpwIndex: data.bpwIndex ?? null,
          totalBPWs: data.totalBPWs ?? null,
          updatedAt: now,
        },
      })).catch((err) => log.debug(`Failed to persist progress: ${err.message}`));
    };

    try {
      if (jobId) {
        await db.patchJob(jobId, (job) => ({
          ...job,
          status: db.JOB_STATUS.running,
          startedAt: job.startedAt ?? Date.now(),
        }));
      }

      // ── 1. Workspace ─────────────────────────────────────────────
      reportProgress({
        stage: 'Preparing',
        progress: 0,
        overall: 0,
        message: 'Setting up workspace (work/output)...',
      });
      await workspace.prepareWorkAndOutput();

      // ── 2. Restore from cache or download ───────────────────────
      reportProgress({
        stage: 'Preparing',
        progress: 5,
        overall: 2,
        message: 'Checking model cache...',
      });
      const restored = await workspace.restoreCachedModel(modelId, {
        onWillCopy: () => {
          reportProgress({
            stage: 'Cached',
            progress: 10,
            overall: 4,
            message:
              'Copying cached model into workspace (slow on WSL + NTFS — not frozen)...',
          });
        },
      });
      if (restored) {
        reportProgress({
          stage: 'Cached',
          progress: 100,
          overall: 15,
          message: `Using cached model ${modelId}`,
        });
      } else {
        reportProgress({
          stage: 'Preparing',
          progress: 0,
          overall: 2,
          message:
            'Clearing old model files (can take several minutes on WSL + E: / NTFS — not frozen)...',
        });
        await workspace.clearModelDir();
        await hf.downloadModel(jobConfig.url, (stage, pct, msg) => {
          reportProgress({ stage, progress: pct, overall: pct * 0.15, message: msg });
        });
      }

      // ── 3. Validate ──────────────────────────────────────────────
      await workspace.validateModelDir();
      if (!restored) {
        await workspace.cacheCurrentModel(modelId);
      }

      // ── 4. Quantize + Upload each BPW ────────────────────────────
      const bpwWeight = 0.85 / jobConfig.bpws.length;

      for (let i = 0; i < jobConfig.bpws.length; i++) {
        if (controller.signal.aborted) break;

        const bpw = jobConfig.bpws[i];
        const baseOffset = 0.15 + i * bpwWeight;
        const repoSuffix = exl3RepoName(modelName);
        const revision = formatExl3Revision(bpw);
        const prechecked = jobConfig.precheckedRepos?.[String(bpw)];
        const repoState =
          prechecked ??
          (await hf.inspectUploadRepo(repoSuffix, {
            sourceModel: modelId,
            profile: jobConfig.profile ?? 'balanced',
            bpw,
            quantOptions: jobConfig.quantOptions ?? {},
            revision,
          }));

        if (repoState.exists && repoState.settingsMatch) {
          const repoBaseUrl = repoState.url || `https://huggingface.co/${repoState.repoId}`;
          const existingUrl = exl3TreeUrl(repoBaseUrl, revision) ?? repoBaseUrl;
          reportProgress({
            stage: 'Reusing',
            progress: 100,
            overall: Math.round((baseOffset + bpwWeight) * 100),
            message: `Found existing matching upload for ${bpw} bpw`,
            currentBPW: bpw,
            bpwIndex: i,
            totalBPWs: jobConfig.bpws.length,
          });
          results.push({
            bpw,
            duration: '0s (reused)',
            url: repoState.url || `https://huggingface.co/${repoState.repoId}`,
            treeUrl: existingUrl,
            revision,
            pushed: true,
            reused: true,
            error: null,
          });
          continue;
        }

        if (
          repoState.exists &&
          repoState.settingsMatch === false &&
          repoState.reason !== 'manifest_missing'
        ) {
          throw new AppError(
            'HF_UPLOAD_FAILED',
            `Upload target ${repoState.repoId}@${revision} already has a manifest with different settings (${repoState.reason ?? 'manifest mismatch'}).`
          );
        }

        // Quantize
        const startTime = Date.now();
        const outputDir = await quantizer.quantize(
          bpw,
          modelName,
          (stage, pct, msg) => {
            const overall = baseOffset + (pct / 100) * bpwWeight * 0.7;
            reportProgress({
              stage,
              progress: pct,
              overall: Math.round(overall * 100),
              message: msg,
              currentBPW: bpw,
              bpwIndex: i,
              totalBPWs: jobConfig.bpws.length,
            });
          },
          controller.signal,
          jobConfig.quantOptions
        );

        const duration = formatDuration(Date.now() - startTime);

        await hf.generateQuantReadme(outputDir, {
          sourceModel: modelId,
          bpw,
          repoSuffix,
          revision,
        });

        // Upload
        const uploadOffset = baseOffset + bpwWeight * 0.7;

        let uploadResult;
        try {
          uploadResult = await hf.uploadModel(
            outputDir,
            repoSuffix,
            (stage, pct, msg) => {
              const overall = uploadOffset + (pct / 100) * bpwWeight * 0.3;
              reportProgress({
                stage,
                progress: pct,
                overall: Math.round(overall * 100),
                message: msg,
                currentBPW: bpw,
                bpwIndex: i,
                totalBPWs: jobConfig.bpws.length,
              });
            },
            { revision }
          );
        } catch (err) {
          log.error(`Upload failed for ${bpw} bpw`, { error: err.message });
          uploadResult = { url: null, error: err.message };
        }

        const treeUrl =
          uploadResult?.tree_url ??
          exl3TreeUrl(uploadResult?.url ?? null, revision);

        results.push({
          bpw,
          duration,
          url: uploadResult?.url ?? null,
          treeUrl,
          revision,
          pushed: !!uploadResult?.url,
          error: uploadResult?.error ?? null,
        });

        await writeManifest(outputDir, {
          version: 1,
          generatedAt: new Date().toISOString(),
          sourceModel: modelId,
          profile: jobConfig.profile ?? 'balanced',
          quantOptions: jobConfig.quantOptions ?? {},
          bpw,
          hfRepo: repoSuffix,
          hfRevision: revision,
          duration,
          upload: {
            pushed: !!uploadResult?.url,
            url: uploadResult?.url ?? null,
            treeUrl,
            error: uploadResult?.error ?? null,
          },
        }).catch((err) => log.debug(`Manifest write skipped: ${err.message}`));

        if (jobId) {
          await db.patchJob(jobId, (job) => ({
            ...job,
            partialResults: results,
          }));
        }

        // Free work dir for this bpw
        await quantizer.cleanWorkDir(bpw);
      }

      const pushedCount = results.filter((r) => r.pushed).length;
      if (results.length > 0 && pushedCount === 0) {
        const lastError = results[results.length - 1]?.error ?? 'Unknown upload error';
        throw new AppError('HF_UPLOAD_FAILED', `All uploads failed. ${lastError}`);
      }

      if (jobId) {
        await db.patchJob(jobId, (job) => ({
          ...job,
          status: db.JOB_STATUS.completed,
          partialResults: results,
          completedAt: Date.now(),
        }));
      }
      jobConfig.onComplete(results);
    } catch (err) {
      log.error('Job failed', { error: err.message, stack: err.stack });
      if (jobId) {
        await db.patchJob(jobId, (job) => ({
          ...job,
          status: db.JOB_STATUS.failed,
          error: err.message,
          failedAt: Date.now(),
          partialResults: results,
        }));
        if (jobConfig.userId && jobConfig.cost) {
          await db
            .refundForJob({
              jobId,
              userId: jobConfig.userId,
              cost: jobConfig.cost,
            })
            .catch((refundErr) => log.error('Refund failed', { error: refundErr.message }));
        }
      }
      jobConfig.onError(err);
    } finally {
      if (jobId) activeJobs.delete(jobId);
      await workspace.cleanup();
      setTimeout(() => {
        const waiting = queue?.size ?? 0;
        const active = queue?.pending ?? 0;
        if (paused) {
          updatePresence({ state: 'paused', waiting, active });
        } else if (active === 0) {
          updatePresence({ state: 'idle', waiting, active });
        }
      }, 0);
    }
  });

  return { promise, cancel: () => controller.abort() };
}

export async function shutdown() {
  log.info('Shutting down queue...');
  const recoverable = await db.listRecoverableJobs().catch(() => []);
  for (const job of recoverable) {
    await db.patchJob(job.id, {
      status: db.JOB_STATUS.interrupted,
      interruptedAt: Date.now(),
    });
  }
  queue?.pause();
  queue?.clear();
  await workspace.cleanup();
  log.info('Queue shutdown complete');
}

export async function recoverPersistedJobs() {
  const jobs = await db.listRecoverableJobs();
  if (!jobs.length) return 0;

  log.info(`Recovering ${jobs.length} persisted job(s)`);

  for (const job of jobs) {
    await db.patchJob(job.id, {
      status: db.JOB_STATUS.queued,
      recoveredAt: Date.now(),
    });

    enqueue({
      jobId: job.id,
      url: job.url,
      bpws: job.bpws ?? [],
      categories: job.categories ?? ['General'],
      userId: job.userId,
      cost: job.cost,
      quantOptions: job.quantOptions,
      onProgress: () => {},
      onComplete: async (results) => {
        if (!job.modelId) return;
        const models = await db.loadModels();
        const existing = models[job.modelId] ?? {};
        models[job.modelId] = {
          ...existing,
          url: job.modelId,
          categories: job.categories ?? existing.categories ?? ['General'],
          authors: Array.from(new Set([...(existing.authors ?? []), job.userId].filter(Boolean))),
          quantizedBPWs: results.filter((r) => r.pushed).map((r) => r.bpw),
          results,
          completedAt: Date.now(),
        };
        await db.saveModels(models);
      },
      onError: () => {},
    });
  }

  return jobs.length;
}
