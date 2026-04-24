/**
 * API client for the Python backend.
 *
 * Replaces the local subprocess calls in quantizer.js when BLOCKQUANT_API_URL is set.
 * The bot submits jobs to the API and polls for status.
 */
import { getLogger } from '../logger.js';

const log = getLogger('api-client');

const API_URL = process.env.BLOCKQUANT_API_URL || 'http://localhost:8000';

/**
 * Check if the backend API is reachable.
 * @returns {Promise<boolean>}
 */
export async function isApiAvailable() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const resp = await fetch(`${API_URL}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!resp.ok) return false;
    const data = await resp.json();
    return data.status === 'ok';
  } catch {
    return false;
  }
}

/**
 * Submit a quantization job to the backend API.
 * @param {Object} params
 * @param {string} params.model_id — HF model ID
 * @param {string} params.format — "exl3" | "gguf"
 * @param {string[]} params.variants — e.g., ["4.0"] or ["q4_k_m"]
 * @param {string} [params.provider] — "local" | "lambda" | etc.
 * @param {string} [params.hf_org]
 */
export async function submitJob({ model_id, format = 'exl3', variants, provider = 'local', hf_org = '', parallel_mode = false, high_quality_bpws = [], head_bits_8_bpws = [], verify_quality = true }) {
  const resp = await fetch(`${API_URL}/api/v1/quant`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id, format, variants, provider, hf_org, parallel_mode, high_quality_bpws, head_bits_8_bpws, verify_quality }),
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`API error ${resp.status}: ${err}`);
  }

  return resp.json(); // { job_id, status, check_url }
}

/**
 * Poll job status.
 * @param {string} jobId
 */
export async function getJobStatus(jobId) {
  const resp = await fetch(`${API_URL}/api/v1/jobs/${jobId}`);
  if (!resp.ok) throw new Error('Failed to fetch job status');
  return resp.json();
}

/**
 * Poll loop that updates Discord with progress.
 * Replaces the existing stdout-parsing progress in quantizer.js.
 * @param {string} jobId
 * @param {(status: string, result?: object, error?: string|null, progress?: object|null) => void} onUpdate
 * @returns {Promise<object>} Resolves with the result on completion
 */
export async function pollJob(jobId, onUpdate) {
  const intervalMs = 5000;

  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const status = await getJobStatus(jobId);

        if (status.status === 'complete') {
          onUpdate?.('complete', status.result, null, null);
          resolve(status.result);
          return;
        }

        if (status.status === 'failed') {
          onUpdate?.('failed', null, status.error || 'Unknown error', null);
          reject(new Error(status.error || 'Job failed'));
          return;
        }

        // Still running — report stage
        onUpdate?.(status.status, status.result, null, status.progress);

        setTimeout(tick, intervalMs);
      } catch (err) {
        log.error('Poll error:', err);
        onUpdate?.('error', null, err.message, null);
        setTimeout(tick, intervalMs * 2);
      }
    };

    tick();
  });
}
