import { spawn } from 'child_process';
import path from 'path';
import { writeFile } from 'fs/promises';
import config from '../config.js';
import { getLogger } from '../logger.js';
import { DIRS } from './workspace.js';
import { AppError, toAppError } from '../errors/taxonomy.js';
import { exl3TreeUrl } from '../utils/hfExl3.js';

const log = getLogger('huggingface');

/** Parse last JSON object line from Python stdout (handles stray lines or \\r suffix on Windows). */
function parseLastJsonObject(stdout) {
  const lines = String(stdout ?? '')
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (!line.startsWith('{')) continue;
    try {
      return JSON.parse(line);
    } catch {
      /* keep scanning */
    }
  }
  return null;
}

/** Minimal model card if generate_quant_readme.py fails — matches Qwen3.5-4B-8bpw-exl3 README layout. */
async function writeFallbackQuantReadme(outputDir, meta) {
  const repo = meta.sourceModel;
  const shortName = String(repo).split('/').pop() ?? repo;
  const bpw = meta.bpw;
  const owner = config.HF_ORG?.trim() || '';
  const fullRepoId = owner ? `${owner}/${meta.repoSuffix}` : meta.repoSuffix;
  const hfBase = 'https://huggingface.co';
  const branchRow =
    meta.revision && owner
      ? `| **HF branch** | \`${meta.revision}\` ([files](${hfBase}/${fullRepoId}/tree/${encodeURIComponent(meta.revision)})) |\n`
      : '';
  const thisRepoCell = owner
    ? `[${fullRepoId}](${hfBase}/${fullRepoId})`
    : `\`${meta.repoSuffix}\``;
  const revisionInfra =
    meta.revision && owner
      ? `\n- Download this branch: \`huggingface-cli download ${fullRepoId} --revision ${meta.revision}\`\n`
      : '';
  const text = `---
base_model: ${repo}
library_name: exllamav3
tags:
- exl3
- exllamav3
- quantized
- text-generation
- blockquant
- exllama
quantization_format: exl3
bits_per_weight: ${bpw}
---

# ${shortName} — ${bpw} bpw EXL3

This model is an [**EXL3**](https://github.com/turboderp-org/exllamav3)-quantized build of **[${repo}](${hfBase}/${repo})**, produced for GPU inference with **ExLlamaV3**.

| | |
| --- | --- |
| **Base model** | [${repo}](${hfBase}/${repo}) |
| **Format** | EXL3 (ExLlamaV3) |
| **Bits per weight** | ${bpw} |
| **This repo** | ${thisRepoCell} |
${branchRow}
## Inference
${revisionInfra}
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) (OpenAI-compatible API, ExLlamaV2/V3)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) with the **ExLlamaV3** loader
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) directly

## License and use

Use and license follow the **base model**. See the base repository for terms, citation, and safety documentation.

---
## Original model README (reference)

_No upstream README was embedded (fallback card). See the base model repo for full documentation._
`;
  await writeFile(path.join(outputDir, 'README.md'), text, 'utf8');
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function resolvePythonLaunch(args) {
  if (config.PYTHON_BIN) {
    return { cmd: config.PYTHON_BIN, cmdArgs: args };
  }
  const isWindows = process.platform === 'win32';
  if (isWindows) return { cmd: 'py', cmdArgs: ['-3', ...args] };
  return { cmd: 'python3', cmdArgs: args };
}

function retryDelay(attempt) {
  const exp = Math.min(config.HF_RETRY_MAX_DELAY_MS, config.HF_RETRY_BASE_DELAY_MS * 2 ** attempt);
  const jitter = Math.floor((exp * config.HF_RETRY_JITTER_PCT) / 100);
  return exp + Math.floor(Math.random() * (jitter + 1));
}

function isRetryable(message) {
  return /(timed out|timeout|temporarily unavailable|ECONNRESET|ENOTFOUND|HTTP 5\d\d|429)/i.test(
    message
  );
}

/** Prefer script JSON errors on stdout; stderr often only has HF UserWarning noise. */
function formatPythonExitError(code, stdout, stderr) {
  const out = String(stdout ?? '');
  const err = String(stderr ?? '');
  const lines = out.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const o = JSON.parse(lines[i]);
      if (o && typeof o.error === 'string' && o.error.length) {
        return `Python exited ${code}: ${o.error}`;
      }
    } catch {
      /* not JSON */
    }
  }
  const stderrTail = err.trim().slice(-1200);
  const stdoutTail = out.trim().slice(-1200);
  const stderrLooksLikeOnlyHfWarnings =
    /UserWarning|HF_HUB_DISABLE_EXPERIMENTAL|experimental/i.test(stderrTail) &&
    !/Traceback|Error:|Exception/i.test(stderrTail);
  if (stderrLooksLikeOnlyHfWarnings && stdoutTail) {
    return `Python exited ${code}: ${stdoutTail}`;
  }
  const detail = stderrTail || stdoutTail || 'No error output from Python process';
  return `Python exited ${code}: ${detail}`;
}

async function runPythonOnce(args, { onStdout, onStderr, timeoutMs, env = {} } = {}) {
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    const { cmd, cmdArgs } = resolvePythonLaunch(args);

    const proc = spawn(cmd, cmdArgs, {
      env: { ...process.env, ...env },
    });
    let timeout;
    if (timeoutMs > 0) {
      timeout = setTimeout(() => {
        proc.kill('SIGTERM');
        setTimeout(() => proc.kill('SIGKILL'), config.PROCESS_KILL_GRACE_MS);
      }, timeoutMs);
    }

    proc.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      stdout += text;
      onStdout?.(text);
    });
    proc.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      stderr += text;
      onStderr?.(text);
    });
    proc.on('error', (err) => {
      if (timeout) clearTimeout(timeout);
      reject(err);
    });
    proc.on('close', (code) => {
      if (timeout) clearTimeout(timeout);
      if (code === 0) return resolve(stdout);
      reject(new Error(formatPythonExitError(code, stdout, stderr)));
    });
  });
}

async function runPython(args, options = {}) {
  const attempts = Math.max(1, options.attempts ?? 1);
  let lastError;

  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await runPythonOnce(args, options);
    } catch (err) {
      lastError = err;
      const retryable = options.retryable !== false && isRetryable(err.message ?? '');
      if (!retryable || attempt === attempts - 1) break;
      const waitMs = retryDelay(attempt);
      log.warn(`Retrying Python step after error (${attempt + 1}/${attempts})`, {
        error: err.message,
        waitMs,
      });
      await sleep(waitMs);
    }
  }
  throw lastError;
}

// ── Pre-flight Checks ───────────────────────────────────────────────────────

/**
 * Validate the HF token has write access and (optionally) check a model exists.
 * Returns { username, modelExists, modelId }.
 */
export async function preflight(modelUrl) {
  const scriptPath = path.join(config.ROOT_DIR, 'scripts', 'preflight.py');
  const args = [scriptPath];

  if (modelUrl) {
    const modelId = parseModelId(modelUrl);
    args.push('--model', modelId);
  }

  const raw = await runPython(args, {
    timeoutMs: config.HF_PREFLIGHT_TIMEOUT_MS,
    attempts: 1,
    retryable: false,
    env: { HF_TOKEN: config.HF_TOKEN, HF_HUB_ENABLE_HF_TRANSFER: '1' },
  });
  try {
    const result = JSON.parse(raw.trim().split('\n').pop());
    if (result.error) throw new AppError('AUTH_INVALID', result.error);
    return result;
  } catch (err) {
    if (err instanceof AppError) throw err;
    throw toAppError(new Error(`Preflight check failed: ${raw.slice(-300)}`), 'AUTH_INVALID');
  }
}

// ── Download ────────────────────────────────────────────────────────────────

export async function downloadModel(modelUrl, onProgress) {
  const modelId = parseModelId(modelUrl);
  const scriptPath = path.join(config.ROOT_DIR, 'scripts', 'download_model.py');

  log.info(`Downloading model: ${modelId}`);
  onProgress?.('Downloading', 0, `Starting download of ${modelId}`);

  try {
    await runPython([scriptPath, modelId, DIRS.model], {
      timeoutMs: config.HF_DOWNLOAD_TIMEOUT_MS,
      attempts: config.HF_RETRY_MAX_ATTEMPTS,
      env: { HF_TOKEN: config.HF_TOKEN, HF_HUB_ENABLE_HF_TRANSFER: '1' },
      onStdout: (text) => parseStatus(text, onProgress),
      onStderr: (text) => {
        // huggingface_hub logs progress to stderr
        if (text.includes('%|')) {
          const match = text.match(/(\d+)%\|/);
          if (match) onProgress?.('Downloading', parseInt(match[1]), 'Downloading files...');
        }
      },
    });
  } catch (err) {
    throw toAppError(err, 'HF_DOWNLOAD_FAILED');
  }

  log.info(`Download complete: ${modelId}`);
}

// ── Model card (README) ────────────────────────────────────────────────────

/**
 * Write README.md into outputDir: HF card YAML + quant intro + upstream base README.
 * Best-effort; failures are logged and do not throw (upload still proceeds).
 */
export async function generateQuantReadme(outputDir, meta) {
  const scriptPath = path.join(config.ROOT_DIR, 'scripts', 'generate_quant_readme.py');
  const outFile = path.join(outputDir, 'README.md');
  const owner = config.HF_ORG || '';
  const args = [
    scriptPath,
    '--output',
    outFile,
    '--source-repo',
    meta.sourceModel,
    '--repo-name',
    meta.repoSuffix,
    '--bpw',
    String(meta.bpw),
    '--org',
    owner,
  ];
  if (meta.revision) {
    args.push('--revision', String(meta.revision));
  }

  try {
    const raw = await runPython(args, {
      timeoutMs: Math.min(config.HF_PREFLIGHT_TIMEOUT_MS * 2, 120_000),
      attempts: 2,
      retryable: true,
      env: { HF_TOKEN: config.HF_TOKEN },
    });
    const j = parseLastJsonObject(raw);
    if (j?.ok) {
      log.info(`Wrote model README: ${outFile}`, { upstream_readme_found: j.upstream_readme_found });
      return;
    }
    if (j && j.ok === false) {
      log.error('generate_quant_readme reported failure', { error: j.error });
    } else {
      log.error('generate_quant_readme: unexpected output (no JSON ok line)', {
        tail: String(raw ?? '').trim().slice(-400),
      });
    }
  } catch (err) {
    log.error('generate_quant_readme failed', { error: err.message });
  }
  try {
    await writeFallbackQuantReadme(outputDir, meta);
    log.info(`Wrote fallback model README: ${outFile}`);
  } catch (err) {
    log.error('Could not write fallback README.md (upload will lack model card)', { error: err.message });
  }
}

// ── Upload ──────────────────────────────────────────────────────────────────

export async function uploadModel(folderPath, repoName, onProgress, options = {}) {
  const scriptPath = path.join(config.ROOT_DIR, 'scripts', 'upload_model.py');
  const owner = config.HF_ORG || ''; // script will resolve to whoami if empty
  const revision = options.revision ? String(options.revision) : '';

  log.info(`Uploading ${repoName} from ${folderPath}`);
  onProgress?.('Uploading', 0, `Uploading ${repoName}${revision ? `@${revision}` : ''}`);

  const pyArgs = [scriptPath, folderPath, repoName, '--org', owner];
  if (revision) pyArgs.push('--revision', revision);

  let raw;
  try {
    raw = await runPython(pyArgs, {
      timeoutMs: config.HF_UPLOAD_TIMEOUT_MS,
      attempts: config.HF_RETRY_MAX_ATTEMPTS,
      env: { HF_TOKEN: config.HF_TOKEN, HF_HUB_ENABLE_HF_TRANSFER: '1' },
      onStdout: (text) => parseStatus(text, onProgress),
    });
  } catch (err) {
    throw toAppError(err, 'HF_UPLOAD_FAILED');
  }

  const lines = raw.trim().split('\n');
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const result = JSON.parse(lines[i]);
      if (result.url) {
        const treeUrl = result.tree_url ?? exl3TreeUrl(result.url);
        log.info(`Upload success: ${result.url}`);
        return { ...result, tree_url: treeUrl };
      }
      if (result.error) throw new AppError('HF_UPLOAD_FAILED', result.error);
    } catch {
      /* keep looking */
    }
  }
  throw new AppError('HF_UPLOAD_FAILED', 'Upload finished but no result URL was returned');
}

export async function inspectUploadRepo(repoName, expected = {}) {
  const scriptPath = path.join(config.ROOT_DIR, 'scripts', 'check_repo.py');
  const owner = config.HF_ORG || '';
  const args = [
    scriptPath,
    repoName,
    '--org',
    owner,
    '--source_model',
    expected.sourceModel ?? '',
    '--profile',
    expected.profile ?? '',
    '--bpw',
    expected.bpw != null ? String(expected.bpw) : '',
    '--quant_options_json',
    JSON.stringify(expected.quantOptions ?? {}),
  ];
  if (expected.revision) {
    args.push('--revision', String(expected.revision));
  }

  const raw = await runPython(args, {
    timeoutMs: config.HF_PREFLIGHT_TIMEOUT_MS,
    attempts: 1,
    retryable: false,
    env: { HF_TOKEN: config.HF_TOKEN },
  });
  const lines = raw.trim().split('\n').filter(Boolean);
  const last = lines[lines.length - 1] ?? '{}';
  try {
    const result = JSON.parse(last);
    if (result.error) throw new AppError('HF_UPLOAD_FAILED', result.error);
    return result;
  } catch (err) {
    if (err instanceof AppError) throw err;
    throw toAppError(new Error(`Repo inspection failed: ${raw.slice(-500)}`), 'HF_UPLOAD_FAILED');
  }
}

// ── Utilities ───────────────────────────────────────────────────────────────

/** Extract "org/model" from various URL formats. */
export function parseModelId(input) {
  // Already in org/model format
  if (/^[\w.-]+\/[\w.-]+$/.test(input)) return input;

  try {
    const url = new URL(input);
    const parts = url.pathname.split('/').filter(Boolean);
    if (parts.length >= 2) return `${parts[0]}/${parts[1]}`;
  } catch {
    /* not a URL */
  }

  throw new Error(`Cannot parse model ID from: ${input}`);
}

function parseStatus(text, onProgress) {
  if (!onProgress) return;
  const matches = text.match(/\[STATUS\](.*?)\[\/STATUS\]/g);
  if (!matches) return;
  for (const m of matches) {
    try {
      const data = JSON.parse(m.replace(/^\[STATUS\]|\[\/STATUS\]$/g, ''));
      const pct = Math.min(100, Math.round(parseFloat(data.completion) * 100));
      onProgress(data.stage, pct, data.status || '');
    } catch {
      /* ignore malformed status */
    }
  }
}
