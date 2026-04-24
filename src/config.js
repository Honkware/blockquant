import fs from 'fs';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const envFilePath = path.join(root, '.env');

// Always load repo .env (not cwd), so paths match regardless of how the process is started.
dotenv.config({ path: envFilePath });

/**
 * Read a key directly from .env so repo file wins over broken EXLLAMAV3_DIR / WORKSPACE_DIR
 * inherited from Windows or the shell (dotenv does not override existing process.env).
 */
function readEnvFileValue(filePath, key) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    for (const line of content.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const eq = trimmed.indexOf('=');
      if (eq <= 0) continue;
      const k = trimmed.slice(0, eq).trim();
      if (k !== key) continue;
      let v = trimmed.slice(eq + 1).trim();
      if (!v) return '';
      if (
        (v.startsWith('"') && v.endsWith('"')) ||
        (v.startsWith("'") && v.endsWith("'"))
      ) {
        return v.slice(1, -1);
      }
      v = v.replace(/\s+#.*$/, '').trim();
      return v;
    }
  } catch {
    /* no .env */
  }
  return null;
}

const workspaceDirRaw = readEnvFileValue(envFilePath, 'WORKSPACE_DIR') ?? process.env.WORKSPACE_DIR;
const exllamav3DirRaw = readEnvFileValue(envFilePath, 'EXLLAMAV3_DIR') ?? process.env.EXLLAMAV3_DIR;

/**
 * Resolve a path from env for this repo. On Linux/macOS, turns Windows drive paths (E:/foo)
 * into WSL-style /mnt/e/foo so EXLLAMAV3_DIR still works when inherited from Windows or shell.
 */
function resolveRepoPath(raw, defaultRel) {
  const val = (raw ?? '').trim();
  const effective = val || defaultRel;
  const posix = effective.replace(/\\/g, '/');
  const winDrive = posix.match(/^([A-Za-z]):\/?(.*)$/);
  if (winDrive && process.platform !== 'win32') {
    const drive = winDrive[1].toLowerCase();
    const tail = winDrive[2].replace(/^\/+/, '');
    return path.normalize(path.join('/mnt', drive, tail));
  }
  if (path.isAbsolute(effective)) {
    return path.normalize(effective);
  }
  return path.normalize(path.resolve(root, effective));
}

function intEnv(key, fallback) {
  const raw = process.env[key];
  if (!raw) return fallback;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

function boolEnv(key, fallback) {
  const raw = process.env[key];
  if (raw == null || raw === '') return fallback;
  return ['1', 'true', 'yes', 'on'].includes(raw.toLowerCase());
}

function required(key) {
  const val = process.env[key];
  if (!val) {
    console.error(`[FATAL] Missing required env var: ${key}`);
    console.error('Copy .env.example to .env and fill in the values.');
    process.exit(1);
  }
  return val;
}

const config = Object.freeze({
  // Discord
  BOT_TOKEN: required('BOT_TOKEN'),
  CLIENT_ID: required('CLIENT_ID'),
  GUILD_ID: required('GUILD_ID'),

  // HuggingFace
  HF_TOKEN: required('HF_TOKEN'),
  HF_ORG: process.env.HF_ORG || '',

  // Access
  ADMIN_IDS: (process.env.ADMIN_IDS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean),

  // Paths (.env file overrides inherited env for these two)
  WORKSPACE_DIR: resolveRepoPath(workspaceDirRaw, './tmp/workdir'),
  EXLLAMAV3_DIR: resolveRepoPath(exllamav3DirRaw, './exllamav3'),
  ROOT_DIR: root,

  // Runtime
  LOG_LEVEL: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  CONCURRENT_JOBS: intEnv('CONCURRENT_JOBS', 1),
  SHUTDOWN_TIMEOUT_MS: intEnv('SHUTDOWN_TIMEOUT_MS', 60_000),
  PROCESS_KILL_GRACE_MS: intEnv('PROCESS_KILL_GRACE_MS', 5_000),
  PYTHON_BIN: process.env.PYTHON_BIN || '',

  // Retry and timeout policy
  HF_PREFLIGHT_TIMEOUT_MS: intEnv('HF_PREFLIGHT_TIMEOUT_MS', 30_000),
  HF_DOWNLOAD_TIMEOUT_MS: intEnv('HF_DOWNLOAD_TIMEOUT_MS', 60 * 60 * 1000),
  HF_UPLOAD_TIMEOUT_MS: intEnv('HF_UPLOAD_TIMEOUT_MS', 60 * 60 * 1000),
  QUANTIZE_TIMEOUT_MS: intEnv('QUANTIZE_TIMEOUT_MS', 12 * 60 * 60 * 1000),
  HF_RETRY_MAX_ATTEMPTS: intEnv('HF_RETRY_MAX_ATTEMPTS', 3),
  HF_RETRY_BASE_DELAY_MS: intEnv('HF_RETRY_BASE_DELAY_MS', 1500),
  HF_RETRY_MAX_DELAY_MS: intEnv('HF_RETRY_MAX_DELAY_MS', 20_000),
  HF_RETRY_JITTER_PCT: intEnv('HF_RETRY_JITTER_PCT', 25),

  // Cache governance
  MODEL_CACHE_ENABLED: boolEnv('MODEL_CACHE_ENABLED', true),
  MODEL_CACHE_TTL_HOURS: intEnv('MODEL_CACHE_TTL_HOURS', 72),
  MODEL_CACHE_MAX_ENTRIES: intEnv('MODEL_CACHE_MAX_ENTRIES', 5),
  MODEL_CACHE_MAX_BYTES: intEnv('MODEL_CACHE_MAX_BYTES', 200 * 1024 * 1024 * 1024),
  MODEL_CACHE_PRUNE_ON_STARTUP: boolEnv('MODEL_CACHE_PRUNE_ON_STARTUP', true),
  MODEL_CACHE_VALIDATE_ON_RESTORE: boolEnv('MODEL_CACHE_VALIDATE_ON_RESTORE', true),

  // Logging and redaction
  LOG_REDACTION_ENABLED: boolEnv('LOG_REDACTION_ENABLED', true),
  LOG_INCLUDE_STACKS: boolEnv('LOG_INCLUDE_STACKS', process.env.NODE_ENV !== 'production'),
  PUBLIC_ERROR_REDACTION_ENABLED: boolEnv('PUBLIC_ERROR_REDACTION_ENABLED', true),

  // Quantization presets  (ExLlamaV3 uses bpw, typically 2-8)
  BPW_OPTIONS: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 8.0],
  HEAD_BITS: intEnv('HEAD_BITS', 6),
  QUANT_PROFILES: Object.freeze({
    fast: { headBits: 4 },
    balanced: { headBits: 6 },
    quality: { headBits: 8 },
  }),
});

export default config;
