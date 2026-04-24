import config from '../config.js';

export class AppError extends Error {
  constructor(code, message, options = {}) {
    super(message);
    this.name = 'AppError';
    this.code = code;
    this.cause = options.cause;
    this.publicMessage = options.publicMessage ?? null;
    this.retryable = options.retryable ?? false;
  }
}

const USER_COPY = {
  AUTH_INVALID: 'Authentication failed. Check your Hugging Face token.',
  AUTH_READ_ONLY: 'Your Hugging Face token is read-only. A write token is required.',
  MODEL_INVALID: 'Invalid model URL or ID. Use format `org/model`.',
  MODEL_NOT_FOUND: 'Model not found or inaccessible with current token.',
  WORKSPACE_INVALID: 'Model files are incomplete on disk. Please retry.',
  HF_DOWNLOAD_FAILED: 'Model download failed. Please retry in a moment.',
  HF_UPLOAD_FAILED: 'Upload failed. Please retry or contact an admin.',
  QUANT_START_FAILED: 'Quantization worker failed to start.',
  QUANT_SETUP_INVALID: 'ExLlamaV3 path is invalid. Set EXLLAMAV3_DIR to your exllamav3 clone.',
  QUANT_EXIT_FAILED: 'Quantization failed during convert step.',
  TIMEOUT: 'Operation timed out. Please retry.',
  INTERNAL: 'Unexpected internal error. Please contact an admin if it persists.',
};

export function toAppError(err, fallbackCode = 'INTERNAL') {
  if (err instanceof AppError) return err;
  const msg = String(err?.message ?? '');
  if (/read-only access/i.test(msg)) return new AppError('AUTH_READ_ONLY', msg, { cause: err });
  if (/model not found|does not exist|not accessible/i.test(msg)) {
    return new AppError('MODEL_NOT_FOUND', msg, { cause: err });
  }
  if (/Cannot parse model ID/i.test(msg)) return new AppError('MODEL_INVALID', msg, { cause: err });
  if (/timed out|timeout/i.test(msg))
    return new AppError('TIMEOUT', msg, { cause: err, retryable: true });
  return new AppError(fallbackCode, msg || 'Unknown error', { cause: err });
}

export function toUserMessage(err) {
  const appErr = toAppError(err);
  if (appErr.publicMessage) return appErr.publicMessage;
  const fallback = USER_COPY[appErr.code] ?? USER_COPY.INTERNAL;
  if (config.PUBLIC_ERROR_REDACTION_ENABLED) return fallback;
  return appErr.message || fallback;
}

export function sanitizeErrorText(text) {
  if (!config.PUBLIC_ERROR_REDACTION_ENABLED) return text;
  return String(text ?? '')
    .replace(/hf_[A-Za-z0-9]{20,}/g, '[REDACTED]')
    .replace(/[MN][A-Za-z\d_-]{20,}\.[A-Za-z\d_-]{6,}\.[A-Za-z\d_-]{20,}/g, '[REDACTED]');
}
