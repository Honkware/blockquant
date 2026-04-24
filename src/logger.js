import winston from 'winston';
import path from 'path';
import { mkdirSync } from 'fs';
import config from './config.js';

const logsDir = path.join(config.ROOT_DIR, 'logs');
mkdirSync(logsDir, { recursive: true });

const fmt = winston.format;
const redactionRules = [
  /hf_[A-Za-z0-9]{20,}/g,
  /[MN][A-Za-z\d_-]{20,}\.[A-Za-z\d_-]{6,}\.[A-Za-z\d_-]{20,}/g,
  /(token|authorization|password|secret)\s*[:=]\s*([^\s,]+)/gi,
];

function redactText(value) {
  if (!config.LOG_REDACTION_ENABLED || typeof value !== 'string') return value;
  let out = value;
  for (const rule of redactionRules) {
    out = out.replace(rule, (match, key) => (key ? `${key}=[REDACTED]` : '[REDACTED]'));
  }
  return out;
}

function redactObject(input) {
  if (input == null) return input;
  if (typeof input === 'string') return redactText(input);
  if (Array.isArray(input)) return input.map((v) => redactObject(v));
  if (typeof input !== 'object') return input;

  const out = {};
  for (const [k, v] of Object.entries(input)) {
    if (/(token|authorization|password|secret|api[_-]?key)/i.test(k)) {
      out[k] = '[REDACTED]';
      continue;
    }
    out[k] = redactObject(v);
  }
  return out;
}

const redactFormat = fmt((info) => {
  for (const [k, v] of Object.entries(info)) {
    if (/(token|authorization|password|secret|api[_-]?key)/i.test(k)) {
      info[k] = '[REDACTED]';
      continue;
    }
    if (typeof v === 'string') {
      info[k] = redactText(v);
    } else if (v && typeof v === 'object') {
      info[k] = redactObject(v);
    }
  }
  if (!config.LOG_INCLUDE_STACKS) {
    delete info.stack;
  } else if (typeof info.stack === 'string') {
    info.stack = redactText(info.stack);
  }
  return info;
});

const consoleFormat = fmt.printf(({ timestamp, level, module, message }) => {
  const tag = module ? `[${module}]` : '';
  return `${timestamp} ${level} ${tag} ${redactText(String(message ?? ''))}`;
});

const logger = winston.createLogger({
  level: config.LOG_LEVEL,
  format: fmt.combine(
    fmt.timestamp({ format: 'HH:mm:ss' }),
    fmt.errors({ stack: true }),
    redactFormat(),
    fmt.json()
  ),
  transports: [
    new winston.transports.File({ filename: path.join(logsDir, 'error.log'), level: 'error' }),
    new winston.transports.File({ filename: path.join(logsDir, 'combined.log') }),
    new winston.transports.Console({
      format: fmt.combine(fmt.colorize(), fmt.timestamp({ format: 'HH:mm:ss' }), consoleFormat),
    }),
  ],
});

/** Create a child logger tagged with a module name. */
export function getLogger(moduleName) {
  return logger.child({ module: moduleName });
}

export default logger;
