import { mkdir } from 'fs/promises';
import { join } from 'path';
import { config } from '../config.js';

export async function init() {
  await mkdir(config.workspace.path, { recursive: true });
  await mkdir(config.workspace.cache, { recursive: true });
  await mkdir('data', { recursive: true });
  await mkdir('logs', { recursive: true });
}

export function path(...parts) {
  return join(config.workspace.path, ...parts);
}
