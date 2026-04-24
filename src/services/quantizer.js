import { spawn } from 'child_process';
import path from 'path';
import { access, constants, mkdir, rm } from 'fs/promises';
import config from '../config.js';
import { getLogger } from '../logger.js';
import { DIRS } from './workspace.js';
import { AppError } from '../errors/taxonomy.js';

const log = getLogger('quantizer');

/**
 * Run ExLlamaV3 convert.py for a single bpw.
 *
 * ExLlamaV3 CLI:
 *   python convert.py -i <input> -o <output> -w <workdir> -b <bpw>
 *
 * Unlike V2, this is a single-step process (no separate measurement pass).
 *
 * @param {number} bpw       - Target bits per weight
 * @param {string} modelName - e.g. "Llama-3.1-8B-Instruct"
 * @param {(stage: string, pct: number, msg: string) => void} onProgress
 * @param {AbortSignal} signal - Cancel support
 * @param {{headBits?: number, extraArgs?: string[]}} options
 * @returns {string} Path to the quantized output directory
 */
export async function quantize(bpw, modelName, onProgress, signal, options = {}) {
  const outputDir = path.join(DIRS.output, `${modelName}-${bpw}bpw-exl3`);
  const workDir = path.join(DIRS.work, `bpw-${bpw}`);

  await mkdir(outputDir, { recursive: true });
  await mkdir(workDir, { recursive: true });

  const convertScript = await resolveConvertScriptPath();

  const headBits = options.headBits ?? config.HEAD_BITS;
  const args = [
    '-u',
    convertScript,
    '-i',
    DIRS.model,
    '-o',
    outputDir,
    '-w',
    workDir,
    '-b',
    bpw.toString(),
    '--head_bits',
    headBits.toString(),
  ];
  if (Array.isArray(options.extraArgs) && options.extraArgs.length > 0) {
    args.push(...options.extraArgs);
  }

  log.info(`Starting quantization: ${modelName} @ ${bpw} bpw`);
  onProgress?.('Quantizing', 0, `Starting ${bpw} bpw`);

  await new Promise((resolve, reject) => {
    const isWindows = process.platform === 'win32';
    const pythonCmd = config.PYTHON_BIN || (isWindows ? 'py' : 'python3');
    const pythonArgs = isWindows && !config.PYTHON_BIN ? ['-3', ...args] : args;

    const exllamaCwd = path.dirname(convertScript);
    const proc = spawn(pythonCmd, pythonArgs, {
      cwd: exllamaCwd,
      env: { ...process.env },
    });
    const timeout = setTimeout(() => {
      proc.kill('SIGTERM');
      setTimeout(() => proc.kill('SIGKILL'), config.PROCESS_KILL_GRACE_MS);
    }, config.QUANTIZE_TIMEOUT_MS);

    if (signal) {
      signal.addEventListener(
        'abort',
        () => {
          clearTimeout(timeout);
          proc.kill('SIGTERM');
          reject(new AppError('QUANT_CANCELLED', 'Quantization cancelled'));
        },
        { once: true }
      );
    }

    let lastPct = 0;

    proc.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      log.debug(text.trim());

      // ExLlamaV3 logs progress lines like:
      //   -- Quantized: model.layers.0 bpw: 3.50 rfn: 0.001234 [2.31 s]
      const layerMatch = text.match(/Quantized:\s+(\S+)\s+bpw:\s+([\d.]+)/);
      if (layerMatch) {
        // Estimate progress from layer index
        const layerNum = parseInt(layerMatch[1].match(/\.(\d+)/)?.[1] ?? '0');
        // Rough estimate — will be refined by total layer count
        const pct = Math.min(95, lastPct + 1);
        lastPct = pct;
        onProgress?.('Quantizing', pct, `Layer ${layerNum}: ${layerMatch[2]} bpw`);
      }

      // Compilation phase
      if (text.includes('Compiling')) {
        onProgress?.('Compiling', 95, 'Compiling quantized model...');
      }
    });

    let stderrBuf = '';
    proc.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      stderrBuf += text;
      const trimmed = text.trim();
      if (trimmed) log.debug(`stderr: ${trimmed}`);
    });

    proc.on('error', (err) => {
      clearTimeout(timeout);
      reject(new AppError('QUANT_START_FAILED', `Failed to start convert.py: ${err.message}`));
    });
    proc.on('close', (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        onProgress?.('Quantizing', 100, `${bpw} bpw complete`);
        resolve();
      } else {
        const tail = stderrBuf.trim().slice(-1500);
        reject(
          new AppError(
            'QUANT_EXIT_FAILED',
            tail
              ? `convert.py exited with code ${code}\n${tail}`
              : `convert.py exited with code ${code}`
          )
        );
      }
    });
  });

  log.info(`Quantization complete: ${outputDir}`);
  return outputDir;
}

/** Clean up a single bpw work directory (keep output). */
export async function cleanWorkDir(bpw) {
  const workDir = path.join(DIRS.work, `bpw-${bpw}`);
  await rm(workDir, { recursive: true, force: true }).catch(() => {});
}

export async function validateSetup() {
  await resolveConvertScriptPath();
}

async function resolveConvertScriptPath() {
  const candidates = [
    path.join(config.EXLLAMAV3_DIR, 'convert.py'),
    path.join(config.ROOT_DIR, 'exllamav3', 'convert.py'),
  ];
  for (const convertScript of candidates) {
    try {
      await access(convertScript, constants.F_OK);
      return convertScript;
    } catch {
      /* try next */
    }
  }
  throw new AppError(
    'QUANT_SETUP_INVALID',
    `ExLlamaV3 convert.py not found. Checked:\n${candidates.join('\n')}\nFix EXLLAMAV3_DIR in .env or add exllamav3 next to the repo.`
  );
}
