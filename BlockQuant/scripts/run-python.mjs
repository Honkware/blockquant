import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);

if (args.length === 0) {
  console.error('Usage: node scripts/run-python.mjs <script.py> [args...]');
  process.exit(1);
}

const candidates =
  process.platform === 'win32'
    ? [
        ['py', ['-3', ...args]],
        ['python', args],
      ]
    : [
        ['python3', args],
        ['python', args],
      ];

for (const [cmd, cmdArgs] of candidates) {
  const result = spawnSync(cmd, cmdArgs, { stdio: 'inherit' });
  if (result.error && result.error.code === 'ENOENT') {
    continue;
  }
  if (typeof result.status === 'number') {
    process.exit(result.status);
  }
}

console.error('Unable to run Python. Install Python and ensure it is on PATH.');
process.exit(1);
