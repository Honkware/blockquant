import { describe, expect, it } from 'vitest';
import { formatDuration, progressBar, truncate } from '../src/utils/format.js';

describe('format utilities', () => {
  it('formats duration in seconds, minutes, and hours', () => {
    expect(formatDuration(4000)).toBe('4s');
    expect(formatDuration(61000)).toBe('1m 1s');
    expect(formatDuration(3665000)).toBe('1h 1m 5s');
  });

  it('builds progress bar with expected width', () => {
    expect(progressBar(50, 10)).toBe('█████░░░░░ 50%');
  });

  it('truncates long strings with ellipsis', () => {
    expect(truncate('abcdef', 5)).toBe('abcd…');
    expect(truncate('abc', 5)).toBe('abc');
  });
});
