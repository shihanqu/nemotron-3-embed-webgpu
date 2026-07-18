import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

interface BenchmarkRow {
  tokens: number;
  previousWebgpu: {
    singleRps: number;
    concurrency16AggregateRps: number;
  };
  webgpu: {
    singleRps: number;
    concurrency16AggregateRps: number;
    lmStudioCosine: number;
    worstBatchCosine: number;
  };
  lmStudio: { singleRps: number; concurrency16AggregateRps: number };
}

const data = JSON.parse(readFileSync(resolve('docs/benchmarks/2026-07-17-webgpu-vs-lm-studio-m3-max.json'), 'utf8')) as {
  naturalQuality: Array<{ lmStudioCosine: number }>;
  rows: BenchmarkRow[];
};

describe('published benchmark matrix', () => {
  it('contains every required exact-token workload', () => {
    expect(data.rows.map((row) => row.tokens)).toEqual([15, 50, 150, 500, 1500, 5000]);
  });

  it.each(data.rows)('$tokens tokens beats raw LM Studio and clears accuracy gates', (row) => {
    expect(row.webgpu.singleRps).toBeGreaterThan(row.lmStudio.singleRps);
    expect(row.webgpu.concurrency16AggregateRps).toBeGreaterThan(row.lmStudio.concurrency16AggregateRps);
    expect(row.webgpu.lmStudioCosine).toBeGreaterThanOrEqual(0.90);
    expect(row.webgpu.worstBatchCosine).toBeGreaterThanOrEqual(0.999);
  });

  it.each(data.rows)('$tokens tokens is at least 30% faster than the previous WebGPU runtime', (row) => {
    expect(row.webgpu.singleRps).toBeGreaterThanOrEqual(row.previousWebgpu.singleRps * 1.3);
    expect(row.webgpu.concurrency16AggregateRps).toBeGreaterThanOrEqual(row.previousWebgpu.concurrency16AggregateRps * 1.3);
  });

  it('clears the natural-prose quality gate', () => {
    expect(data.naturalQuality).toHaveLength(2);
    for (const fixture of data.naturalQuality) {
      expect(fixture.lmStudioCosine).toBeGreaterThanOrEqual(0.90);
    }
  });
});
