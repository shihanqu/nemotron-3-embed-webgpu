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
const windowsFast150 = JSON.parse(readFileSync(resolve('docs/benchmarks/2026-07-22-windows-rtx3070-fast150.json'), 'utf8')) as {
  method: {
    executionProfile: string;
    integrationUrl: string;
    kernelProfile: string;
  };
  webgpu: {
    singleRps: number;
    concurrency16AggregateRps: number;
    lmStudioCosine: number;
    worstBatchCosine: number;
  };
  lmStudio: { singleRps: number; concurrency16AggregateRps: number };
  qualityProfileCheck: { lmStudioCosine: number };
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

describe('Windows RTX 3070 fast-150 acceptance', () => {
  it('records the explicit integration profiles', () => {
    expect(windowsFast150.method.kernelProfile).toContain('nvidia-rtx30');
    expect(windowsFast150.method.executionProfile).toBe('fast-150');
    expect(windowsFast150.method.integrationUrl).toContain('kernels=nvidia-rtx30');
    expect(windowsFast150.method.integrationUrl).toContain('execution=fast-150');
  });

  it('clears the requested LM Studio speedups and accuracy gates', () => {
    expect(windowsFast150.webgpu.singleRps).toBeGreaterThanOrEqual(windowsFast150.lmStudio.singleRps * 1.3);
    expect(windowsFast150.webgpu.concurrency16AggregateRps).toBeGreaterThanOrEqual(
      windowsFast150.lmStudio.concurrency16AggregateRps * 3,
    );
    expect(windowsFast150.webgpu.lmStudioCosine).toBeGreaterThanOrEqual(0.90);
    expect(windowsFast150.webgpu.worstBatchCosine).toBeGreaterThanOrEqual(0.999);
  });

  it('keeps the default profile above the natural-prose quality floor', () => {
    expect(windowsFast150.qualityProfileCheck.lmStudioCosine).toBeGreaterThanOrEqual(0.90);
  });
});
