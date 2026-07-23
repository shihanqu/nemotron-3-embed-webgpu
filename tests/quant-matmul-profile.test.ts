import { describe, expect, it } from 'vitest';
import {
  selectCompactMatmulPipeline,
  type CompactMatmulPipelineKind,
  type CompactMatmulProfile,
} from '../src/webgpu/quant-matmul.ts';

const rows = [16, 17, 31, 32, 33, 127, 128] as const;

const expected: Record<
  CompactMatmulProfile,
  Record<2048 | 4096, CompactMatmulPipelineKind[]>
> = {
  portable: {
    2048: ['latency', 'throughput', 'throughput', 'throughput', 'subgroup', 'subgroup', 'subgroup'],
    4096: [
      'compact-subgroup-16',
      'compact-subgroup-16',
      'compact-subgroup-16',
      'compact-subgroup-16',
      'subgroup',
      'subgroup',
      'subgroup',
    ],
  },
  'nvidia-rtx30': {
    2048: ['latency', 'throughput', 'throughput', 'compact-tiny', 'compact-small', 'compact-small', 'compact-wide'],
    4096: [
      'compact-subgroup-16',
      'compact-subgroup-16',
      'compact-subgroup-16',
      'compact-tiny',
      'compact-small',
      'compact-small',
      'compact-wide',
    ],
  },
};

describe.each(['portable', 'nvidia-rtx30'] as const)('%s compact matmul policy', (profile) => {
  it.each([2048, 4096] as const)('selects the expected boundary kernels at output width %i', (width) => {
    expect(rows.map((m) => selectCompactMatmulPipeline(m, width, profile))).toEqual(expected[profile][width]);
  });
});
