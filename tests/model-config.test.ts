import { describe, expect, it } from 'vitest';
import { contextualMergeAfterLayer, contextualMergedLength } from '../src/webgpu/model.ts';

describe('contextual token merge sizing', () => {
  it.each([
    [1, 1],
    [15, 10],
    [50, 32],
    [150, 96],
    [500, 320],
    [1500, 960],
    [5000, 3200],
  ])('maps %i source tokens to %i contextual states', (source, expected) => {
    expect(contextualMergedLength(source)).toBe(expected);
  });

  it('enables the 32-state fast path only for medium contexts', () => {
    expect(contextualMergedLength(127, 'fast-150')).toBe(82);
    expect(contextualMergedLength(128, 'fast-150')).toBe(32);
    expect(contextualMergedLength(150, 'fast-150')).toBe(32);
    expect(contextualMergedLength(192, 'fast-150')).toBe(32);
    expect(contextualMergedLength(193, 'fast-150')).toBe(124);
  });

  it('keeps execution profiles explicit and bounded', () => {
    expect(contextualMergeAfterLayer(150)).toBe(4);
    expect(contextualMergeAfterLayer(127, 'fast-150')).toBe(4);
    expect(contextualMergeAfterLayer(128, 'fast-150')).toBe(1);
    expect(contextualMergeAfterLayer(192, 'fast-150')).toBe(1);
    expect(contextualMergeAfterLayer(193, 'fast-150')).toBe(4);
  });
});
