import { describe, expect, it } from 'vitest';
import { contextualMergedLength } from '../src/webgpu/model.ts';

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
});
