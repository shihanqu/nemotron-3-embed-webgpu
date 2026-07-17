/** Convert an IEEE-754 binary16 bit-pattern to a JavaScript number. */
export function halfToFloat(bits: number): number {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function scaleAndMin(scales: Uint8Array, index: number): [number, number] {
  if (index < 4) return [scales[index] & 63, scales[index + 4] & 63];
  return [
    (scales[index + 4] & 0x0f) | ((scales[index - 4] >>> 6) << 4),
    (scales[index + 4] >>> 4) | ((scales[index] >>> 6) << 4),
  ];
}

export function dequantizeQ4KBlock(bytes: Uint8Array, byteOffset = 0, output = new Float32Array(256)): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset + byteOffset, 144);
  const d = halfToFloat(view.getUint16(0, true));
  const dmin = halfToFloat(view.getUint16(2, true));
  const scales = bytes.subarray(byteOffset + 4, byteOffset + 16);
  const quants = bytes.subarray(byteOffset + 16, byteOffset + 144);
  let scaleIndex = 0;
  for (let group = 0; group < 4; group += 1) {
    const [scaleLow, minLow] = scaleAndMin(scales, scaleIndex++);
    const [scaleHigh, minHigh] = scaleAndMin(scales, scaleIndex++);
    const quantOffset = group * 32;
    const outputOffset = group * 64;
    for (let lane = 0; lane < 32; lane += 1) {
      const packed = quants[quantOffset + lane];
      output[outputOffset + lane] = d * scaleLow * (packed & 0x0f) - dmin * minLow;
      output[outputOffset + lane + 32] = d * scaleHigh * (packed >>> 4) - dmin * minHigh;
    }
  }
  return output;
}

export function dequantizeQ40Block(bytes: Uint8Array, byteOffset = 0, output = new Float32Array(32)): Float32Array {
  const view=new DataView(bytes.buffer,bytes.byteOffset+byteOffset,18); const d=halfToFloat(view.getUint16(0,true));
  for(let lane=0;lane<16;lane+=1){const packed=bytes[byteOffset+2+lane];output[lane]=d*((packed&15)-8);output[lane+16]=d*((packed>>>4)-8);}
  return output;
}

export function dequantizeQ6KBlock(bytes: Uint8Array, byteOffset = 0, output = new Float32Array(256)): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset + byteOffset, 210);
  const ql = bytes.subarray(byteOffset, byteOffset + 128);
  const qh = bytes.subarray(byteOffset + 128, byteOffset + 192);
  const scales = new Int8Array(bytes.buffer, bytes.byteOffset + byteOffset + 192, 16);
  const d = halfToFloat(view.getUint16(208, true));

  for (let half = 0; half < 2; half += 1) {
    const lowOffset = half * 64;
    const highOffset = half * 32;
    const scaleOffset = half * 8;
    const outputOffset = half * 128;
    for (let lane = 0; lane < 32; lane += 1) {
      const scalePair = lane >>> 4;
      const high = qh[highOffset + lane];
      const lowA = ql[lowOffset + lane];
      const lowB = ql[lowOffset + lane + 32];
      const q1 = (lowA & 15) | ((high & 3) << 4);
      const q2 = (lowB & 15) | (((high >>> 2) & 3) << 4);
      const q3 = (lowA >>> 4) | (((high >>> 4) & 3) << 4);
      const q4 = (lowB >>> 4) | (((high >>> 6) & 3) << 4);
      output[outputOffset + lane] = d * scales[scaleOffset + scalePair] * (q1 - 32);
      output[outputOffset + lane + 32] = d * scales[scaleOffset + scalePair + 2] * (q2 - 32);
      output[outputOffset + lane + 64] = d * scales[scaleOffset + scalePair + 4] * (q3 - 32);
      output[outputOffset + lane + 96] = d * scales[scaleOffset + scalePair + 6] * (q4 - 32);
    }
  }
  return output;
}

/** Quantize 32 values using llama.cpp's symmetric Q4_0 convention. */
export function quantizeQ40Block(values: Float32Array, valueOffset = 0, output = new Uint8Array(18)): Uint8Array {
  let signedMaximum = 0;
  let maximumAbsolute = 0;
  for (let index = 0; index < 32; index += 1) {
    const value = values[valueOffset + index];
    const absolute = Math.abs(value);
    if (absolute > maximumAbsolute) {
      maximumAbsolute = absolute;
      signedMaximum = value;
    }
  }
  const scale = signedMaximum === 0 ? 0 : signedMaximum / -8;
  const inverseScale = scale === 0 ? 0 : 1 / scale;
  const view = new DataView(output.buffer, output.byteOffset, output.byteLength);
  view.setUint16(0, toFloat16Bits(new Float32Array([scale]))[0], true);
  for (let index = 0; index < 16; index += 1) {
    const low = Math.min(15, Math.max(0, Math.floor(values[valueOffset + index] * inverseScale + 8.5)));
    const high = Math.min(15, Math.max(0, Math.floor(values[valueOffset + index + 16] * inverseScale + 8.5)));
    output[index + 2] = low | (high << 4);
  }
  return output;
}

/** Requantize one row-major GGML matrix to Q4_0 without materializing it as F32. */
export function requantizeMatrixToQ40(
  bytes: Uint8Array,
  type: 2 | 12 | 14,
  k: number,
  n: number,
): Uint8Array {
  if (k % 256 !== 0) throw new Error(`K=${k} is not divisible by a K-quant block`);
  const sourceBlockBytes = type === 12 ? 144 : type === 14 ? 210 : 18;
  const sourceBlockValues = type === 2 ? 32 : 256;
  const expectedBytes = n * (k / sourceBlockValues) * sourceBlockBytes;
  if (bytes.byteLength !== expectedBytes) throw new Error(`source has ${bytes.byteLength} bytes; expected ${expectedBytes}`);
  const output = new Uint8Array(n * (k / 32) * 18);
  const decoded = new Float32Array(sourceBlockValues);
  const encoded = new Uint8Array(18);
  let sourceOffset = 0;
  let outputOffset = 0;
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < k; column += sourceBlockValues) {
      if (type === 12) dequantizeQ4KBlock(bytes, sourceOffset, decoded);
      else if (type === 14) dequantizeQ6KBlock(bytes, sourceOffset, decoded);
      else dequantizeQ40Block(bytes, sourceOffset, decoded);
      for (let subblock = 0; subblock < sourceBlockValues; subblock += 32) {
        quantizeQ40Block(decoded, subblock, encoded);
        output.set(encoded, outputOffset);
        outputOffset += 18;
      }
      sourceOffset += sourceBlockBytes;
    }
  }
  return output;
}
import { toFloat16Bits } from '../math/f16.ts';
