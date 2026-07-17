import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { GGUFReader, NEMOTRON3_METADATA_KEYS } from '../src/gguf/reader.ts';
import { requantizeMatrixToQ40 } from '../src/gguf/quantization.ts';
import { GGMLType, type GGUFMetadataValue } from '../src/gguf/types.ts';
import { buildPrepackedFile, packQ40Tile32, rawTensor, type PrepackedMetadataValue, type Q40TensorPart } from '../src/prepacked/format.ts';

const sourcePath = process.argv[2] ?? 'models/nemotron-3-embed-1b-q4_k_m.gguf';
const outputPath = process.argv[3] ?? 'models/nemotron-3-embed-1b-q4-webgpu.wgpack';
const sourceFile = await readFile(sourcePath);
const sourceBuffer = sourceFile.buffer.slice(sourceFile.byteOffset, sourceFile.byteOffset + sourceFile.byteLength) as ArrayBuffer;
const gguf = new GGUFReader(sourceBuffer).parse({ metadataKeys: NEMOTRON3_METADATA_KEYS });
if (gguf.metadata.get('general.architecture') !== 'mistral3') throw new Error('expected a mistral3 GGUF');
const sourceSha256 = createHash('sha256').update(sourceFile).digest('hex');
const consumed = new Set<string>();

function jsonMetadata(value: GGUFMetadataValue): PrepackedMetadataValue {
  const convert = (item: string | number | bigint | boolean): string | number | boolean => {
    if (typeof item !== 'bigint') return item;
    const number = Number(item);
    if (!Number.isSafeInteger(number)) throw new Error(`metadata bigint ${item} is not safely representable`);
    return number;
  };
  return Array.isArray(value) ? value.map(convert) : convert(value);
}

function part(name: string): Q40TensorPart {
  const tensor = gguf.tensors.get(name);
  if (!tensor || ![GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q6_K].includes(tensor.type) || tensor.dimensions.length !== 2) throw new Error(`${name} is not a supported quantized matrix`);
  consumed.add(name);
  const [k, n] = tensor.dimensions;
  const source = new Uint8Array(sourceBuffer, tensor.byteOffset, tensor.byteLength);
  const bytes = requantizeMatrixToQ40(source, tensor.type as 2 | 12 | 14, k, n);
  return { name, k, n, bytes };
}

const packed = [];
for (let layer = 0; layer < 16; layer += 1) {
  const prefix = `blk.${layer}`;
  packed.push(packQ40Tile32(`${prefix}.attn_qkv.weight`, [part(`${prefix}.attn_q.weight`), part(`${prefix}.attn_k.weight`), part(`${prefix}.attn_v.weight`)]));
  packed.push(packQ40Tile32(`${prefix}.attn_output.weight`, [part(`${prefix}.attn_output.weight`)]));
  packed.push(packQ40Tile32(`${prefix}.ffn_gate_up.weight`, [part(`${prefix}.ffn_gate.weight`), part(`${prefix}.ffn_up.weight`)]));
  packed.push(packQ40Tile32(`${prefix}.ffn_down.weight`, [part(`${prefix}.ffn_down.weight`)]));
}

const unexpectedMatrices = Array.from(gguf.tensors.values()).filter((tensor) => tensor.dimensions.length === 2 && tensor.name !== 'token_embd.weight' && !consumed.has(tensor.name));
if (unexpectedMatrices.length > 0) throw new Error(`unpacked matrices: ${unexpectedMatrices.map((tensor) => tensor.name).join(', ')}`);

const raw = Array.from(gguf.tensors.values())
  .filter((tensor) => !consumed.has(tensor.name))
  .map((tensor) => rawTensor(
    tensor.name,
    tensor.dimensions,
    tensor.type,
    new Uint8Array(sourceBuffer, tensor.byteOffset, tensor.byteLength),
  ));
const metadata = Object.fromEntries(Array.from(gguf.metadata, ([key, value]) => [key, jsonMetadata(value)]));
const output = buildPrepackedFile(sourceSha256, sourceFile.byteLength, metadata, [...raw, ...packed]);
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, output);
console.log(JSON.stringify({ sourcePath, outputPath, sourceSha256, rawTensors: raw.length, packedTensors: packed.length, tensors: raw.length + packed.length, bytes: output.byteLength }, null, 2));
