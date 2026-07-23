import { GGMLType } from '../gguf/types.ts';
import { createBufferWithData } from './device.ts';

const TILE_M = 16;
const TILE_N = 16;
const TILE_K = 256;
const SUBGROUP_ROWS = 32;

const shaderPrelude = /* wgsl */`
enable f16;
enable subgroups;

struct Params {
  m: u32,
  n: u32,
  k: u32,
  blocks_per_row: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(3) var<uniform> params: Params;

fn byte_at(offset: u32) -> u32 {
  let word = weights[offset >> 2u];
  return (word >> ((offset & 3u) * 8u)) & 255u;
}

fn half_at(offset: u32) -> f32 {
  let bits = byte_at(offset) | (byte_at(offset + 1u) << 8u);
  return unpack2x16float(bits).x;
}

`;

const q4Dequant = /* wgsl */`
fn dequant(row: u32, column: u32) -> f16 {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 144u;
  let within = column & 255u;
  let group = within / 64u;
  let in_group = within & 63u;
  let scale_index = group * 2u + select(0u, 1u, in_group >= 32u);
  let scale_base = block + 4u;
  var scale: u32;
  var minimum: u32;
  if (scale_index < 4u) {
    scale = byte_at(scale_base + scale_index) & 63u;
    minimum = byte_at(scale_base + scale_index + 4u) & 63u;
  } else {
    scale = (byte_at(scale_base + scale_index + 4u) & 15u) |
      ((byte_at(scale_base + scale_index - 4u) >> 6u) << 4u);
    minimum = (byte_at(scale_base + scale_index + 4u) >> 4u) |
      ((byte_at(scale_base + scale_index) >> 6u) << 4u);
  }
  let packed = byte_at(block + 16u + group * 32u + (in_group & 31u));
  let quant = select(packed & 15u, packed >> 4u, in_group >= 32u);
  return f16(half_at(block) * f32(scale) * f32(quant) - half_at(block + 2u) * f32(minimum));
}

fn dequant4(row: u32, column: u32) -> vec4<f16> {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 144u;
  let within = column & 255u;
  let group = within / 64u;
  let in_group = within & 63u;
  let scale_index = group * 2u + select(0u, 1u, in_group >= 32u);
  let scale_base = block + 4u;
  var scale: u32;
  var minimum: u32;
  if (scale_index < 4u) {
    scale = byte_at(scale_base + scale_index) & 63u;
    minimum = byte_at(scale_base + scale_index + 4u) & 63u;
  } else {
    scale = (byte_at(scale_base + scale_index + 4u) & 15u) |
      ((byte_at(scale_base + scale_index - 4u) >> 6u) << 4u);
    minimum = (byte_at(scale_base + scale_index + 4u) >> 4u) |
      ((byte_at(scale_base + scale_index) >> 6u) << 4u);
  }
  let quant_word = weights[(block + 16u + group * 32u + (in_group & 31u)) >> 2u];
  let packed = unpack4xU8(quant_word);
  let quants = select(packed & vec4<u32>(15u), packed >> vec4<u32>(4u), vec4<bool>(in_group >= 32u));
  let values = half_at(block) * f32(scale) * vec4<f32>(quants) - half_at(block + 2u) * f32(minimum);
  return vec4<f16>(values);
}
`;

const q40Dequant = /* wgsl */`
fn dequant4(row:u32,column:u32)->vec4<f16>{
  let block=(row*params.blocks_per_row+column/32u)*18u;let within=column&31u;
  let qoffset=block+2u+(within&15u);
  let packed=vec4<u32>(byte_at(qoffset),byte_at(qoffset+1u),byte_at(qoffset+2u),byte_at(qoffset+3u));
  let quants=select(packed&vec4<u32>(15u),packed>>vec4<u32>(4u),vec4<bool>(within>=16u));
  return vec4<f16>(half_at(block)*(vec4<f32>(quants)-vec4<f32>(8.0)));
}
`;

const q40PrepackedDequant = /* wgsl */`
fn dequant4(row:u32,column:u32)->vec4<f16>{
  let tile=row/32u;let local_row=row&31u;let k_block=column/32u;
  let block=((tile*params.blocks_per_row+k_block)*32u+local_row)*5u;
  let scale=f16(unpack2x16float(weights[block]).x);
  let packed=unpack4xU8(weights[block+1u+(column&15u)/4u]);
  let quants=select(packed&vec4<u32>(15u),packed>>vec4<u32>(4u),vec4<bool>((column&31u)>=16u));
  return vec4<f16>(scale)*(vec4<f16>(quants)-vec4<f16>(8.0));
}
`;

const q6Dequant = /* wgsl */`
fn dequant(row: u32, column: u32) -> f16 {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 210u;
  let within = column & 255u;
  let half = within / 128u;
  let position = within & 127u;
  let quadrant = position / 32u;
  let lane = position & 31u;
  let low_byte = byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u);
  let low = select(low_byte & 15u, low_byte >> 4u, quadrant >= 2u);
  let high_byte = byte_at(block + 128u + half * 32u + lane);
  let high = (high_byte >> (quadrant * 2u)) & 3u;
  let quant = i32(low | (high << 4u)) - 32;
  let scale_byte = byte_at(block + 192u + half * 8u + (lane / 16u) + quadrant * 2u);
  let scale = select(i32(scale_byte), i32(scale_byte) - 256, scale_byte >= 128u);
  return f16(half_at(block + 208u) * f32(scale * quant));
}

fn dequant4(row: u32, column: u32) -> vec4<f16> {
  let block_index = row * params.blocks_per_row + column / 256u;
  let block = block_index * 210u;
  let within = column & 255u;
  let half = within / 128u;
  let position = within & 127u;
  let quadrant = position / 32u;
  let lane = position & 31u;
  let low_bytes = vec4<u32>(
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u),
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u + 1u),
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u + 2u),
    byte_at(block + half * 64u + lane + (quadrant & 1u) * 32u + 3u)
  );
  let lows = select(low_bytes & vec4<u32>(15u), low_bytes >> vec4<u32>(4u), vec4<bool>(quadrant >= 2u));
  let high_bytes = vec4<u32>(
    byte_at(block + 128u + half * 32u + lane), byte_at(block + 128u + half * 32u + lane + 1u),
    byte_at(block + 128u + half * 32u + lane + 2u), byte_at(block + 128u + half * 32u + lane + 3u)
  );
  let highs = (high_bytes >> vec4<u32>(quadrant * 2u)) & vec4<u32>(3u);
  let quants = vec4<i32>(lows | (highs << vec4<u32>(4u))) - vec4<i32>(32);
  let scale_byte = byte_at(block + 192u + half * 8u + (lane / 16u) + quadrant * 2u);
  let scale = select(i32(scale_byte), i32(scale_byte) - 256, scale_byte >= 128u);
  return vec4<f16>(half_at(block + 208u) * f32(scale) * vec4<f32>(quants));
}
`;

function createShaderMain(tileK: number, halfAccumulation = false, fullOutputTiles = false): string {
  const vectorK = tileK / 4;
  const tileElements = 16 * vectorK;
  const accumulatorType = halfAccumulation ? 'f16' : 'f32';
  const dotExpression = (left: string, right: string) => halfAccumulation ? `dot(${left}, ${right})` : `dot(vec4<f32>(${left}), vec4<f32>(${right}))`;
  const weightLoad = fullOutputTiles ? 'dequant4(weight_row, base_k + tile_k)' : 'select(vec4<f16>(0.0), dequant4(weight_row, base_k + tile_k), weight_row < params.n)';
  const computeCondition = fullOutputTiles ? 'global_m0 < params.m' : 'global_m0 < params.m && global_n0 < params.n';
  const outputStores = fullOutputTiles
    ? `if (global_m0 < params.m) { output[global_m0 * params.n + global_n0] = f16(sums.x); output[global_m0 * params.n + global_n1] = f16(sums.y); }
  if (global_m1 < params.m) { output[global_m1 * params.n + global_n0] = f16(sums.z); output[global_m1 * params.n + global_n1] = f16(sums.w); }`
    : `if (global_m0 < params.m && global_n0 < params.n) { output[global_m0 * params.n + global_n0] = f16(sums.x); }
  if (global_m0 < params.m && global_n1 < params.n) { output[global_m0 * params.n + global_n1] = f16(sums.y); }
  if (global_m1 < params.m && global_n0 < params.n) { output[global_m1 * params.n + global_n0] = f16(sums.z); }
  if (global_m1 < params.m && global_n1 < params.n) { output[global_m1 * params.n + global_n1] = f16(sums.w); }`;
  const accumulation = `sums += vec4<${accumulatorType}>(
          ${dotExpression('av0', 'wv0')}, ${dotExpression('av0', 'wv1')},
          ${dotExpression('av1', 'wv0')}, ${dotExpression('av1', 'wv1')}
        );`;
  return /* wgsl */`
var<workgroup> tile_a: array<vec4<f16>, ${tileElements}>;
var<workgroup> tile_w: array<vec4<f16>, ${tileElements}>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  let global_m0 = workgroup_id.y * 16u + local_id.y;
  let global_m1 = global_m0 + 8u;
  let global_n0 = workgroup_id.x * 16u + local_id.x;
  let global_n1 = global_n0 + 8u;
  var sums = vec4<${accumulatorType}>(0.0);
  var base_k = 0u;
  loop {
    if (base_k >= params.k) { break; }
    var index = local_index;
    loop {
      if (index >= ${tileElements}u) { break; }
      let tile_row = index / ${vectorK}u;
      let tile_k = (index % ${vectorK}u) * 4u;
      let input_row = workgroup_id.y * 16u + tile_row;
      let weight_row = workgroup_id.x * 16u + tile_row;
      tile_a[index] = select(vec4<f16>(0.0), input[(input_row * params.k + base_k + tile_k) / 4u], input_row < params.m);
      tile_w[index] = ${weightLoad};
      index += 64u;
    }
    workgroupBarrier();
    if (${computeCondition}) {
      let a0 = local_id.y * ${vectorK}u;
      let a1 = (local_id.y + 8u) * ${vectorK}u;
      let w0 = local_id.x * ${vectorK}u;
      let w1 = (local_id.x + 8u) * ${vectorK}u;
      for (var k = 0u; k < ${vectorK}u; k += 1u) {
        let av0 = tile_a[a0 + k];
        let av1 = tile_a[a1 + k];
        let wv0 = tile_w[w0 + k];
        let wv1 = tile_w[w1 + k];
        ${accumulation}
      }
    }
    workgroupBarrier();
    base_k += ${tileK}u;
  }
  ${outputStores}
}
`;
}

function createBatchShaderMain(): string {
  return /* wgsl */`
var<workgroup> tile_a: array<vec4<f16>, 512>;
var<workgroup> tile_w: array<vec4<f16>, 512>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let m0=group.y*32u+local.y; let n0=group.x*32u+local.x;
  var sums0=vec4<f16>(0.0); var sums1=vec4<f16>(0.0); var sums2=vec4<f16>(0.0); var sums3=vec4<f16>(0.0);
  for (var base_k=0u; base_k<params.k; base_k+=64u) {
    var index=lane;
    loop {
      if (index>=1024u) { break; }
      let is_weight=index>=512u; let tile_index=index%512u; let row=tile_index/16u; let k4=tile_index%16u;
      if (!is_weight) {
        let global_m=group.y*32u+row;
        tile_a[tile_index]=select(vec4<f16>(0.0),input[(global_m*params.k+base_k+k4*4u)/4u],global_m<params.m);
      } else {
        let global_n=group.x*32u+row;
        tile_w[tile_index]=select(vec4<f16>(0.0),dequant4(global_n,base_k+k4*4u),global_n<params.n);
      }
      index+=64u;
    }
    workgroupBarrier();
    let a0=local.y*16u; let a1=(local.y+8u)*16u; let a2=(local.y+16u)*16u; let a3=(local.y+24u)*16u;
    let w0=local.x*16u; let w1=(local.x+8u)*16u; let w2=(local.x+16u)*16u; let w3=(local.x+24u)*16u;
    for (var k4=0u;k4<16u;k4+=1u) {
      let b0=tile_w[w0+k4]; let b1=tile_w[w1+k4]; let b2=tile_w[w2+k4]; let b3=tile_w[w3+k4];
      let av0=tile_a[a0+k4]; let av1=tile_a[a1+k4]; let av2=tile_a[a2+k4]; let av3=tile_a[a3+k4];
      sums0+=vec4<f16>(dot(av0,b0),dot(av0,b1),dot(av0,b2),dot(av0,b3));
      sums1+=vec4<f16>(dot(av1,b0),dot(av1,b1),dot(av1,b2),dot(av1,b3));
      sums2+=vec4<f16>(dot(av2,b0),dot(av2,b1),dot(av2,b2),dot(av2,b3));
      sums3+=vec4<f16>(dot(av3,b0),dot(av3,b1),dot(av3,b2),dot(av3,b3));
    }
    workgroupBarrier();
  }
  let rows=array<u32,4>(m0,m0+8u,m0+16u,m0+24u); let cols=array<u32,4>(n0,n0+8u,n0+16u,n0+24u);
  let all_sums=array<vec4<f16>,4>(sums0,sums1,sums2,sums3);
  for(var r=0u;r<4u;r+=1u) { for(var c=0u;c<4u;c+=1u) { if(rows[r]<params.m&&cols[c]<params.n) { output[rows[r]*params.n+cols[c]]=f16(all_sums[r][c]); } } }
}
`;
}

function createCompactWideBatchShaderMain(): string {
  const sumDeclarations = Array.from({ length: 8 }, (_, pair) => `var sums${pair}=vec4<f16>(0.0);`).join('');
  const accumulations = Array.from({ length: 8 }, (_, pair) => {
    const firstRow = pair * 32;
    const secondRow = firstRow + 16;
    return `let av${pair}a=tile_a[(local.y+${firstRow}u)*8u+k4];let av${pair}b=tile_a[(local.y+${secondRow}u)*8u+k4];
      sums${pair}+=vec4<f16>(dot(av${pair}a,b0),dot(av${pair}a,b1),dot(av${pair}b,b0),dot(av${pair}b,b1));`;
  }).join('\n      ');
  const stores = Array.from({ length: 8 }, (_, pair) => {
    const firstRow = pair * 32;
    const secondRow = firstRow + 16;
    return `let row${pair}a=m0+${firstRow}u;let row${pair}b=m0+${secondRow}u;
  if(row${pair}a<params.m&&n0<params.n){output[row${pair}a*params.n+n0]=sums${pair}.x;}
  if(row${pair}a<params.m&&n1<params.n){output[row${pair}a*params.n+n1]=sums${pair}.y;}
  if(row${pair}b<params.m&&n0<params.n){output[row${pair}b*params.n+n0]=sums${pair}.z;}
  if(row${pair}b<params.m&&n1<params.n){output[row${pair}b*params.n+n1]=sums${pair}.w;}`;
  }).join('\n  ');
  return /* wgsl */`
var<workgroup> tile_a:array<vec4<f16>,2048>;
var<workgroup> tile_w:array<vec4<f16>,256>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let m0=group.y*256u+local.y;let n0=group.x*32u+local.x;let n1=n0+16u;
  ${sumDeclarations}
  for(var k_block=0u;k_block<params.blocks_per_row;k_block+=1u){
    for(var index=lane;index<2304u;index+=256u){
      let is_weight=index>=2048u;let tile_index=select(index,index-2048u,is_weight);let row=tile_index/8u;let vector=tile_index&7u;
      if(is_weight){
        let global_n=group.x*32u+row;let output_tile=global_n/32u;let local_row=global_n&31u;
        let record=((output_tile*params.blocks_per_row+k_block)*32u+local_row)*5u;
        let scale=f16(unpack2x16float(weights[record]).x);
        let word=vector&3u;let packed=unpack4xU8(weights[record+1u+word]);
        let quant=select(packed&vec4<u32>(15u),packed>>vec4<u32>(4u),vector>=4u);
        tile_w[tile_index]=vec4<f16>(scale)*(vec4<f16>(quant)-vec4<f16>(8.0));
      }else{
        let global_m=group.y*256u+row;
        if(global_m<params.m){tile_a[tile_index]=input[(global_m*params.k+k_block*32u+vector*4u)/4u];}
        else{tile_a[tile_index]=vec4<f16>(0.0);}
      }
    }
    workgroupBarrier();
    let w0=local.x*8u;let w1=(local.x+16u)*8u;
    for(var k4=0u;k4<8u;k4+=1u){
      let b0=tile_w[w0+k4];let b1=tile_w[w1+k4];
      ${accumulations}
    }
    workgroupBarrier();
  }
  ${stores}
}
`;
}

function createCompactBatchShaderMain(): string {
  return /* wgsl */`
var<workgroup> tile_a:array<vec4<f16>,256>;
var<workgroup> tile_w:array<vec4<f16>,256>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let m0=group.y*32u+local.y;let n0=group.x*32u+local.x;
  var sums0=vec4<f16>(0.0);var sums1=vec4<f16>(0.0);var sums2=vec4<f16>(0.0);var sums3=vec4<f16>(0.0);
  for(var k_block=0u;k_block<params.blocks_per_row;k_block+=1u){
    if(lane<32u){
      let row=lane;let record=((group.x*params.blocks_per_row+k_block)*32u+row)*5u;
      let scale=f16(unpack2x16float(weights[record]).x);
      for(var word=0u;word<4u;word+=1u){
        let packed=unpack4xU8(weights[record+1u+word]);
        tile_w[row*8u+word]=vec4<f16>(scale)*(vec4<f16>(packed&vec4<u32>(15u))-vec4<f16>(8.0));
        tile_w[row*8u+word+4u]=vec4<f16>(scale)*(vec4<f16>(packed>>vec4<u32>(4u))-vec4<f16>(8.0));
      }
    }else{
      let row=lane-32u;let global_m=group.y*32u+row;
      for(var vector=0u;vector<8u;vector+=1u){
        tile_a[row*8u+vector]=select(vec4<f16>(0.0),input[(global_m*params.k+k_block*32u+vector*4u)/4u],global_m<params.m);
      }
    }
    workgroupBarrier();
    let a0=local.y*8u;let a1=(local.y+8u)*8u;let a2=(local.y+16u)*8u;let a3=(local.y+24u)*8u;
    let w0=local.x*8u;let w1=(local.x+8u)*8u;let w2=(local.x+16u)*8u;let w3=(local.x+24u)*8u;
    for(var k4=0u;k4<8u;k4+=1u){
      let b0=tile_w[w0+k4];let b1=tile_w[w1+k4];let b2=tile_w[w2+k4];let b3=tile_w[w3+k4];
      let av0=tile_a[a0+k4];let av1=tile_a[a1+k4];let av2=tile_a[a2+k4];let av3=tile_a[a3+k4];
      sums0+=vec4<f16>(dot(av0,b0),dot(av0,b1),dot(av0,b2),dot(av0,b3));
      sums1+=vec4<f16>(dot(av1,b0),dot(av1,b1),dot(av1,b2),dot(av1,b3));
      sums2+=vec4<f16>(dot(av2,b0),dot(av2,b1),dot(av2,b2),dot(av2,b3));
      sums3+=vec4<f16>(dot(av3,b0),dot(av3,b1),dot(av3,b2),dot(av3,b3));
    }
    workgroupBarrier();
  }
  let rows=array<u32,4>(m0,m0+8u,m0+16u,m0+24u);let cols=array<u32,4>(n0,n0+8u,n0+16u,n0+24u);
  let all_sums=array<vec4<f16>,4>(sums0,sums1,sums2,sums3);
  for(var r=0u;r<4u;r+=1u){for(var c=0u;c<4u;c+=1u){if(rows[r]<params.m&&cols[c]<params.n){output[rows[r]*params.n+cols[c]]=all_sums[r][c];}}}
}
`;
}

function createCompactSmallBatchShaderMain(): string {
  return /* wgsl */`
var<workgroup> tile_a:array<vec4<f16>,512>;
var<workgroup> tile_w:array<vec4<f16>,512>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let m0=group.y*64u+local.y;let n0=group.x*64u+local.x;
  var sums0=vec4<f16>(0.0);var sums1=vec4<f16>(0.0);var sums2=vec4<f16>(0.0);var sums3=vec4<f16>(0.0);
  for(var k_block=0u;k_block<params.blocks_per_row;k_block+=1u){
    for(var index=lane;index<1024u;index+=256u){
      let is_weight=index>=512u;let tile_index=index&511u;let row=tile_index/8u;let vector=tile_index&7u;
      if(is_weight){
        let global_n=group.x*64u+row;let output_tile=global_n/32u;let local_row=global_n&31u;
        let record=((output_tile*params.blocks_per_row+k_block)*32u+local_row)*5u;
        let scale=f16(unpack2x16float(weights[record]).x);
        let word=vector&3u;let packed=unpack4xU8(weights[record+1u+word]);
        let quant=select(packed&vec4<u32>(15u),packed>>vec4<u32>(4u),vector>=4u);
        tile_w[tile_index]=vec4<f16>(scale)*(vec4<f16>(quant)-vec4<f16>(8.0));
      }else{
        let global_m=group.y*64u+row;
        if(global_m<params.m){tile_a[tile_index]=input[(global_m*params.k+k_block*32u+vector*4u)/4u];}
        else{tile_a[tile_index]=vec4<f16>(0.0);}
      }
    }
    workgroupBarrier();
    let a0=local.y*8u;let a1=(local.y+16u)*8u;let a2=(local.y+32u)*8u;let a3=(local.y+48u)*8u;
    let w0=local.x*8u;let w1=(local.x+16u)*8u;let w2=(local.x+32u)*8u;let w3=(local.x+48u)*8u;
    for(var k4=0u;k4<8u;k4+=1u){
      let b0=tile_w[w0+k4];let b1=tile_w[w1+k4];let b2=tile_w[w2+k4];let b3=tile_w[w3+k4];
      let av0=tile_a[a0+k4];let av1=tile_a[a1+k4];let av2=tile_a[a2+k4];let av3=tile_a[a3+k4];
      sums0+=vec4<f16>(dot(av0,b0),dot(av0,b1),dot(av0,b2),dot(av0,b3));
      sums1+=vec4<f16>(dot(av1,b0),dot(av1,b1),dot(av1,b2),dot(av1,b3));
      sums2+=vec4<f16>(dot(av2,b0),dot(av2,b1),dot(av2,b2),dot(av2,b3));
      sums3+=vec4<f16>(dot(av3,b0),dot(av3,b1),dot(av3,b2),dot(av3,b3));
    }
    workgroupBarrier();
  }
  let rows=array<u32,4>(m0,m0+16u,m0+32u,m0+48u);let cols=array<u32,4>(n0,n0+16u,n0+32u,n0+48u);
  let all_sums=array<vec4<f16>,4>(sums0,sums1,sums2,sums3);
  for(var r=0u;r<4u;r+=1u){for(var c=0u;c<4u;c+=1u){if(rows[r]<params.m&&cols[c]<params.n){output[rows[r]*params.n+cols[c]]=all_sums[r][c];}}}
}
`;
}

function createCompactTinyBatchShaderMain(): string {
  return /* wgsl */`
var<workgroup> tile_a:array<vec4<f16>,256>;
var<workgroup> tile_w:array<vec4<f16>,512>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) local:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let m0=group.y*32u+local.y;let m1=m0+16u;let n0=group.x*64u+local.x;
  var sums0=vec4<f16>(0.0);var sums1=vec4<f16>(0.0);
  for(var k_block=0u;k_block<params.blocks_per_row;k_block+=1u){
    for(var index=lane;index<768u;index+=256u){
      let is_weight=index>=256u;let tile_index=select(index,index-256u,is_weight);let row=tile_index/8u;let vector=tile_index&7u;
      if(is_weight){
        let global_n=group.x*64u+row;let output_tile=global_n/32u;let local_row=global_n&31u;
        let record=((output_tile*params.blocks_per_row+k_block)*32u+local_row)*5u;
        let scale=f16(unpack2x16float(weights[record]).x);
        let word=vector&3u;let packed=unpack4xU8(weights[record+1u+word]);
        let quant=select(packed&vec4<u32>(15u),packed>>vec4<u32>(4u),vector>=4u);
        tile_w[tile_index]=vec4<f16>(scale)*(vec4<f16>(quant)-vec4<f16>(8.0));
      }else{
        let global_m=group.y*32u+row;
        if(global_m<params.m){tile_a[tile_index]=input[(global_m*params.k+k_block*32u+vector*4u)/4u];}
        else{tile_a[tile_index]=vec4<f16>(0.0);}
      }
    }
    workgroupBarrier();
    let a0=local.y*8u;let a1=(local.y+16u)*8u;
    let w0=local.x*8u;let w1=(local.x+16u)*8u;let w2=(local.x+32u)*8u;let w3=(local.x+48u)*8u;
    for(var k4=0u;k4<8u;k4+=1u){
      let b0=tile_w[w0+k4];let b1=tile_w[w1+k4];let b2=tile_w[w2+k4];let b3=tile_w[w3+k4];
      let av0=tile_a[a0+k4];let av1=tile_a[a1+k4];
      sums0+=vec4<f16>(dot(av0,b0),dot(av0,b1),dot(av0,b2),dot(av0,b3));
      sums1+=vec4<f16>(dot(av1,b0),dot(av1,b1),dot(av1,b2),dot(av1,b3));
    }
    workgroupBarrier();
  }
  let cols=array<u32,4>(n0,n0+16u,n0+32u,n0+48u);
  for(var c=0u;c<4u;c+=1u){
    if(m0<params.m&&cols[c]<params.n){output[m0*params.n+cols[c]]=sums0[c];}
    if(m1<params.m&&cols[c]<params.n){output[m1*params.n+cols[c]]=sums1[c];}
  }
}
`;
}

function createMidBatchShaderMain(): string {
  return /* wgsl */`
var<workgroup> tile_a: array<vec4<f16>, 256>;
var<workgroup> tile_w: array<vec4<f16>, 512>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let m0=group.y*16u+local.y; let m1=m0+8u;
  let n0=group.x*32u+local.x;
  var sums0=vec4<f16>(0.0); var sums1=vec4<f16>(0.0);
  for (var base_k=0u; base_k<params.k; base_k+=64u) {
    var index=lane;
    loop {
      if (index>=768u) { break; }
      let is_weight=index>=256u;
      let tile_index=select(index,index-256u,is_weight);
      let row=tile_index/16u; let k4=tile_index%16u;
      if (!is_weight) {
        let global_m=group.y*16u+row;
        tile_a[tile_index]=select(vec4<f16>(0.0),input[(global_m*params.k+base_k+k4*4u)/4u],global_m<params.m);
      } else {
        let global_n=group.x*32u+row;
        tile_w[tile_index]=select(vec4<f16>(0.0),dequant4(global_n,base_k+k4*4u),global_n<params.n);
      }
      index+=64u;
    }
    workgroupBarrier();
    let a0=local.y*16u; let a1=(local.y+8u)*16u;
    let w0=local.x*16u; let w1=(local.x+8u)*16u; let w2=(local.x+16u)*16u; let w3=(local.x+24u)*16u;
    for (var k4=0u;k4<16u;k4+=1u) {
      let av0=tile_a[a0+k4]; let av1=tile_a[a1+k4];
      let b0=tile_w[w0+k4]; let b1=tile_w[w1+k4]; let b2=tile_w[w2+k4]; let b3=tile_w[w3+k4];
      sums0+=vec4<f16>(dot(av0,b0),dot(av0,b1),dot(av0,b2),dot(av0,b3));
      sums1+=vec4<f16>(dot(av1,b0),dot(av1,b1),dot(av1,b2),dot(av1,b3));
    }
    workgroupBarrier();
  }
  let cols=array<u32,4>(n0,n0+8u,n0+16u,n0+24u);
  for(var c=0u;c<4u;c+=1u) {
    if(m0<params.m&&cols[c]<params.n) { output[m0*params.n+cols[c]]=sums0[c]; }
    if(m1<params.m&&cols[c]<params.n) { output[m1*params.n+cols[c]]=sums1[c]; }
  }
}
`;
}

function createSubgroupShaderMain(): string {
  const sumDeclarations = Array.from({ length: SUBGROUP_ROWS }, (_, row) => `var sum${row}=f16(0.0);`).join('\n  ');
  const accumulations = Array.from({ length: SUBGROUP_ROWS }, (_, row) => `sum${row}+=dot(subgroupBroadcast(own_activation,${row}u),weight);`).join('\n    ');
  const stores = Array.from({ length: SUBGROUP_ROWS }, (_, row) => `if(m_base+${row}u<params.m){output[(m_base+${row}u)*params.n+n]=sum${row};}`).join('\n    ');
  return /* wgsl */`
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n=group.x*32u+lane;
  let m_base=group.y*${SUBGROUP_ROWS}u;
  ${sumDeclarations}
  for(var column=0u;column<params.k;column+=4u){
    var weight=vec4<f16>(0.0);
    if(n<params.n){weight=dequant4(n,column);}
    var own_activation=vec4<f16>(0.0);
    if(lane<${SUBGROUP_ROWS}u&&m_base+lane<params.m){own_activation=input[((m_base+lane)*params.k+column)/4u];}
    ${accumulations}
  }
  if(n<params.n){
    ${stores}
  }
}
`;
}

function createCompactSubgroupShaderMain(rowsPerGroup: 16 | 32 | 64): string {
  const activationGroups = Math.ceil(rowsPerGroup / 32);
  const sumDeclarations = Array.from({ length: rowsPerGroup }, (_, row) => `var sum${row}=f16(0.0);`).join('\n  ');
  const activationDeclarations = Array.from({ length: activationGroups }, (_, group) =>
    `var own_activation_low${group}=vec4<f16>(0.0);var own_activation_high${group}=vec4<f16>(0.0);`,
  ).join('\n      ');
  const activationLoads = Array.from({ length: activationGroups }, (_, group) => {
    const rowOffset = group * 32;
    return `if(m_base+${rowOffset}u+lane<params.m){
        own_activation_low${group}=input[((m_base+${rowOffset}u+lane)*params.k+column_low)/4u];
        own_activation_high${group}=input[((m_base+${rowOffset}u+lane)*params.k+column_high)/4u];
      }`;
  }).join('\n      ');
  const lowAccumulations = Array.from({ length: rowsPerGroup }, (_, row) => {
    const group = Math.floor(row / 32);
    return `sum${row}+=dot(subgroupBroadcast(own_activation_low${group},${row % 32}u),weight_low);`;
  }).join('\n      ');
  const highAccumulations = Array.from({ length: rowsPerGroup }, (_, row) => {
    const group = Math.floor(row / 32);
    return `sum${row}+=dot(subgroupBroadcast(own_activation_high${group},${row % 32}u),weight_high);`;
  }).join('\n      ');
  const stores = Array.from({ length: rowsPerGroup }, (_, row) => `if(m_base+${row}u<params.m){output[(m_base+${row}u)*params.n+n]=sum${row};}`).join('\n    ');
  return /* wgsl */`
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group: vec3<u32>,@builtin(subgroup_invocation_id) lane:u32){
  let n=group.x*32u+lane;let m_base=group.y*${rowsPerGroup}u;
  ${sumDeclarations}
  for(var k_block=0u;k_block<params.blocks_per_row;k_block+=1u){
    let record=((group.x*params.blocks_per_row+k_block)*32u+lane)*5u;
    let scale=f16(unpack2x16float(weights[record]).x);
    for(var word=0u;word<4u;word+=1u){
      let packed=unpack4xU8(weights[record+1u+word]);
      let weight_low=vec4<f16>(scale)*(vec4<f16>(packed&vec4<u32>(15u))-vec4<f16>(8.0));
      let weight_high=vec4<f16>(scale)*(vec4<f16>(packed>>vec4<u32>(4u))-vec4<f16>(8.0));
      let column_low=k_block*32u+word*4u;let column_high=column_low+16u;
      ${activationDeclarations}
      ${activationLoads}
      ${lowAccumulations}
      ${highAccumulations}
    }
  }
  if(n<params.n){${stores}}
}
`;
}

export interface QuantMatmulRun {
  input: GPUBuffer;
  weights: GPUBuffer;
  output: GPUBuffer;
  bindGroup: GPUBindGroup;
  pipeline: GPUComputePipeline;
  m: number;
  n: number;
  k: number;
}

export type QuantWeightLayout = 'gguf' | 'q4_0-tile32-compact';
export type CompactMatmulProfile = 'portable' | 'nvidia-rtx30';
export type CompactMatmulPipelineKind =
  | 'compact-subgroup-16'
  | 'latency'
  | 'throughput'
  | 'subgroup'
  | 'compact-tiny'
  | 'compact-small'
  | 'compact-wide';

export function selectCompactMatmulPipeline(
  m: number,
  n: number,
  profile: CompactMatmulProfile = 'portable',
): CompactMatmulPipelineKind {
  if (profile === 'portable') {
    if (m <= 32 && n >= 4096) return 'compact-subgroup-16';
    if (m <= 16) return 'latency';
    if (m <= 32) return 'throughput';
    return 'subgroup';
  }
  if (m <= 16 && n >= 4096) return 'compact-subgroup-16';
  if (m <= 16) return 'latency';
  if (m < 32 && n >= 4096) return 'compact-subgroup-16';
  if (m < 32) return 'throughput';
  if (m === 32) return 'compact-tiny';
  if (m < 128) return 'compact-small';
  return 'compact-wide';
}

export class QuantMatmulKernel {
  readonly latencyPipeline: GPUComputePipeline;
  readonly throughputPipeline: GPUComputePipeline;
  readonly midBatchPipeline: GPUComputePipeline;
  readonly batchPipeline: GPUComputePipeline;
  readonly compactWideBatchPipeline?: GPUComputePipeline;
  readonly compactSmallBatchPipeline?: GPUComputePipeline;
  readonly compactTinyBatchPipeline?: GPUComputePipeline;
  readonly subgroupPipeline: GPUComputePipeline;
  readonly compactSubgroup16Pipeline: GPUComputePipeline;

  constructor(
    readonly device: GPUDevice,
    readonly type: GGMLType.Q4_0 | GGMLType.Q4_K | GGMLType.Q6_K,
    readonly layout: QuantWeightLayout = 'gguf',
    readonly compactProfile: CompactMatmulProfile = 'portable',
  ) {
    if (layout === 'q4_0-tile32-compact' && type !== GGMLType.Q4_0) throw new Error('prepacked tile32 layout only supports Q4_0 sources');
    this.latencyPipeline = this.createPipeline(128, 'latency');
    this.throughputPipeline = this.createPipeline(64, 'throughput', layout === 'q4_0-tile32-compact', layout === 'q4_0-tile32-compact');
    this.midBatchPipeline = this.createMidBatchPipeline();
    this.batchPipeline = this.createBatchPipeline();
    if (layout === 'q4_0-tile32-compact' && compactProfile === 'nvidia-rtx30') {
      this.compactWideBatchPipeline = this.createCompactWideBatchPipeline();
      this.compactSmallBatchPipeline = this.createCompactSmallBatchPipeline();
      this.compactTinyBatchPipeline = this.createCompactTinyBatchPipeline();
    }
    this.subgroupPipeline = this.createSubgroupPipeline(32);
    this.compactSubgroup16Pipeline = layout === 'q4_0-tile32-compact' ? this.createSubgroupPipeline(16) : this.subgroupPipeline;
  }

  private createSubgroupPipeline(rowsPerGroup: 16 | 32): GPUComputePipeline {
    const code=this.layout==='q4_0-tile32-compact'
      ? this.shaderPrelude()+createCompactSubgroupShaderMain(rowsPerGroup)
      : this.shaderPrelude()+this.dequantShader()+createSubgroupShaderMain();
    const module=this.device.createShaderModule({code,label:`prepacked subgroup-${rowsPerGroup} matmul shader`});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:`prepacked subgroup-${rowsPerGroup} matmul`});
  }

  private createMidBatchPipeline(): GPUComputePipeline {
    const dequant=this.dequantShader();
    const code=this.shaderPrelude()+dequant+createMidBatchShaderMain();
    const quantName=this.type===GGMLType.Q4_0?'Q4_0':this.type===GGMLType.Q4_K?'Q4_K':'Q6_K';
    const module=this.device.createShaderModule({code,label:`${quantName} mid-batch shader`});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:`${quantName} fused matmul (mid-batch)`});
  }

  private createBatchPipeline(): GPUComputePipeline {
    const dequant=this.dequantShader();
    const code=this.layout==='q4_0-tile32-compact'
      ? this.shaderPrelude()+createCompactBatchShaderMain()
      : this.shaderPrelude()+dequant+createBatchShaderMain();
    const quantName=this.type===GGMLType.Q4_0?'Q4_0':this.type===GGMLType.Q4_K?'Q4_K':'Q6_K';
    const module=this.device.createShaderModule({code,label:`${quantName} batch shader`});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:`${quantName} fused matmul (batch)`});
  }

  private createCompactWideBatchPipeline(): GPUComputePipeline {
    const code=this.shaderPrelude()+createCompactWideBatchShaderMain();
    const module=this.device.createShaderModule({code,label:'Q4_0 compact 256x32 matmul shader'});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:'Q4_0 compact 256x32 matmul'});
  }

  private createCompactSmallBatchPipeline(): GPUComputePipeline {
    const code=this.shaderPrelude()+createCompactSmallBatchShaderMain();
    const module=this.device.createShaderModule({code,label:'Q4_0 compact 64x64 matmul shader'});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:'Q4_0 compact 64x64 matmul'});
  }

  private createCompactTinyBatchPipeline(): GPUComputePipeline {
    const code=this.shaderPrelude()+createCompactTinyBatchShaderMain();
    const module=this.device.createShaderModule({code,label:'Q4_0 compact 32x64 matmul shader'});
    return this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label:'Q4_0 compact 32x64 matmul'});
  }

  private createPipeline(tileK: number, variant: string, halfAccumulation = false, fullOutputTiles = false): GPUComputePipeline {
    const dequant=this.dequantShader();
    const code = this.shaderPrelude() + dequant + createShaderMain(tileK, halfAccumulation, fullOutputTiles);
    const quantName = this.type===GGMLType.Q4_0?'Q4_0':this.type === GGMLType.Q4_K ? 'Q4_K' : 'Q6_K';
    const module = this.device.createShaderModule({ code, label: `${quantName} ${variant} shader` });
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
      label: `${quantName} fused matmul (${variant})`,
    });
  }

  private dequantShader(): string {
    if (this.layout === 'q4_0-tile32-compact') return q40PrepackedDequant;
    return this.type===GGMLType.Q4_0?q40Dequant:this.type===GGMLType.Q4_K?q4Dequant:q6Dequant;
  }

  private shaderPrelude(): string {
    return shaderPrelude;
  }

  createRun(input: Uint16Array, weights: Uint8Array, m: number, n: number, k: number): QuantMatmulRun {
    if (k % TILE_K !== 0) throw new Error('quantized matmul K must be divisible by 256');
    if (input.length !== m * k) throw new Error(`input has ${input.length} values; expected ${m * k}`);
    const inputBuffer = createBufferWithData(this.device, input, GPUBufferUsage.STORAGE, 'matmul input');
    const weightBuffer = createBufferWithData(this.device, weights, GPUBufferUsage.STORAGE, 'quantized weights');
    return this.createRunFromBuffers(inputBuffer, weightBuffer, m, n, k);
  }

  createRunFromBuffers(inputBuffer: GPUBuffer, weightBuffer: GPUBuffer, m: number, n: number, k: number, outputBuffer?: GPUBuffer): QuantMatmulRun {
    if (k % TILE_K !== 0) throw new Error('quantized matmul K must be divisible by 256');
    const output = outputBuffer ?? this.device.createBuffer({
      size: Math.ceil(m * n * 2 / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'matmul output',
    });
    const params = createBufferWithData(
      this.device,
      new Uint32Array([m, n, k, k / (this.type === GGMLType.Q4_0 ? 32 : 256)]),
      GPUBufferUsage.UNIFORM,
      'matmul params',
    );
    const pipeline = this.layout === 'q4_0-tile32-compact'
      ? this.compactPipeline(selectCompactMatmulPipeline(m, n, this.compactProfile))
      : m >= 512 ? this.batchPipeline : m >= 256 ? this.midBatchPipeline : m >= 16 ? this.throughputPipeline : this.latencyPipeline;
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    return { input: inputBuffer, weights: weightBuffer, output, bindGroup, pipeline, m, n, k };
  }

  encode(pass: GPUComputePassEncoder, run: QuantMatmulRun): void {
    pass.setPipeline(run.pipeline);
    pass.setBindGroup(0, run.bindGroup);
    const batch = run.pipeline === this.batchPipeline;
    const midBatch = run.pipeline === this.midBatchPipeline;
    if (run.pipeline === this.compactSubgroup16Pipeline) {
      pass.dispatchWorkgroups(Math.ceil(run.n / 32), Math.ceil(run.m / 16));
      return;
    }
    if (run.pipeline === this.subgroupPipeline) {
      pass.dispatchWorkgroups(Math.ceil(run.n / 32), Math.ceil(run.m / 32));
      return;
    }
    if (this.compactSmallBatchPipeline && run.pipeline === this.compactSmallBatchPipeline) {
      pass.dispatchWorkgroups(Math.ceil(run.n / 64), Math.ceil(run.m / 64));
      return;
    }
    if (this.compactTinyBatchPipeline && run.pipeline === this.compactTinyBatchPipeline) {
      pass.dispatchWorkgroups(Math.ceil(run.n / 64), Math.ceil(run.m / 32));
      return;
    }
    if (this.compactWideBatchPipeline && run.pipeline === this.compactWideBatchPipeline) {
      pass.dispatchWorkgroups(Math.ceil(run.n / 32), Math.ceil(run.m / 256));
      return;
    }
    pass.dispatchWorkgroups(Math.ceil(run.n / (batch || midBatch ? 32 : TILE_N)), Math.ceil(run.m / (batch ? 32 : TILE_M)));
  }

  private compactPipeline(kind: CompactMatmulPipelineKind): GPUComputePipeline {
    if (kind === 'compact-subgroup-16') return this.compactSubgroup16Pipeline;
    if (kind === 'latency') return this.latencyPipeline;
    if (kind === 'throughput') return this.throughputPipeline;
    if (kind === 'subgroup') return this.subgroupPipeline;
    const pipeline = kind === 'compact-tiny'
      ? this.compactTinyBatchPipeline
      : kind === 'compact-small'
        ? this.compactSmallBatchPipeline
        : this.compactWideBatchPipeline;
    if (!pipeline) throw new Error(`compact matmul pipeline ${kind} requires the nvidia-rtx30 profile`);
    return pipeline;
  }

  dispatch(run: QuantMatmulRun, repetitions = 1): void {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (let index = 0; index < repetitions; index += 1) this.encode(pass, run);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  ownsPipeline(pipeline: GPUComputePipeline): boolean {
    return [
      this.latencyPipeline,
      this.throughputPipeline,
      this.midBatchPipeline,
      this.batchPipeline,
      this.compactWideBatchPipeline,
      this.compactSmallBatchPipeline,
      this.compactTinyBatchPipeline,
      this.subgroupPipeline,
      this.compactSubgroup16Pipeline,
    ].includes(pipeline);
  }
}
