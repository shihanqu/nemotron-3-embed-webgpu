import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

interface Row {
  tokens: number;
  webgpu: { singleRps: number; concurrency16AggregateRps: number };
  lmStudio: { singleRps: number; concurrency16AggregateRps: number };
}

interface BenchmarkData {
  date: string;
  testedHardware: string;
  rows: Row[];
}

const inputPath = resolve('docs/benchmarks/2026-07-17-webgpu-vs-lm-studio-m3-max.json');
const data = JSON.parse(readFileSync(inputPath, 'utf8')) as BenchmarkData;

function renderChart(options: {
  title: string;
  description: string;
  outputPath: string;
  maximum: number;
  ticks: number[];
  lmValue: (row: Row) => number;
  webgpuValue: (row: Row) => number;
  footer: string;
}): void {
  const width = 940;
  const height = 590;
  const plot = { x: 120, y: 126, width: 720, height: 360 };
  const rowHeight = 58;
  const text = (x: number, y: number, value: string, className = 'label', anchor = 'start') =>
    `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${value}</text>`;
  const grid = options.ticks.map((tick) => {
    const x = plot.x + (tick / options.maximum) * plot.width;
    return `<line x1="${x}" y1="${plot.y}" x2="${x}" y2="${plot.y + plot.height}" class="grid"/>${text(x, plot.y + plot.height + 27, String(tick), 'tick', 'middle')}`;
  }).join('');
  const marks = data.rows.map((row, index) => {
    const y = plot.y + index * rowHeight + 4;
    const lm = options.lmValue(row);
    const gpu = options.webgpuValue(row);
    const lmWidth = (lm / options.maximum) * plot.width;
    const gpuWidth = (gpu / options.maximum) * plot.width;
    return `
      ${text(plot.x - 18, y + 34, `${row.tokens.toLocaleString()} tok`, 'row-label', 'end')}
      <rect x="${plot.x}" y="${y}" width="${lmWidth}" height="20" rx="4" class="lm"/>
      <rect x="${plot.x}" y="${y + 25}" width="${gpuWidth}" height="20" rx="4" class="gpu"/>
      ${text(Math.min(plot.x + lmWidth + 8, width - 52), y + 15, lm.toFixed(2), 'value')}
      ${text(Math.min(plot.x + gpuWidth + 8, width - 52), y + 40, gpu.toFixed(2), 'value')}
    `.trim();
  }).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${options.title}</title>
  <desc id="desc">${options.description}</desc>
  <rect width="${width}" height="${height}" fill="#0d1117"/>
  <style>
    .heading { font: 600 24px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #f0f6fc; }
    .subheading { font: 600 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #f0f6fc; }
    .label, .row-label, .value, .tick, .caption { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #b1bac4; }
    .row-label { font-size: 14px; font-weight: 600; }
    .value { font-size: 13px; font-weight: 600; }
    .tick, .caption { font-size: 13px; }
    .grid { stroke: #30363d; stroke-width: 1; }
    .lm { fill: #8b949e; }
    .gpu { fill: #58a6ff; }
  </style>
  ${text(42, 45, 'Nemotron 3 Embed 1B Q4 · WebGPU vs LM Studio', 'heading')}
  ${text(42, 73, data.testedHardware, 'caption')}
  ${text(plot.x, 108, options.title, 'subheading')}
  ${grid}
  ${marks}
  <rect x="120" y="535" width="16" height="16" rx="3" class="lm"/>${text(144, 548, 'LM Studio', 'caption')}
  <rect x="260" y="535" width="16" height="16" rx="3" class="gpu"/>${text(284, 548, 'Custom WebGPU', 'caption')}
  ${text(42, 577, options.footer, 'caption')}
</svg>\n`;
  writeFileSync(resolve(options.outputPath), svg);
  console.log(`Wrote ${options.outputPath}`);
}

renderChart({
  title: 'Single-stream throughput (requests/second)',
  description: 'Horizontal bars compare warmed single-stream requests per second for LM Studio and the custom WebGPU runtime at six exact token counts.',
  outputPath: 'docs/single-stream-comparison.svg',
  maximum: 80,
  ticks: [0, 20, 40, 60, 80],
  lmValue: (row) => row.lmStudio.singleRps,
  webgpuValue: (row) => row.webgpu.singleRps,
  footer: `Measured ${data.date}; exact token counts include BOS; warmed, isolated conditions.`,
});

renderChart({
  title: '16-concurrent aggregate throughput (requests/second)',
  description: 'Horizontal bars compare aggregate requests per second with 16 simultaneous requests for LM Studio and the custom WebGPU runtime at six exact token counts.',
  outputPath: 'docs/concurrency-16-comparison.svg',
  maximum: 280,
  ticks: [0, 70, 140, 210, 280],
  lmValue: (row) => row.lmStudio.concurrency16AggregateRps,
  webgpuValue: (row) => row.webgpu.concurrency16AggregateRps,
  footer: `Measured ${data.date}; contextual token merge is described in the benchmark methodology.`,
});
