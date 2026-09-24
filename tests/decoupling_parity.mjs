/*
 * Harness for tests/test_decoupling_parity.py.
 *
 * Reads {"doc": <stream document>, "intervals": [[start, end], ...]} on stdin — or
 * {"file": <path to a .gpx/.fit>, ...}, which web/activity_file.js parses first — and
 * writes what web/decoupling.js makes of it to stdout, so the Python test can compare the
 * two implementations number by number. A parsed document is echoed back as `doc`, for
 * the parser half of the comparison. NaN has no JSON spelling, so it travels as null and
 * the Python side treats null and NaN as the same thing.
 *
 * Run by hand with:  node tests/decoupling_parity.mjs < job.json
 */
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import {
  prepare,
  loadStreams,
  analyzeInterval,
  comparisonMetrics,
  decouplingLabel,
  intervalsAreValid,
  paceAxisRange,
} from '../web/decoupling.js';
import { parseActivityFile } from '../web/activity_file.js';

const clean = (value) => (Number.isFinite(value) ? value : null);
const cleanArray = (values) => Array.from(values, clean);

const job = JSON.parse(readFileSync(0, 'utf8'));
const doc = job.file ? parseActivityFile(readFileSync(job.file), basename(job.file)) : job.doc;
const data = prepare(loadStreams(doc), { doubleCadence: job.doubleCadence });

const intervals = job.intervals.map(([start, end]) => {
  const result = analyzeInterval(data, start, end);
  if (!result) return null;
  const { indices, ...fields } = result;
  return Object.fromEntries(Object.entries(fields).map(([k, v]) => [k, clean(v)]));
});

const pairs = [];
for (let i = 0; i + 1 < intervals.length; i += 2) {
  const metrics = comparisonMetrics(intervals[i], intervals[i + 1]);
  pairs.push({
    metrics: Object.fromEntries(Object.entries(metrics).map(([k, v]) => [k, clean(v)])),
    label: decouplingLabel(metrics.decoupling),
    valid: intervalsAreValid(...job.intervals[i], ...job.intervals[i + 1]),
  });
}

process.stdout.write(
  JSON.stringify({
    n: data.n,
    analysisSamples: data.analysisMask.filter(Boolean).length,
    movingSamples: data.moving.filter(Boolean).length,
    hasLatlng: data.hasLatlng,
    cadence: cleanArray(data.cadence),
    paceAxisRange: paceAxisRange(data.pacePlot),
    paceSmooth: cleanArray(data.paceSmooth),
    timeMin: cleanArray(data.timeMin),
    pacePlot: cleanArray(data.pacePlot),
    grade: cleanArray(data.grade),
    gradedSpeedKmh: cleanArray(data.gradedSpeedKmh),
    intervals,
    pairs,
    doc: job.file ? doc : undefined,
  })
);
