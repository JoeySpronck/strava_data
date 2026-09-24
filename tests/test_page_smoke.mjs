/*
 * Smoke test for web/decoupling.html's inline page script.
 *
 * The maths is covered by test_decoupling_parity.py; this covers the wiring around it,
 * which is where the page has actually broken: a map trace type the bundle did not have,
 * and twice a helper dropped by an edit, leaving a drag handler throwing ReferenceError
 * at every pointer move. `node --check` catches none of that — it only parses.
 *
 * So this runs the real page script against a stubbed DOM and a stubbed Plotly, then
 * drives the things a person does: choose the GPX export from tests/fixtures, drag each
 * handle, switch basemap, drop the FIT export on the page, drop something that is not an
 * activity. Nothing here asserts on how anything *looks*.
 *
 *   node tests/test_page_smoke.mjs
 */
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(HERE);

let failures = 0;

// The page script is imported as a data: URL, so an uncaught error prints the whole
// encoded module — tens of KB of noise around one line that matters. Report the message
// and keep going; the checks below will fail on their own and say what broke.
const pageErrors = [];
function noteError(error) {
  const message = (error && error.message) || String(error);
  if (!pageErrors.includes(message)) pageErrors.push(message);
}
process.on('unhandledRejection', noteError);
process.on('uncaughtException', noteError);

function check(label, condition, detail = '') {
  if (condition) {
    console.log(`  ok  ${label}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${label}${detail ? ` — ${detail}` : ''}`);
  }
}

// ---------------------------------------------------------------- stubs

function makeElement(id) {
  const element = {
    id, textContent: '', innerHTML: '', value: '', label: '', hidden: false,
    style: {}, children: [], listeners: {},
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    append(child) { this.children.push(child); },
    getBoundingClientRect: () => ({ width: 640, height: 300 }),
    fire(type, event = {}) {
      for (const fn of this.listeners[type] || []) fn({ preventDefault() {}, ...event });
    },
  };
  return element;
}

const elements = new Map();
const byId = (id) => {
  if (!elements.has(id)) elements.set(id, makeElement(id));
  return elements.get(id);
};

const plots = new Map();
const counts = { relayout: 0, restyle: 0 };

// What a File from <input type="file"> or a drop gives the page: a name and its bytes.
function fixtureFile(name) {
  const bytes = readFileSync(join(HERE, 'fixtures', name));
  return { name, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length) };
}

globalThis.document = { getElementById: byId, createElement: (tag) => makeElement(tag) };
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.localStorage = { getItem: () => null, setItem() {} };
// The page reads only the file it is given; any request at all would be a regression
// towards publishing data again.
let fetches = 0;
globalThis.fetch = async () => { fetches += 1; return { ok: false, status: 404 }; };
globalThis.Plotly = {
  PlotSchema: { get: () => ({ traces: { scatter: {}, scattermap: {} } }) },
  newPlot(node, traces, layout) { plots.set(node.id, { traces, layout }); return Promise.resolve(); },
  relayout() { counts.relayout += 1; return Promise.resolve(); },
  restyle() { counts.restyle += 1; return Promise.resolve(); },
};

// ---------------------------------------------------------------- run the page

const html = readFileSync(join(ROOT, 'web', 'decoupling.html'), 'utf8');
const inline = /<script type="module">([\s\S]*?)<\/script>/.exec(html);
if (!inline) {
  console.log('  FAIL could not find the inline module in web/decoupling.html');
  process.exit(1);
}
// The inline module is imported from a data: URL, so its relative imports need
// absolute ones.
const source = inline[1].replace(/from '\.\/([\w.]+\.js)'/g,
  (_, file) => `from ${JSON.stringify(pathToFileURL(join(ROOT, 'web', file)).href)}`);
await import(`data:text/javascript,${encodeURIComponent(source)}`);
await new Promise((resolve) => setTimeout(resolve, 50));

console.log('before a file');
check('nothing drawn until a file is chosen', !plots.has('chart'));

console.log('choose the GPX export');
byId('file').fire('change', { target: { files: [fixtureFile('synthetic_run.gpx')] } });
await new Promise((resolve) => setTimeout(resolve, 100));
check('no error status', byId('status').textContent === '', byId('status').textContent);
check('analysis is visible', byId('app').hidden === false);
check('the file name replaces the prompt', byId('filename').textContent === 'synthetic_run.gpx',
  byId('filename').textContent);
const gpxDecoupling = (/big">([^<]+)</.exec(byId('stats').innerHTML) || [])[1];
check('full-activity card rendered', /FULL ACTIVITY/.test(byId('overall').innerHTML));
check('chart drawn', plots.has('chart') && plots.get('chart').traces.length === 3);

// The slider track is inset to the chart's plot area; if these drift apart, a handle no
// longer points at the moment below it.
const chartMargin = plots.get('chart').layout.margin;
check('slider track is inset to the plot area',
  byId('range').style.marginLeft === `${chartMargin.l}px`
    && byId('range').style.marginRight === `${chartMargin.r}px`,
  `track ${byId('range').style.marginLeft}/${byId('range').style.marginRight}`
    + ` vs plot ${chartMargin.l}/${chartMargin.r}`);
check('a handle centre is a plain fraction of the track',
  byId('label0').style.left === '20%', byId('label0').style.left);

const paceTrace = plots.get('chart').traces[0];
check('the drawn pace is smoothed', /avg/.test(plots.get('chart').layout.yaxis.title.text),
  plots.get('chart').layout.yaxis.title.text);

const paceAxis = plots.get('chart').layout.yaxis;
check('pace axis is pinned to percentiles, not the extremes',
  Array.isArray(paceAxis.range) && paceAxis.autorange === false
    && paceAxis.range[0] > paceAxis.range[1],
  JSON.stringify(paceAxis.range));
// The synthetic run sits around 5 min/km with one deliberate GPS spike; the axis must
// ignore that rather than stretch to it.
check('the pace spike is left off-axis', paceAxis.range && paceAxis.range[0] < 12,
  JSON.stringify(paceAxis.range));

check('running cadence is doubled to whole steps',
  /Cadence<\/div><div class="value">1\d\d<span class="unit">spm/.test(byId('overall').innerHTML),
  (/Cadence.{0,80}/.exec(byId('overall').innerHTML) || [])[0]);

const map = plots.get('map');
check('map drawn with outlined intervals and badges',
  map && map.traces.length === 9, map ? `${map.traces.length} traces` : 'no map');
if (map) {
  const widths = map.traces.slice(1, 5).map((t) => `${t.line.width}@${t.line.color}`).join(' ');
  check('both intervals are orange over a black outline',
    widths === '10@#000 6@#fb5200 10@#000 6@#fb5200', widths);
}

// ---------------------------------------------------------------- interaction

const handleIds = ['i1start', 'i1end', 'i2start', 'i2end'];
const values = () => handleIds.map((id) => Number(byId(id).value));
const labelPositions = () => [0, 1, 2, 3].map((i) => byId(`label${i}`).style.left);
const decoupling = () => (/big">([^<]+)</.exec(byId('stats').innerHTML) || [])[1];

async function drag(index, to) {
  byId(handleIds[index]).value = String(to);
  try {
    byId(handleIds[index]).fire('input');
  } catch (error) {
    noteError(error);  // a broken handler must fail a check, not abort the run
  }
  await new Promise((resolve) => setTimeout(resolve, 30));
}

console.log('interaction');
const before = { labels: labelPositions(), decoupling: decoupling(), relayout: counts.relayout };
await drag(1, 520);
check('dragging repaints the handle numbers',
  JSON.stringify(labelPositions()) !== JSON.stringify(before.labels));
check('dragging redraws the chart bands', counts.relayout > before.relayout);
check('dragging recomputes the decoupling', decoupling() !== before.decoupling,
  `still ${decoupling()}`);
check('a handle pushes its neighbours along', values()[2] >= values()[1],
  values().join(','));

await drag(0, 900);
check('pushing right keeps every handle in order and in range',
  values().every((v, i, a) => v >= 0 && v <= 1000 && (i === 0 || v > a[i - 1])),
  values().join(','));
await drag(3, 60);
check('pushing left keeps every handle in order and in range',
  values().every((v, i, a) => v >= 0 && v <= 1000 && (i === 0 || v > a[i - 1])),
  values().join(','));

console.log('basemap');
const picker = byId('mapstyle');
check('defaults to the dark basemap', map && map.layout.map.style === 'carto-darkmatter',
  map && String(map.layout.map.style));
picker.value = 'satellite';
picker.fire('change');
await new Promise((resolve) => setTimeout(resolve, 50));
check('switching basemap redraws the map', plots.get('map').layout.map.style === 'satellite',
  String(plots.get('map').layout.map.style));

console.log('drop the FIT export');
byId('dropzone').fire('drop', { dataTransfer: { files: [fixtureFile('synthetic_run.fit')] } });
await new Promise((resolve) => setTimeout(resolve, 100));
check('no error status', byId('status').textContent === '', byId('status').textContent);
check('the dropped file is analysed', byId('filename').textContent === 'synthetic_run.fit'
  && /FULL ACTIVITY/.test(byId('overall').innerHTML), byId('filename').textContent);
check('the handles start over for the new file', values().join(',') === '200,400,600,800',
  values().join(','));
// Same run, same default intervals: the watch's speed and GPS-derived speed should
// tell the same story to within a couple of points.
check('FIT and GPX of the same run land close together',
  Math.abs(parseFloat(decoupling()) - parseFloat(gpxDecoupling)) < 2,
  `${gpxDecoupling} (gpx) vs ${decoupling()} (fit)`);

console.log('a file that is not an activity');
byId('file').fire('change', {
  target: { files: [{ name: 'notes.txt', arrayBuffer: async () => new TextEncoder().encode('hello').buffer }] },
});
await new Promise((resolve) => setTimeout(resolve, 50));
check('says it could not read the file', /Could not read notes\.txt/.test(byId('status').textContent),
  byId('status').textContent);
check('and hides the old analysis', byId('app').hidden === true);

check('no network requests were made', fetches === 0, `${fetches} fetch(es)`);

if (pageErrors.length) {
  failures += 1;
  console.log('\nerrors raised by the page script:');
  for (const message of pageErrors) console.log(`  ${message}`);
}

console.log(failures ? `\n${failures} check(s) failed.` : '\nPage wiring is sound.');
process.exit(failures ? 1 : 0);
