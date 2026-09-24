/*
 * One activity from a file you downloaded from Strava, as a stream document.
 *
 * The browser twin of strava_data/activity_file.py: a GPX (⋯ → Export GPX) or a FIT
 * (⋯ → Export Original, from a watch) goes in, and the same document decoupling.js's
 * loadStreams() reads comes out. The file is read in the page and never leaves it.
 *
 * FIT keeps the watch's own distance and speed. GPX has only positions and times, so
 * distance is summed from the positions and speed is the distance covered over a short
 * centred window. Neither has Strava's moving flag or grade stream: moving is "faster
 * than a slow walk", and prepare() already falls back to grade from altitude.
 *
 * tests/test_decoupling_parity.py parses the same GPX and FIT fixtures here and in
 * Python and fails if any sample differs, so the derived series below are spelled
 * operation for operation like the Python. The FIT decoder is written out here rather
 * than pulled from a CDN so that the Node test runs exactly what the page runs; the
 * Python side reads FIT with fitdecode, which makes the test a check on this decoder.
 *
 * Pure: bytes in, document out. No DOM, so it runs under Node as well.
 */

import { MAX_SAMPLE_GAP_S, SCHEMA } from './decoupling.js';

export const EARTH_RADIUS_M = 6371008.8;  // mean Earth radius
export const SPEED_WINDOW_S = 10.0;       // GPX speed: distance over a centred window this long
export const MOVING_SPEED_MS = 0.5;       // below this you are standing, not walking
const SEMICIRCLE_DEG = 180.0 / 2 ** 31;

// ============================================================
// ENTRY POINTS
// ============================================================

/**
 * File contents -> stream document. The format is decided by content, not the name: a
 * FIT file says ".FIT" in its header, and anything else is tried as GPX.
 */
export function parseActivityFile(input, filename = 'activity') {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const fallbackName = filename.replace(/\.[^.]*$/, '') || 'activity';
  const isFit = bytes.length >= 12
    && String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]) === '.FIT';
  return isFit ? parseFit(bytes, fallbackName) : parseGpx(bytes, fallbackName);
}

/** Whether cadence is one-leg running cadence (to double) and NGP's model applies. */
export function isRun(doc) {
  return (doc.sport || '').includes('run');
}

// ============================================================
// GPX
// ============================================================
// A tag scanner rather than DOMParser, which Node does not have. Strava's and Garmin's
// GPX is flat and regular; tags are matched by local name so gpxtpx:hr, ns3:hr and a
// bare hr all read the same, as ElementTree's local names do on the Python side.

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" };

function decodeEntities(text) {
  return text.replace(/&(#x[0-9a-f]+|#\d+|\w+);/gi, (whole, code) => {
    if (code[0] === '#') {
      const n = code[1] === 'x' || code[1] === 'X'
        ? parseInt(code.slice(2), 16) : parseInt(code.slice(1), 10);
      return String.fromCodePoint(n);
    }
    return ENTITIES[code] ?? whole;
  });
}

function firstText(xml, name) {
  const match = new RegExp(`<(?:[\\w.-]+:)?${name}\\b[^>]*>([^<]*)<`).exec(xml);
  return match ? decodeEntities(match[1]) : null;
}

function number(text) {
  if (text === null || text === undefined) return NaN;
  const trimmed = text.trim();
  // Number('') is 0, where Python's float('') is an error; both mean "absent" here.
  return trimmed ? Number(trimmed) : NaN;
}

/** ISO time -> seconds since 1970, or null. A bare time is taken as UTC. */
function epochSeconds(text) {
  if (!text) return null;
  let iso = text.trim();
  if (/T\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/.test(iso)) iso += 'Z';
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms / 1000 : null;
}

function parseGpx(bytes, fallbackName) {
  const xml = new TextDecoder('utf-8').decode(bytes);
  if (!/<(?:[\w.-]+:)?gpx\b/.test(xml)) {
    throw new Error('not a FIT file, and not readable as GPX either');
  }

  let name = null;
  let sport = null;
  const trk = /<(?:[\w.-]+:)?trk\b[^>]*>([\s\S]*?)(?:<(?:[\w.-]+:)?trkseg\b|<\/(?:[\w.-]+:)?trk>)/.exec(xml);
  if (trk) {
    name = (firstText(trk[1], 'name') || '').trim() || null;
    sport = (firstText(trk[1], 'type') || '').trim() || null;
  }

  const epoch = [];
  const lat = [];
  const lng = [];
  const alt = [];
  const hr = [];
  const cad = [];
  const point = /<(?:[\w.-]+:)?trkpt\b([^>]*?)(?:\/>|>([\s\S]*?)<\/(?:[\w.-]+:)?trkpt>)/g;
  const attr = (attrs, key) => {
    const match = new RegExp(`\\b${key}\\s*=\\s*["']([^"']*)["']`).exec(attrs);
    return match ? match[1] : null;
  };
  for (const [, attrs, body = ''] of xml.matchAll(point)) {
    const seconds = epochSeconds(firstText(body, 'time'));
    if (seconds === null) continue;  // an untimed point says nothing about pace
    epoch.push(seconds);
    lat.push(number(attr(attrs, 'lat')));
    lng.push(number(attr(attrs, 'lon')));
    alt.push(number(firstText(body, 'ele')));
    hr.push(number(firstText(body, 'hr')));
    cad.push(number(firstText(body, 'cad')));
  }

  if (epoch.length < 2) throw new Error('the GPX file has no timed track points');

  const dist = cumulativeDistance(lat, lng);
  const t = epoch.map((e) => e - epoch[0]);
  return makeDocument({
    name: name || fallbackName,
    sport,
    startEpoch: epoch[0],
    source: 'gpx',
    t, dist, speed: windowedSpeed(t, dist), hr, cad, alt, lat, lng,
  });
}

// ============================================================
// FIT
// ============================================================
// Just enough of the FIT protocol for an activity: definition and data messages,
// compressed-timestamp headers, developer fields (skipped), both byte orders and
// chained files. Only record, session and sport messages are decoded.

const FIT_EPOCH_S = 631065600;  // 1989-12-31T00:00:00Z, where FIT timestamps count from

// base type -> [size in bytes, reader, invalid value]
const BASE_TYPES = {
  0x00: [1, 'getUint8', 0xff],          // enum
  0x01: [1, 'getInt8', 0x7f],           // sint8
  0x02: [1, 'getUint8', 0xff],          // uint8
  0x03: [2, 'getInt16', 0x7fff],        // sint16
  0x04: [2, 'getUint16', 0xffff],       // uint16
  0x05: [4, 'getInt32', 0x7fffffff],    // sint32
  0x06: [4, 'getUint32', 0xffffffff],   // uint32
  0x08: [4, 'getFloat32', null],        // float32 (invalid is a NaN)
  0x09: [8, 'getFloat64', null],        // float64
  0x0a: [1, 'getUint8', 0x00],          // uint8z
  0x0b: [2, 'getUint16', 0x0000],       // uint16z
  0x0c: [4, 'getUint32', 0x00000000],   // uint32z
};

const MESG_SESSION = 18;
const MESG_RECORD = 20;
const MESG_SPORT = 12;
const FIELD_TIMESTAMP = 253;

// The FIT profile's sport names, as fitdecode reports them, for the ones worth naming.
const SPORTS = {
  0: 'generic', 1: 'running', 2: 'cycling', 3: 'transition', 4: 'fitness_equipment',
  5: 'swimming', 10: 'training', 11: 'walking', 12: 'cross_country_skiing',
  13: 'alpine_skiing', 14: 'snowboarding', 15: 'rowing', 16: 'mountaineering',
  17: 'hiking', 18: 'multisport', 19: 'paddling',
};

function readFit(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const records = [];
  const session = {};
  let fileStart = 0;

  while (fileStart + 12 <= bytes.length) {
    const headerSize = bytes[fileStart];
    const dataSize = view.getUint32(fileStart + 4, true);
    const magic = String.fromCharCode(...bytes.subarray(fileStart + 8, fileStart + 12));
    if (magic !== '.FIT') break;
    const end = Math.min(fileStart + headerSize + dataSize, bytes.length);
    let pos = fileStart + headerSize;
    const definitions = new Map();
    let lastTimestamp = null;

    while (pos < end) {
      const header = bytes[pos++];
      let local;
      let timestamp = null;

      if (header & 0x80) {
        // Compressed timestamp header: a 5-bit offset on the last full timestamp.
        local = (header >> 5) & 0x03;
        const offset = header & 0x1f;
        if (lastTimestamp !== null) {
          timestamp = ((lastTimestamp & ~0x1f) >>> 0) + offset;
          if (offset < (lastTimestamp & 0x1f)) timestamp += 0x20;
          lastTimestamp = timestamp;
        }
      } else if (header & 0x40) {
        local = header & 0x0f;
        const hasDeveloperFields = header & 0x20;
        const littleEndian = bytes[pos + 1] === 0;
        const global = view.getUint16(pos + 2, littleEndian);
        const fieldCount = bytes[pos + 4];
        pos += 5;
        const fields = [];
        for (let i = 0; i < fieldCount; i++) {
          fields.push({ num: bytes[pos], size: bytes[pos + 1], base: bytes[pos + 2] & 0x1f });
          pos += 3;
        }
        let developerSize = 0;
        if (hasDeveloperFields) {
          const devCount = bytes[pos++];
          for (let i = 0; i < devCount; i++) {
            developerSize += bytes[pos + 1];
            pos += 3;
          }
        }
        definitions.set(local, { global, littleEndian, fields, developerSize });
        continue;
      } else {
        local = header & 0x0f;
      }

      const def = definitions.get(local);
      if (!def) throw new Error('the FIT file is damaged (data before its definition)');

      const values = {};
      for (const field of def.fields) {
        const type = BASE_TYPES[field.base];
        if (type && type[0] === field.size) {
          const raw = view[type[1]](pos, def.littleEndian);
          const invalid = type[2] === null ? Number.isNaN(raw) : raw === type[2];
          if (!invalid) values[field.num] = raw;
        }
        pos += field.size;
      }
      pos += def.developerSize;

      if (values[FIELD_TIMESTAMP] !== undefined) {
        lastTimestamp = values[FIELD_TIMESTAMP];
        timestamp = lastTimestamp;
      }

      if (def.global === MESG_RECORD) {
        if (timestamp !== null) records.push({ timestamp, values });
      } else if (def.global === MESG_SESSION && session.sport === undefined) {
        session.sport = values[5];
        session.startTime = values[2];
      } else if (def.global === MESG_SPORT && session.sport === undefined) {
        session.sport = values[0];
      }
    }
    fileStart = end + 2;  // skip the CRC; a chained file may follow
  }
  return { records, session };
}

function parseFit(bytes, fallbackName) {
  const { records, session } = readFit(bytes);
  if (records.length < 2) throw new Error('the FIT file has no timed records');

  // Scale and offset exactly as fitdecode applies them: value / scale - offset.
  const scaled = (raw, scale, offset = 0) => (raw === undefined ? NaN : raw / scale - offset);
  const pick = (values, preferred, fallback) =>
    (values[preferred] !== undefined ? values[preferred] : values[fallback]);

  const epoch = [];
  const lat = [];
  const lng = [];
  const alt = [];
  const hr = [];
  const cad = [];
  let dist = [];
  let speed = [];
  for (const { timestamp, values: v } of records) {
    epoch.push(timestamp + FIT_EPOCH_S);
    lat.push(v[0] === undefined ? NaN : v[0] * SEMICIRCLE_DEG);
    lng.push(v[1] === undefined ? NaN : v[1] * SEMICIRCLE_DEG);
    alt.push(scaled(pick(v, 78, 2), 5, 500));  // enhanced_altitude, else altitude
    hr.push(v[3] === undefined ? NaN : v[3]);
    cad.push(v[4] === undefined ? NaN : v[4]);
    dist.push(scaled(v[5], 100));
    speed.push(scaled(pick(v, 73, 6), 1000));  // enhanced_speed, else speed
  }

  const t = epoch.map((e) => e - epoch[0]);
  // A watch without GPS or footpod data still logs time; fall back to what GPX does.
  if (!dist.some(Number.isFinite)) dist = cumulativeDistance(lat, lng);
  if (!speed.some(Number.isFinite)) speed = windowedSpeed(t, dist);

  const sport = session.sport === undefined ? null : (SPORTS[session.sport] ?? String(session.sport));
  return makeDocument({
    name: fallbackName,
    sport,
    startEpoch: session.startTime !== undefined ? session.startTime + FIT_EPOCH_S : epoch[0],
    source: 'fit',
    t, dist, speed, hr, cad, alt, lat, lng,
  });
}

// ============================================================
// DERIVED SERIES  (kept step-for-step identical to activity_file.py)
// ============================================================

function haversine(lat1, lng1, lat2, lng2) {
  const rad = Math.PI / 180.0;
  const p1 = lat1 * rad;
  const p2 = lat2 * rad;
  const sinDp = Math.sin(((lat2 - lat1) * rad) / 2.0);
  const sinDl = Math.sin(((lng2 - lng1) * rad) / 2.0);
  const a = sinDp * sinDp + Math.cos(p1) * Math.cos(p2) * sinDl * sinDl;
  return 2.0 * EARTH_RADIUS_M * Math.asin(Math.sqrt(Math.min(a, 1.0)));
}

// Rounded to a centimetre so a last-bit difference between two maths libraries' sin/cos
// cannot reach anything downstream; floor(x + 0.5), because Math.round and Python's
// round() break ties differently.
const roundCm = (x) => Math.floor(x * 100.0 + 0.5) / 100.0;

/** Metres along the track, summed between consecutive positioned points. */
function cumulativeDistance(lat, lng) {
  let total = 0.0;
  const out = [0.0];
  let prev = Number.isFinite(lat[0]) && Number.isFinite(lng[0]) ? 0 : null;
  for (let i = 1; i < lat.length; i++) {
    if (Number.isFinite(lat[i]) && Number.isFinite(lng[i])) {
      if (prev !== null) total += haversine(lat[prev], lng[prev], lat[i], lng[i]);
      prev = i;
    }
    out.push(roundCm(total));
  }
  return out;
}

/** m/s over a centred SPEED_WINDOW_S window that never spans a recording gap. */
function windowedSpeed(t, dist) {
  const n = t.length;
  const segment = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    segment[i] = segment[i - 1] + (t[i] - t[i - 1] > MAX_SAMPLE_GAP_S ? 1 : 0);
  }

  const half = SPEED_WINDOW_S / 2.0;
  const out = [];
  for (let i = 0; i < n; i++) {
    let lo = i;
    while (lo > 0 && segment[lo - 1] === segment[i] && t[i] - t[lo - 1] <= half) lo -= 1;
    let hi = i;
    while (hi < n - 1 && segment[hi + 1] === segment[i] && t[hi + 1] - t[i] <= half) hi += 1;
    const span = t[hi] - t[lo];
    out.push(span > 0 ? (dist[hi] - dist[lo]) / span : 0.0);
  }
  return out;
}

/** Numbers -> JSON-safe array (NaN as null), or null when there is nothing in it. */
function series(values) {
  const out = values.map((v) => (Number.isFinite(v) ? v : null));
  return out.some((v) => v !== null) ? out : null;
}

function makeDocument({ name, sport, startEpoch, source, t, dist, speed, hr, cad, alt, lat, lng }) {
  const moving = speed.map((s) => (Number.isFinite(s) && s > MOVING_SPEED_MS ? 1 : 0));
  const candidates = {
    t,
    dist: series(dist),
    speed: series(speed),
    moving,
    hr: series(hr),
    cad: series(cad),
    alt: series(alt),
  };
  if (series(lat) !== null && series(lng) !== null) {
    candidates.lat = series(lat);
    candidates.lng = series(lng);
  }
  const streams = Object.fromEntries(Object.entries(candidates).filter(([, v]) => v !== null));
  if (!streams.dist || !streams.speed) {
    throw new Error('the file has neither distance nor positions to measure pace from');
  }

  return {
    schema: SCHEMA,
    name,
    sport: (sport || '').trim().toLowerCase() || null,
    start_date: new Date(startEpoch * 1000).toISOString().replace(/\.\d{3}Z$/, 'Z'),
    source,
    n: t.length,
    streams,
  };
}
