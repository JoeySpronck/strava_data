/*
 * Aerobic decoupling, in the browser.
 *
 * A line-by-line port of strava_data/decoupling.py (plus the array-decoding half of
 * strava_data/streams.py). activity_file.js turns an exported GPX or FIT into a stream
 * document; everything below turns that into the numbers the page shows, so there is no
 * server doing analysis and nothing leaves the browser when you drag a slider.
 *
 * Decoupling compares the efficiency factor (EF = normalized graded speed / heart rate)
 * of an early stretch of a session against a later one. Holding a pace aerobically keeps
 * EF flat; running out of aerobic fitness costs extra heartbeats for the same speed and
 * EF drops. Decoupling is that drop as a percentage, so bigger is worse.
 *
 * Keeping this in step with the Python is the job of tests/test_decoupling_parity.py,
 * which runs both over the same activity and fails if any number disagrees. If you change
 * the maths here, change it there too — that is the deal that lets the notebook and the
 * page claim to show the same thing.
 *
 * Everything in this file is pure: arrays in, numbers out, no DOM, no fetch.
 */

export const SCHEMA = 1;
export const SIGMA_CLIP = 3.0;         // pace outliers beyond this many SD are dropped
export const NGP_WINDOW_S = 30.0;      // rolling window for the 4th-power normalization
export const MAX_SAMPLE_GAP_S = 5.0;   // a longer gap is a paused recording
export const SMOOTH_WINDOW_S = 20.0;   // display smoothing only; never used for a metric

// ============================================================
// SMALL NUMERIC HELPERS (the numpy calls the Python side uses)
// ============================================================

const isFinite_ = Number.isFinite;

/** np.clip for a scalar. */
function clip(x, lo, hi) {
  return Math.min(Math.max(x, lo), hi);
}

/** Mean of the finite entries; NaN if there are none. Mirrors np.nanmean. */
function nanMean(values) {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    if (isFinite_(values[i])) {
      sum += values[i];
      count += 1;
    }
  }
  return count ? sum / count : NaN;
}

/** Population SD (ddof=0, as np.nanstd) of the finite entries; NaN if there are none. */
function nanStd(values) {
  const mean = nanMean(values);
  if (!isFinite_(mean)) return NaN;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    if (isFinite_(values[i])) {
      const d = values[i] - mean;
      sum += d * d;
      count += 1;
    }
  }
  return count ? Math.sqrt(sum / count) : NaN;
}

/** np.diff(values, prepend=values[0]): a leading 0, then successive differences. */
function diffPrepend(values) {
  const out = new Float64Array(values.length);
  for (let i = 1; i < values.length; i++) out[i] = values[i] - values[i - 1];
  return out;
}

/**
 * np.interp: linear interpolation of (xp, fp) at x, clamped to the end values.
 * xp must be sorted ascending.
 */
function interp(x, xp, fp) {
  const last = xp.length - 1;
  if (!isFinite_(x)) return NaN;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[last]) return fp[last];
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  const span = xp[hi] - xp[lo];
  if (span === 0) return fp[lo];
  return fp[lo] + ((fp[hi] - fp[lo]) * (x - xp[lo])) / span;
}

// ============================================================
// NORMALIZED GRADED PACE
//
// 1) Grade adjustment: speed on a slope becomes the flat-ground speed of equal energy
//    cost, using the running economy model of Minetti et al. (2002),
//    C(i) = 155.4 i^5 - 30.4 i^4 - 43.3 i^3 + 46.3 i^2 + 19.5 i + 3.6  (J/kg/m),
//    so graded speed = speed * C(i) / C(0).
// 2) Normalization, as for Normalized Power: a 30 s rolling mean of graded speed, raised
//    to the 4th power, averaged, then the 4th root. Surges cost more than they give back.
//
// TrainingPeaks' own NGP model is proprietary; this is the published physiology
// underneath it, so numbers can differ a little from theirs.
// ============================================================

/** Minetti energy cost of running at `grade` (a fraction, not a percentage), J/kg/m. */
export function runningCost(grade) {
  const i = clip(grade, -0.45, 0.45);
  const i2 = i * i;
  const i3 = i2 * i;
  const i4 = i3 * i;
  const i5 = i4 * i;
  return 155.4 * i5 - 30.4 * i4 - 43.3 * i3 + 46.3 * i2 + 19.5 * i + 3.6;
}

const COST_FLAT = runningCost(0.0);

/**
 * Fallback grade (fraction) from altitude, over a +/-10 m window along the route.
 *
 * Only used when Strava did not supply grade_smooth. Differencing altitude between
 * neighbouring samples would amplify GPS noise into double-digit grades, so the rise is
 * measured across a fixed distance instead.
 */
export function gradeFromAltitude(distanceM, altitude, windowM = 20.0) {
  const n = distanceM.length;
  const grade = new Float64Array(n);

  // np.unique(distance[ok], return_index=True): sorted unique distances, each keeping the
  // altitude recorded at its first occurrence.
  const firstAt = new Map();
  for (let i = 0; i < n; i++) {
    if (isFinite_(altitude[i]) && isFinite_(distanceM[i]) && !firstAt.has(distanceM[i])) {
      firstAt.set(distanceM[i], altitude[i]);
    }
  }
  if (firstAt.size < 2) return grade;

  const d = Array.from(firstAt.keys()).sort((a, b) => a - b);
  const alt = d.map((key) => firstAt.get(key));

  const half = windowM / 2;
  const dMin = d[0];
  const dMax = d[d.length - 1];
  for (let i = 0; i < n; i++) {
    const lo = clip(distanceM[i] - half, dMin, dMax);
    const hi = clip(distanceM[i] + half, dMin, dMax);
    const span = hi - lo;
    // A NaN distance makes span NaN, the comparison false, and the grade 0 — the same
    // path np.where + np.nan_to_num take on the Python side.
    const value = span > 1.0 ? (interp(hi, d, alt) - interp(lo, d, alt)) / span : 0.0;
    grade[i] = isFinite_(value) ? value : 0.0;
  }
  return grade;
}

/**
 * 4th-power normalization of `speed` over a rolling `windowS` window.
 *
 * `timeS` is cumulative *moving* time, so a stop never widens a window. As with
 * Normalized Power the first window is discarded when the interval is long enough for
 * that to leave something behind: it averages over fewer samples and reads high.
 */
export function normalizedSpeed(speed, timeS, windowS = NGP_WINDOW_S) {
  const n = speed.length;
  if (n < 2) return NaN;

  const csum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) csum[i + 1] = csum[i] + speed[i];

  // np.searchsorted(timeS, timeS - windowS, side="right"): how many samples fall at or
  // before the start of each window. Both sequences are non-decreasing, so one forward
  // pointer does the whole sweep.
  const rolling = new Float64Array(n);
  let left = 0;
  for (let i = 0; i < n; i++) {
    const cutoff = timeS[i] - windowS;
    while (left < n && timeS[left] <= cutoff) left += 1;
    rolling[i] = (csum[i + 1] - csum[left]) / (i + 1 - left);
  }

  let start = 0;
  if (timeS[n - 1] - timeS[0] > 2 * windowS) {
    while (start < n && timeS[start] - timeS[0] < windowS) start += 1;
  }

  let sum = 0;
  let count = 0;
  for (let i = start; i < n; i++) {
    const v = rolling[i];
    sum += v * v * v * v;
    count += 1;
  }
  return Math.pow(sum / count, 0.25);
}

/**
 * Centred rolling mean of `values` over a `windowS` window. For drawing only.
 *
 * A GPS pace trace is far noisier than the running it describes — second-to-second it
 * swings by a minute per km over nothing — and once the axis is scaled to the real range
 * (see paceAxisRange) that noise is all you can see. Averaging over a few seconds shows
 * the shape of the effort instead.
 *
 * Samples that were NaN stay NaN, so pauses and clipped outliers remain gaps in the line
 * rather than being bridged. No metric is computed from the result: efficiency factor,
 * NGP and decoupling all run on the raw samples.
 */
export function smoothSeries(values, timeMin, windowS = SMOOTH_WINDOW_S) {
  const n = values.length;
  const seconds = new Float64Array(n);
  for (let i = 0; i < n; i++) seconds[i] = timeMin[i] * 60.0;
  const half = windowS / 2.0;

  const totals = new Float64Array(n + 1);
  const counts = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) {
    const ok = Number.isFinite(values[i]);
    totals[i + 1] = totals[i] + (ok ? values[i] : 0);
    counts[i + 1] = counts[i] + (ok ? 1 : 0);
  }

  // Both window edges only ever move forward, so one pointer each sweeps the whole
  // series: `lo` is searchsorted(..., "left"), `hi` is searchsorted(..., "right").
  const out = new Float64Array(n);
  let lo = 0;
  let hi = 0;
  for (let i = 0; i < n; i++) {
    const from = seconds[i] - half;
    const to = seconds[i] + half;
    while (lo < n && seconds[lo] < from) lo += 1;
    while (hi < n && seconds[hi] <= to) hi += 1;
    const count = counts[hi] - counts[lo];
    out[i] = Number.isFinite(values[i]) && count > 0
      ? (totals[hi] - totals[lo]) / count
      : NaN;
  }
  return out;
}

/** numpy's default percentile ("linear"), over an already-sorted ascending array. */
function percentileOf(sorted, p) {
  const position = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/**
 * Pace axis limits from the running part only, slow-to-fast (so faster is up).
 *
 * Sigma clipping already removes the wild samples, but a long session still holds plenty
 * of legitimate near-stops — a gate, a road crossing, the top of a climb — and on a
 * linear axis a handful of 20 min/km samples squash the entire rest of the trace into a
 * band a few pixels tall. Taking percentiles instead lets those few be drawn off-axis
 * rather than dictating the scale for everything else.
 *
 * Returns null when there is nothing finite to measure.
 */
export function paceAxisRange(pace, lowPct = 1.0, highPct = 97.0, pad = 0.08) {
  const finite = [];
  for (let i = 0; i < pace.length; i++) {
    if (Number.isFinite(pace[i])) finite.push(pace[i]);
  }
  if (!finite.length) return null;
  finite.sort((a, b) => a - b);
  const fast = percentileOf(finite, lowPct);
  const slow = percentileOf(finite, highPct);
  const margin = Math.max((slow - fast) * pad, 0.05);
  return [slow + margin, fast - margin];
}

export function percentChange(oldValue, newValue) {
  if (isFinite_(oldValue) && isFinite_(newValue) && oldValue !== 0) {
    return ((newValue - oldValue) / oldValue) * 100;
  }
  return NaN;
}

// ============================================================
// LOADING  (the JS twin of strava_data/streams.py:load)
// ============================================================

/**
 * A stream document -> arrays in SI units.
 *
 * Series the file never had come back all-NaN (all-false for `moving`) at the right
 * length, so nothing downstream has to ask which streams this activity happened to have.
 * Stored nulls are the non-finite samples; JSON has no NaN of its own.
 */
export function loadStreams(doc) {
  if (doc.schema !== SCHEMA) {
    throw new Error(`unsupported stream schema ${doc.schema} (expected ${SCHEMA})`);
  }
  const streams = doc.streams;
  const n = streams.t.length;

  const series = (name) => {
    const values = streams[name];
    const out = new Float64Array(n);
    if (!values) {
      out.fill(NaN);
      return out;
    }
    for (let i = 0; i < n; i++) out[i] = values[i] === null ? NaN : values[i];
    return out;
  };

  const speed = series('speed');
  let moving;
  if (streams.moving) {
    moving = streams.moving.map((v) => v > 0);
  } else {
    // Pre-2018 activities have no moving stream; standing still is the next best thing.
    moving = Array.from(speed, (v) => v > 0);
  }

  return {
    n,
    t: series('t'),
    dist: series('dist'),
    speed,
    hr: series('hr'),
    cad: series('cad'),
    alt: series('alt'),
    grade: series('grade'),
    lat: series('lat'),
    lng: series('lng'),
    moving,
    hasLatlng: Boolean(streams.lat),
  };
}

// ============================================================
// PREPARE
// ============================================================

/**
 * Turn loadStreams() output into analysis-ready arrays.
 *
 * `doubleCadence` turns Strava's running cadence into whole steps per minute: it reports
 * one foot only, so a normal 170 spm arrives as 85. Cycling cadence is already a
 * whole-crank rpm, so pass false for a ride.
 *
 * Samples are put in time order, converted to the units the dashboard shows (minutes, km,
 * km/h, min/km), and reduced to one `analysisMask`: moving samples whose pace is within
 * 3 SD of the session mean. Every interval metric uses exactly that mask, so a red light
 * or a GPS glitch cannot quietly move the numbers.
 */
export function prepare(loaded, { doubleCadence = true } = {}) {
  const order = [];
  for (let i = 0; i < loaded.n; i++) {
    if (isFinite_(loaded.t[i]) && isFinite_(loaded.dist[i]) && isFinite_(loaded.speed[i])) {
      order.push(i);
    }
  }
  // Stable by time, so equal timestamps keep their recorded order here and in numpy.
  order.sort((a, b) => loaded.t[a] - loaded.t[b] || a - b);

  const n = order.length;
  // `convert` must spell each unit change exactly as decoupling.py does — `/ 60`, not
  // `* (1 / 60)`. They differ in the last bit, and normalizedSpeed decides its window
  // edges with an exact `<=`, so a 5e-14 drift in cumulative time moves a sample in or
  // out of a 30 s window and shifts NGP in the fourth decimal. The parity test catches
  // exactly this, which is the reason it compares to 1e-9 rather than to 1e-6.
  const take = (name, convert) => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const value = loaded[name][order[i]];
      out[i] = convert ? convert(value) : value;
    }
    return out;
  };

  const timeMin = take('t', (v) => v / 60.0);
  const distanceKm = take('dist', (v) => v / 1000.0);
  const speedMs = take('speed');
  const speedKmh = take('speed', (v) => v * 3.6);
  const heartRate = take('hr');
  const cadence = take('cad', doubleCadence ? (v) => v * 2 : undefined);
  const altitude = take('alt');
  const gradePct = take('grade');
  const latitude = take('lat');
  const longitude = take('lng');

  const moving = new Array(n);
  for (let i = 0; i < n; i++) moving[i] = Boolean(loaded.moving[order[i]]);

  const paceMinKm = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    paceMinKm[i] = speedMs[i] > 0 ? 1000.0 / speedMs[i] / 60.0 : NaN;
  }

  // Pauses never affect the pace trace, its range, or the interval metrics.
  const movingPace = new Float64Array(n);
  for (let i = 0; i < n; i++) movingPace[i] = moving[i] ? paceMinKm[i] : NaN;

  const paceMean = nanMean(movingPace);
  const paceStd = nanStd(movingPace);
  const low = paceMean - SIGMA_CLIP * paceStd;
  const high = paceMean + SIGMA_CLIP * paceStd;

  const pacePlot = new Float64Array(n);
  const analysisMask = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = movingPace[i];
    // NaN thresholds make both comparisons false, exactly as in numpy, so a session with
    // no usable pace keeps its NaNs rather than throwing everything away.
    pacePlot[i] = p < low || p > high ? NaN : p;
    analysisMask[i] = isFinite_(pacePlot[i]);
  }

  // A drawing-only copy of the pace trace. Every metric below uses `pacePlot`; this
  // exists so the chart shows the shape of the effort rather than GPS jitter.
  const paceSmooth = smoothSeries(pacePlot, timeMin);

  let anyGrade = false;
  for (let i = 0; i < n; i++) {
    if (isFinite_(gradePct[i])) {
      anyGrade = true;
      break;
    }
  }

  let grade;
  if (anyGrade) {
    grade = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const g = gradePct[i] / 100.0;
      grade[i] = isFinite_(g) ? g : 0.0;
    }
  } else {
    const distanceM = new Float64Array(n);
    for (let i = 0; i < n; i++) distanceM[i] = distanceKm[i] * 1000.0;
    grade = gradeFromAltitude(distanceM, altitude);
  }

  const gradedSpeedKmh = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    gradedSpeedKmh[i] = (speedKmh[i] * runningCost(grade[i])) / COST_FLAT;
  }

  return {
    n,
    timeMin,
    distanceKm,
    speedKmh,
    paceMinKm,
    pacePlot,
    paceSmooth,
    moving,
    analysisMask,
    heartRate,
    cadence,
    altitude,
    grade,
    gradedSpeedKmh,
    latitude,
    longitude,
    hasLatlng: loaded.hasLatlng,
  };
}

// ============================================================
// INTERVAL ANALYSIS
// ============================================================

/** Metrics for the samples of `data` between `start` and `end` minutes, or null. */
export function analyzeInterval(data, start, end) {
  if (end <= start) return null;

  const { timeMin } = data;
  const n = data.n;
  const indices = [];
  for (let i = 0; i < n; i++) {
    if (timeMin[i] >= start && timeMin[i] <= end && data.analysisMask[i]) indices.push(i);
  }
  if (indices.length < 2) return null;

  const timeDeltas = diffPrepend(timeMin);
  const distanceDeltas = diffPrepend(data.distanceKm);

  let duration = 0;
  let intervalDistance = 0;
  for (const i of indices) {
    duration += timeDeltas[i];
    intervalDistance += distanceDeltas[i];
  }

  const pick = (array) => {
    const out = new Float64Array(indices.length);
    for (let k = 0; k < indices.length; k++) out[k] = array[indices[k]];
    return out;
  };

  const meanHr = nanMean(pick(data.heartRate));
  const meanCadence = nanMean(pick(data.cadence));
  const meanSpeed = nanMean(pick(data.speedKmh));
  const meanAltitude = nanMean(pick(data.altitude));

  // Pace from the mean speed, not the mean of instantaneous paces: the latter
  // over-weights the slow samples, because pace is 1/speed.
  const meanPace = isFinite_(meanSpeed) && meanSpeed > 0 ? 60.0 / meanSpeed : NaN;

  // Moving time is rebuilt from sample gaps, each capped, so a paused recording does not
  // drop a 10-minute hole into the middle of a 30 s rolling window.
  const movingTimeS = new Float64Array(indices.length);
  let elapsed = 0;
  for (let k = 0; k < indices.length; k++) {
    elapsed += Math.min(timeDeltas[indices[k]] * 60.0, MAX_SAMPLE_GAP_S);
    movingTimeS[k] = elapsed;
  }

  const ngpSpeed = normalizedSpeed(pick(data.gradedSpeedKmh), movingTimeS);
  const ngp = isFinite_(ngpSpeed) && ngpSpeed > 0 ? 60.0 / ngpSpeed : NaN;

  // Efficiency factor = normalized graded speed / HR, as TrainingPeaks defines it.
  const ef =
    isFinite_(ngpSpeed) && isFinite_(meanHr) && meanHr > 0 ? ngpSpeed / meanHr : NaN;

  return {
    start,
    end,
    duration,
    hr: meanHr,
    cadence: meanCadence,
    speed: meanSpeed,
    pace: meanPace,
    ngpSpeed,
    ngp,
    altitude: meanAltitude,
    ef,
    distance: intervalDistance,
    samples: indices.length,
    indices,
  };
}

/** Interval 1 must end before interval 2 starts. */
export function intervalsAreValid(i1Start, i1End, i2Start, i2End) {
  return i1Start < i1End && i1End < i2Start && i2Start < i2End;
}

/** Percentage changes from interval 1 to interval 2, plus the decoupling itself. */
export function comparisonMetrics(interval1, interval2) {
  if (!interval1 || !interval2) {
    return { hrChange: NaN, paceChange: NaN, ngpChange: NaN, decoupling: NaN };
  }
  const efChange = percentChange(interval1.ef, interval2.ef);
  return {
    hrChange: percentChange(interval1.hr, interval2.hr),
    paceChange: percentChange(interval1.pace, interval2.pace),
    ngpChange: percentChange(interval1.ngp, interval2.ngp),
    decoupling: isFinite_(efChange) ? -efChange : NaN,
  };
}

/** The wording the dashboard puts under the decoupling number. */
export function decouplingLabel(decoupling) {
  if (!isFinite_(decoupling)) return 'Unavailable';
  if (decoupling < 0) return 'Negative';
  if (decoupling < 3) return 'Very low';
  if (decoupling < 5) return 'Low';
  if (decoupling < 10) return 'Moderate';
  return 'High';
}

// ============================================================
// FORMATTING
// ============================================================

const DASH = '—';

/** Decimal minutes per km -> "m:ss". */
export function formatPace(pace) {
  if (!isFinite_(pace)) return DASH;
  let minutes = Math.floor((pace * 60.0) / 60);
  let seconds = Math.round((pace * 60.0) % 60);
  if (seconds === 60) {
    minutes += 1;
    seconds = 0;
  }
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

/** Decimal minutes -> "m:ss" or "h:mm:ss". */
export function formatDuration(minutes) {
  if (!isFinite_(minutes)) return DASH;
  const total = Math.round(minutes * 60);
  const hours = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

export function formatChange(value) {
  if (!isFinite_(value)) return DASH;
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

export function formatValue(value, decimals = 1) {
  if (!isFinite_(value)) return DASH;
  return value.toFixed(decimals);
}
