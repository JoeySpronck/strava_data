"""Aerobic decoupling: how much efficiency drifts between two stretches of one session.

Decoupling compares the *efficiency factor* (EF = normalized graded speed / heart rate)
of an early interval against a later one. If the pace is honest and aerobic, EF barely
moves; as aerobic fitness runs out, holding the pace costs more heartbeats and EF falls.
Decoupling is that drop as a percentage, so bigger is worse::

    decoupling = -(EF2 - EF1) / EF1 * 100

Normalized Graded Pace, in two steps:

1. Grade adjustment. Speed on a slope becomes the flat-ground speed of equal energy
   cost, via the running economy model of Minetti et al. (2002),
   ``C(i) = 155.4 i^5 - 30.4 i^4 - 43.3 i^3 + 46.3 i^2 + 19.5 i + 3.6`` (J/kg/m), so
   ``graded speed = speed * C(i) / C(0)``.
2. Normalization, as for Normalized Power: a 30 s rolling mean of graded speed, raised
   to the 4th power, averaged, then the 4th root. Surges cost more than they give back.

TrainingPeaks' own NGP model is proprietary; this is the published physiology underneath
it, so numbers can differ a little from theirs.

``web/decoupling.js`` is a line-by-line port of this module — the browser does the same
arithmetic on the same stored streams. ``tests/test_decoupling_parity.py`` runs both over
the same input and fails if they disagree, so the two stay honest about being twins.
"""
import numpy as np

SIGMA_CLIP = 3.0        # pace outliers beyond this many SD are dropped from analysis
NGP_WINDOW_S = 30.0     # rolling window for the 4th-power normalization
MAX_SAMPLE_GAP_S = 5.0  # a longer gap is a paused recording, not 'time spent running'
SMOOTH_WINDOW_S = 20.0  # display smoothing for the pace trace; never used for a metric


def running_cost(grade):
    """Minetti energy cost of running at `grade` (a fraction, not percent), J/kg/m."""
    i = np.clip(grade, -0.45, 0.45)
    return (
        155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
        + 46.3 * i**2 + 19.5 * i + 3.6
    )


def grade_from_altitude(distance_m, altitude, window_m=20.0):
    """Fallback grade (fraction) from altitude, over a +/-10 m window along the route.

    Only used when Strava did not supply `grade_smooth`. Differencing altitude between
    neighbouring samples would amplify GPS noise into double-digit grades, so the rise is
    measured across a fixed distance instead.
    """
    grade = np.zeros_like(distance_m, dtype=float)
    ok = np.isfinite(altitude) & np.isfinite(distance_m)
    if ok.sum() < 2:
        return grade
    d, idx = np.unique(distance_m[ok], return_index=True)
    alt = altitude[ok][idx]
    if d.size < 2:
        return grade
    half = window_m / 2
    lo = np.clip(distance_m - half, d[0], d[-1])
    hi = np.clip(distance_m + half, d[0], d[-1])
    span = hi - lo
    rise = np.interp(hi, d, alt) - np.interp(lo, d, alt)
    with np.errstate(invalid="ignore", divide="ignore"):
        grade = np.where(span > 1.0, rise / span, 0.0)
    return np.nan_to_num(grade)


def normalized_speed(speed, time_s, window_s=NGP_WINDOW_S):
    """4th-power normalization of `speed` over a rolling `window_s` window.

    `time_s` is cumulative *moving* time, so a stop never widens a window. As with
    Normalized Power, the first window is discarded when the interval is long enough for
    that to leave something behind — it averages over fewer samples and reads high.
    """
    speed = np.asarray(speed, dtype=float)
    if speed.size < 2:
        return np.nan

    csum = np.concatenate([[0.0], np.cumsum(speed)])
    left = np.searchsorted(time_s, time_s - window_s, side="right")
    counts = np.arange(1, speed.size + 1) - left
    rolling = (csum[1:] - csum[left]) / counts

    if time_s[-1] - time_s[0] > 2 * window_s:
        rolling = rolling[time_s - time_s[0] >= window_s]

    return float(np.mean(rolling**4) ** 0.25)


def smooth_series(values, time_min, window_s=SMOOTH_WINDOW_S):
    """Centred rolling mean of `values` over a `window_s` window. For drawing only.

    A GPS pace trace is far noisier than the running it describes — second-to-second it
    swings by a minute per km over nothing — and once the axis is scaled to the real
    range (see :func:`pace_axis_range`) that noise is all you can see. Averaging over a
    few seconds shows the shape of the effort instead.

    Samples that were NaN stay NaN, so pauses and clipped outliers remain gaps in the
    line rather than being bridged. No metric is computed from the result: efficiency
    factor, NGP and decoupling all run on the raw samples.
    """
    values = np.asarray(values, dtype=float)
    seconds = np.asarray(time_min, dtype=float) * 60.0
    half = window_s / 2.0

    finite = np.isfinite(values)
    totals = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    counts = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])

    lo = np.searchsorted(seconds, seconds - half, side="left")
    hi = np.searchsorted(seconds, seconds + half, side="right")
    window_total = totals[hi] - totals[lo]
    window_count = counts[hi] - counts[lo]

    out = np.full(values.shape, np.nan)
    usable = finite & (window_count > 0)
    out[usable] = window_total[usable] / window_count[usable]
    return out


def pace_axis_range(pace, low_pct=1.0, high_pct=97.0, pad=0.08):
    """Pace axis limits from the running part only, slow-to-fast (so faster is up).

    Sigma clipping already removes the wild samples, but a long session still holds
    plenty of legitimate near-stops — a gate, a road crossing, the top of a climb — and
    on a linear axis a handful of 20 min/km samples squash the entire rest of the trace
    into a band a few pixels tall. Taking percentiles instead lets those few samples be
    drawn off-axis rather than dictating the scale for everything else.

    Returns None when there is nothing finite to measure.
    """
    pace = np.asarray(pace, dtype=float)
    pace = pace[np.isfinite(pace)]
    if pace.size == 0:
        return None
    fast, slow = np.percentile(pace, [low_pct, high_pct])
    margin = max((slow - fast) * pad, 0.05)
    return [float(slow + margin), float(fast - margin)]


def percent_change(old, new):
    if np.isfinite(old) and np.isfinite(new) and old != 0:
        return (new - old) / old * 100
    return np.nan


def _nanmean(values):
    """np.nanmean without the all-NaN warning."""
    values = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(values)):
        return np.nan
    return float(np.nanmean(values))


def prepare(loaded, double_cadence=True):
    """Turn :func:`strava_data.streams.load` output into analysis-ready arrays.

    `double_cadence` turns Strava's running cadence into whole steps per minute: it
    reports one foot only, so a normal 170 spm arrives as 85. Cycling cadence is already
    a whole-crank rpm, so pass False for a ride.

    Samples are put in time order, converted to the units the dashboard shows
    (minutes, km, km/h, min/km), and reduced to one `analysis_mask`: moving samples whose
    pace is within 3 SD of the session mean. Every interval metric uses exactly that mask,
    so a red light or a GPS glitch cannot quietly move the numbers.
    """
    valid = (
        np.isfinite(loaded["t"])
        & np.isfinite(loaded["dist"])
        & np.isfinite(loaded["speed"])
    )
    # Stable sort, so equal timestamps keep their recorded order in Python and JS alike.
    order = np.argsort(loaded["t"][valid], kind="stable")

    def clean(name):
        return loaded[name][valid][order]

    time_min = clean("t") / 60.0
    distance_km = clean("dist") / 1000.0
    speed_ms = clean("speed")
    speed_kmh = speed_ms * 3.6

    pace_min_km = np.full_like(speed_ms, np.nan)
    positive = speed_ms > 0
    pace_min_km[positive] = 1000.0 / speed_ms[positive] / 60.0

    moving = clean("moving").astype(bool)
    heart_rate = clean("hr")
    cadence = clean("cad") * (2.0 if double_cadence else 1.0)
    altitude = clean("alt")
    grade_pct = clean("grade")
    latitude = clean("lat")
    longitude = clean("lng")

    # Pauses never affect the pace trace, its range, or the interval metrics.
    moving_pace = np.where(moving, pace_min_km, np.nan)
    pace_mean = np.nanmean(moving_pace) if np.any(np.isfinite(moving_pace)) else np.nan
    pace_std = np.nanstd(moving_pace) if np.any(np.isfinite(moving_pace)) else np.nan

    pace_plot = moving_pace.copy()
    with np.errstate(invalid="ignore"):
        pace_plot[
            (pace_plot < pace_mean - SIGMA_CLIP * pace_std)
            | (pace_plot > pace_mean + SIGMA_CLIP * pace_std)
        ] = np.nan

    analysis_mask = np.isfinite(pace_plot)

    # A drawing-only copy of the pace trace. Every metric below uses `pace_plot`; this
    # exists so the chart shows the shape of the effort rather than GPS jitter.
    pace_smooth = smooth_series(pace_plot, time_min)

    if np.any(np.isfinite(grade_pct)):
        grade = np.nan_to_num(grade_pct / 100.0)
    else:
        grade = grade_from_altitude(distance_km * 1000.0, altitude)

    graded_speed_kmh = speed_kmh * running_cost(grade) / running_cost(0.0)

    return {
        "time_min": time_min,
        "distance_km": distance_km,
        "speed_kmh": speed_kmh,
        "pace_min_km": pace_min_km,
        "pace_plot": pace_plot,
        "pace_smooth": pace_smooth,
        "moving": moving,
        "analysis_mask": analysis_mask,
        "heart_rate": heart_rate,
        "cadence": cadence,
        "altitude": altitude,
        "grade": grade,
        "graded_speed_kmh": graded_speed_kmh,
        "latitude": latitude,
        "longitude": longitude,
    }


def analyze_interval(data, start, end):
    """Metrics for the samples of `data` between `start` and `end` minutes, or None."""
    if end <= start:
        return None

    time_min = data["time_min"]
    time_mask = (time_min >= start) & (time_min <= end)
    mask = time_mask & data["analysis_mask"]

    if int(np.sum(mask)) < 2:
        return None

    time_deltas = np.diff(time_min, prepend=time_min[0])
    duration = float(np.sum(time_deltas[mask]))

    mean_hr = _nanmean(data["heart_rate"][mask])
    mean_cadence = _nanmean(data["cadence"][mask])
    mean_speed = _nanmean(data["speed_kmh"][mask])
    mean_altitude = _nanmean(data["altitude"][mask])

    # Pace from the mean speed, not the mean of instantaneous paces: the latter
    # over-weights the slow samples, because pace is 1/speed.
    mean_pace = 60.0 / mean_speed if np.isfinite(mean_speed) and mean_speed > 0 else np.nan

    # Moving time is rebuilt from sample gaps, each capped, so a paused recording does
    # not drop a 10-minute hole into the middle of a 30 s rolling window.
    moving_dt_s = np.minimum(time_deltas[mask] * 60.0, MAX_SAMPLE_GAP_S)
    moving_time_s = np.cumsum(moving_dt_s)
    ngp_speed = normalized_speed(data["graded_speed_kmh"][mask], moving_time_s)

    ngp = 60.0 / ngp_speed if np.isfinite(ngp_speed) and ngp_speed > 0 else np.nan

    # Efficiency factor = normalized graded speed / HR, as TrainingPeaks defines it.
    if np.isfinite(ngp_speed) and np.isfinite(mean_hr) and mean_hr > 0:
        efficiency_factor = ngp_speed / mean_hr
    else:
        efficiency_factor = np.nan

    distance_deltas = np.diff(data["distance_km"], prepend=data["distance_km"][0])
    interval_distance = float(np.sum(distance_deltas[mask]))

    return {
        "start": float(start),
        "end": float(end),
        "duration": duration,
        "hr": mean_hr,
        "cadence": mean_cadence,
        "speed": mean_speed,
        "pace": mean_pace,
        "ngp_speed": ngp_speed,
        "ngp": ngp,
        "altitude": mean_altitude,
        "ef": efficiency_factor,
        "distance": interval_distance,
        "samples": int(np.sum(mask)),
        # The sample selections themselves, for callers that draw the interval rather
        # than just read its numbers — the notebook highlights these on the map and the
        # pace trace. decoupling.js returns the equivalent as `indices`.
        "mask": mask,
        "time_mask": time_mask,
    }


def intervals_are_valid(i1_start, i1_end, i2_start, i2_end):
    """Interval 1 must end before interval 2 starts."""
    return i1_start < i1_end < i2_start < i2_end


def comparison_metrics(interval_1, interval_2):
    """Percentage changes from interval 1 to interval 2, plus the decoupling itself."""
    if interval_1 is None or interval_2 is None:
        return {
            "hr_change": np.nan,
            "pace_change": np.nan,
            "ngp_change": np.nan,
            "decoupling": np.nan,
        }

    ef_change = percent_change(interval_1["ef"], interval_2["ef"])

    return {
        "hr_change": percent_change(interval_1["hr"], interval_2["hr"]),
        "pace_change": percent_change(interval_1["pace"], interval_2["pace"]),
        "ngp_change": percent_change(interval_1["ngp"], interval_2["ngp"]),
        "decoupling": -ef_change if np.isfinite(ef_change) else np.nan,
    }


def decoupling_label(decoupling):
    """The wording the dashboard puts under the decoupling number."""
    if not np.isfinite(decoupling):
        return "Unavailable"
    if decoupling < 0:
        return "Negative"
    if decoupling < 3:
        return "Very low"
    if decoupling < 5:
        return "Low"
    if decoupling < 10:
        return "Moderate"
    return "High"
