"""Split runs that were partly hiked into a run part and a hike part, and link activities.

Some (trail) runs are tagged ``<int>% hike`` in the title, description or private note,
e.g. "steep one, ~30% hiked". Counted whole, those runs inflate the running volume and
make the pace look slower than the running actually was. ``split_activities`` replaces each
tagged run with two rows sharing its id and date:

* a run row: only the running distance and moving time, so the average speed (and thus
  pace, risk and the weekly targets) reflect the running alone;
* a hike row (type ``Hike``), which lands in the hiking plots.

Where the split falls comes from the cadence stream: hiking cadence (~50-60 one-leg spm)
sits well below running cadence (~80-90), so each moving sample is labelled by a threshold
fitted per activity. The tagged percentage is only a sanity check; a large mismatch prints
a warning. Fallbacks, in order: no cadence -> the slowest p% of distance by pace is hike;
no stream at all -> the distance is split by p and the run keeps the original pace.

Only the explicit percentage triggers a split, so notes like "walked the uphills" don't.

Linked activities (a split pair, or activities on one day that all mention ``multisport``)
get a ``link_marker`` so the stacked weekly plots can mark them with a black symbol:
the first linked group in a week gets a dot, the next a triangle, then square, then x.
"""
import re

import numpy as np
import pandas as pd

HIKE_TAG = re.compile(r'\b(\d{1,2})\s*%\s*hik(?:e|ing|ed)\b', re.IGNORECASE)
# "multisport", "multi sport" or "multi-sport", any case.
MULTISPORT_TAG = re.compile(r'\bmulti[\s-]?sport\b', re.IGNORECASE)

# Stream types the split needs; fetched only for tagged runs.
SPLIT_STREAM_TYPES = ["time", "distance", "cadence", "moving", "velocity_smooth", "heartrate"]

# Cadence is one-leg steps/min, as Strava reports it.
CADENCE_FALLBACK_THRESHOLD = 70   # used when the activity's cadence isn't clearly bimodal
CADENCE_THRESHOLD_RANGE = (60, 78)  # a fitted threshold is clamped into this band
CADENCE_MIN_CLASS_GAP = 15        # class means closer than this = one mode, not two
CADENCE_SMOOTH_S = 20             # rolling-median window, seconds
MIN_SEGMENT_S = 30                # run/hike stretches shorter than this merge into neighbours
MIN_VALID_CADENCE_FRACTION = 0.5  # below this share of valid samples, fall back to pace
SANITY_TOLERANCE = 0.15           # warn if the cadence split is >15 %-points off the tag

LINK_MARKERS = ["o", "^", "s", "x"]  # per week: 1st linked group, 2nd, 3rd, 4th+


def parse_hike_percent(*texts):
    """First ``<int>% hike`` percentage found in the given texts (1-99), else None."""
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        m = HIKE_TAG.search(t)
        if m and 0 < int(m.group(1)) < 100:
            return int(m.group(1))
    return None


def has_multisport_tag(*texts):
    return any(isinstance(t, str) and MULTISPORT_TAG.search(t) for t in texts)


def _as_array(streams, key, n):
    values = streams.get(key)
    if values is None or len(values) != n:
        return np.full(n, np.nan)
    return np.array([np.nan if v is None else float(v) for v in values])


def _cadence_threshold(cad):
    """Otsu threshold between the hike and run cadence modes, or the fixed fallback.

    Otsu always finds *a* split, even in a run that was never hiked, so the result is only
    trusted when the two classes are far apart; it is then clamped to a plausible band.
    """
    cad = cad[np.isfinite(cad) & (cad > 30) & (cad < 120)]
    if cad.size < 10:
        return CADENCE_FALLBACK_THRESHOLD
    best_t, best_var = None, -1.0
    for t in np.arange(40, 100, 0.5):
        lo, hi = cad[cad < t], cad[cad >= t]
        if lo.size == 0 or hi.size == 0:
            continue
        w_lo, w_hi = lo.size / cad.size, hi.size / cad.size
        between = w_lo * w_hi * (lo.mean() - hi.mean()) ** 2
        if between > best_var:
            best_var, best_t = between, t
    if best_t is None:
        return CADENCE_FALLBACK_THRESHOLD
    gap = cad[cad >= best_t].mean() - cad[cad < best_t].mean()
    if gap < CADENCE_MIN_CLASS_GAP:
        return CADENCE_FALLBACK_THRESHOLD
    return float(np.clip(best_t, *CADENCE_THRESHOLD_RANGE))


def _merge_short_segments(is_hike, dt, min_s):
    """Flip run/hike stretches shorter than ``min_s`` seconds, shortest first."""
    is_hike = is_hike.copy()
    while True:
        edges = np.flatnonzero(np.diff(is_hike.astype(int))) + 1
        starts = np.r_[0, edges]
        ends = np.r_[edges, is_hike.size]
        if starts.size <= 1:
            return is_hike
        durations = np.array([dt[s:e].sum() for s, e in zip(starts, ends)])
        i = int(np.argmin(durations))
        if durations[i] >= min_s:
            return is_hike
        is_hike[starts[i]:ends[i]] = ~is_hike[starts[i]:ends[i]]


def classify_streams(streams, hike_percent=None):
    """Label each moving interval of a stream as run or hike.

    ``streams`` maps Strava stream types to lists (``time``, ``distance`` required;
    ``cadence``, ``moving``, ``velocity_smooth``, ``heartrate`` optional).

    Returns a dict with per-part distance (m), moving time (s) and mean heart rate, the
    method used (``cadence`` or ``pace``) and the fitted cadence threshold, or None when
    the stream can't be split (missing time/distance, or no cadence and no percentage).
    """
    t = streams.get("time")
    d = streams.get("distance")
    if not t or not d or len(t) != len(d) or len(t) < 3:
        return None
    n = len(t)
    t = _as_array(streams, "time", n)
    d = _as_array(streams, "distance", n)
    cad = _as_array(streams, "cadence", n)
    hr = _as_array(streams, "heartrate", n)
    moving_raw = streams.get("moving")
    if moving_raw is not None and len(moving_raw) == n:
        moving = np.array([bool(v) for v in moving_raw])
    else:
        speed = _as_array(streams, "velocity_smooth", n)
        moving = ~(speed <= 0)

    # Interval i spans samples i -> i+1; it counts when the sample it ends on is moving.
    dt = np.diff(t)
    dd = np.clip(np.diff(d), 0, None)
    keep = moving[1:] & np.isfinite(dt) & np.isfinite(dd) & (dt > 0)
    dt, dd = dt[keep], dd[keep]
    cad_iv, hr_iv = cad[1:][keep], hr[1:][keep]
    if dt.size == 0 or dd.sum() <= 0:
        return None

    # Cadence 0 while moving is a sensor dropout, not a standstill: treat as missing.
    cad_iv = np.where(cad_iv > 0, cad_iv, np.nan)
    valid_cad = np.isfinite(cad_iv).mean()

    threshold = None
    if valid_cad >= MIN_VALID_CADENCE_FRACTION:
        window = max(1, int(round(CADENCE_SMOOTH_S / max(np.median(dt), 1e-3))))
        smooth = (pd.Series(cad_iv).rolling(window, center=True, min_periods=1)
                  .median().to_numpy())
        # Any gap the median couldn't bridge is filled from its neighbours.
        smooth = pd.Series(smooth).ffill().bfill().to_numpy()
        threshold = _cadence_threshold(smooth)
        is_hike = _merge_short_segments(smooth < threshold, dt, MIN_SEGMENT_S)
        method = "cadence"
    elif hike_percent is not None:
        # Slowest intervals first, until they cover the tagged share of the distance.
        speed = dd / dt
        order = np.argsort(speed, kind="stable")
        cum = np.cumsum(dd[order])
        n_hike = int(np.searchsorted(cum, dd.sum() * hike_percent / 100.0)) + 1
        is_hike = np.zeros(dt.size, dtype=bool)
        is_hike[order[:n_hike]] = True
        method = "pace"
    else:
        return None

    def part(mask):
        h = hr_iv[mask]
        w = dt[mask] * np.isfinite(h)
        mean_hr = float(np.nansum(h * dt[mask]) / w.sum()) if w.sum() > 0 else None
        return float(dd[mask].sum()), float(dt[mask].sum()), mean_hr

    run_d, run_t, run_hr = part(~is_hike)
    hike_d, hike_t, hike_hr = part(is_hike)
    return dict(
        method=method, threshold=threshold,
        run_distance=run_d, run_time=run_t, run_hr=run_hr,
        hike_distance=hike_d, hike_time=hike_t, hike_hr=hike_hr,
        hike_fraction=hike_d / (run_d + hike_d),
    )


def split_run(row, hike_percent, streams=None, warn=print):
    """(run_row, hike_row) dicts for one tagged run, or None to leave it untouched.

    Stream totals are rescaled to the activity's summary distance and moving time, so
    splitting never changes the total volume, only where it's counted.
    """
    row = dict(row)
    total_d = float(row["distance"])
    total_t = float(row["moving_time"])
    label = f"{row.get('name')!r} ({row.get('id')})"

    result = classify_streams(streams, hike_percent) if streams else None
    if result is not None:
        frac_d = result["hike_fraction"]
        frac_t = result["hike_time"] / (result["run_time"] + result["hike_time"])
        hike_d, hike_t = total_d * frac_d, total_t * frac_t
        run_hr, hike_hr = result["run_hr"], result["hike_hr"]
        if abs(frac_d - hike_percent / 100.0) > SANITY_TOLERANCE:
            warn(f"  hike split {label}: {result['method']} split gives {frac_d:.0%} hiked, "
                 f"tagged {hike_percent}% — using the {result['method']} split")
        method = result["method"]
    else:
        # No usable stream: split distance by the tag; the run keeps the original pace.
        hike_d = total_d * hike_percent / 100.0
        speed = float(row.get("average_speed") or 0.0)
        run_time = (total_d - hike_d) / speed if speed > 0 else total_t * (1 - hike_percent / 100.0)
        hike_t = max(total_t - run_time, 0.0)
        run_hr = hike_hr = row.get("average_heartrate")
        method = "percentage"
        warn(f"  hike split {label}: no usable stream, split {hike_percent}% by distance")

    run_d, run_t = total_d - hike_d, total_t - hike_t
    if run_d <= 0 or run_t <= 0 or hike_d <= 0 or hike_t <= 0:
        warn(f"  hike split {label}: split left an empty part, keeping it as one run")
        return None

    def scaled(part_t):
        e = row.get("elapsed_time")
        return float(e) * part_t / total_t if e is not None and total_t > 0 else None

    run_row = {**row, "distance": run_d, "moving_time": run_t, "elapsed_time": scaled(run_t),
               "average_speed": run_d / run_t, "average_heartrate": run_hr,
               "split_part": "run", "split_method": method}
    hike_row = {**row, "type": "Hike", "sport_type": "Hike",
                "distance": hike_d, "moving_time": hike_t, "elapsed_time": scaled(hike_t),
                "average_speed": hike_d / hike_t, "average_heartrate": hike_hr,
                "split_part": "hike", "split_method": method}
    # Summary-only fields that can't be apportioned without guessing.
    for r in (run_row, hike_row):
        for key in ("total_elevation_gain", "kilojoules", "max_heartrate", "max_speed",
                    "elev_high", "elev_low"):
            if key in r:
                r[key] = None
    return run_row, hike_row


def tagged_runs(df):
    """Ids of runs tagged ``<int>% hike`` -> the percentage."""
    out = {}
    for _, r in df[df["type"] == "Run"].iterrows():
        p = parse_hike_percent(r.get("name"), r.get("description"), r.get("private_note"))
        if p is not None:
            out[r["id"]] = p
    return out


def split_activities(df, streams_by_id, warn=print):
    """Replace each tagged run in ``df`` with its run + hike rows.

    ``df`` needs the summary columns plus ``description`` / ``private_note``;
    ``streams_by_id`` maps an id to its stream dict (or None). Adds a ``split_part``
    column: "run" / "hike" for split rows, None elsewhere.
    """
    tags = tagged_runs(df)
    rows = []
    for _, r in df.iterrows():
        pct = tags.get(r["id"])
        parts = split_run(r, pct, streams_by_id.get(r["id"]), warn=warn) if pct else None
        if parts is None:
            rows.append({**r.to_dict(), "split_part": None, "split_method": None})
        else:
            rows.extend(parts)
    return pd.DataFrame(rows, columns=list(df.columns) + ["split_part", "split_method"])


def _local_day(row):
    d = row.get("start_date_local")
    if d is None or pd.isna(d):
        d = row["start_date"]
    return pd.Timestamp(d).date()


def assign_link_markers(df):
    """Add ``link_marker``: a matplotlib marker for linked activities, None otherwise.

    A group is a split pair (same id) or all activities on one local day that mention
    ``multisport``. Groups are numbered per week by their first start, so the first
    linked group of a week gets LINK_MARKERS[0], the second LINK_MARKERS[1], and so on.
    """
    df = df.copy()
    keys = pd.Series([None] * len(df), index=df.index, dtype=object)
    for idx, r in df.iterrows():
        if r.get("split_part") in ("run", "hike"):
            keys[idx] = f"split:{r['id']}"
        elif has_multisport_tag(r.get("name"), r.get("description"), r.get("private_note")):
            keys[idx] = f"multisport:{_local_day(r)}"
    # A lone "multisport" activity has nothing to link to.
    counts = keys.value_counts()
    keys = keys.where(keys.map(counts).fillna(0) > 1)

    df["link_marker"] = None
    linked = df[keys.notna()]
    if len(linked) == 0:
        return df
    starts = pd.to_datetime(linked["start_date"], utc=True)
    groups = pd.DataFrame({"key": keys[linked.index], "start": starts})
    first = groups.groupby("key")["start"].min().sort_values()
    week = first.dt.tz_convert(None).dt.to_period("W-SUN")
    rank = first.groupby(week).cumcount()
    marker_of = {k: LINK_MARKERS[min(i, len(LINK_MARKERS) - 1)] for k, i in rank.items()}
    df.loc[linked.index, "link_marker"] = keys[linked.index].map(marker_of)
    return df
