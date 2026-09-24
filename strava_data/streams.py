"""Strava sample streams: fetch once, store compactly, load the same way everywhere.

One activity's streams are ~1 sample per second across nine series. Kept as Strava hands
them over that is a few hundred KB of JSON per activity; rounded to the precision the
numbers actually carry (``speed`` to mm/s, ``lat``/``lng`` to ~1 m) it is a fraction of
that, and every consumer — ``update_plots.py``, the notebook, and the browser on the
decoupling page — reads back exactly the same arrays.

A stored document looks like::

    {"schema": 1, "id": 123, "name": "Morning Run", ..., "streams": {"t": [...], ...}}

`schema` is there so the page can refuse a document it does not understand instead of
drawing nonsense from it. Non-finite samples are stored as ``null``: ``json.dumps`` would
otherwise emit a bare ``NaN``, which is valid Python but not valid JSON, and
``JSON.parse`` in the browser rejects it.
"""
import math

import numpy as np

SCHEMA = 1

# Compact name -> (Strava stream name, decimals). `None` decimals means store as 0/1.
# The compact names are the field names in the stored JSON and in decoupling.js.
SCALAR_FIELDS = (
    ("t", "time", 0),                 # seconds since the start
    ("dist", "distance", 1),          # metres, cumulative
    ("speed", "velocity_smooth", 3),  # m/s
    ("moving", "moving", None),       # Strava's own moving/stopped flag
    ("hr", "heartrate", 0),           # bpm
    ("cad", "cadence", 0),            # spm (one leg, as Strava reports it)
    ("alt", "altitude", 1),           # metres
    ("grade", "grade_smooth", 2),     # percent
)

LATLNG_DECIMALS = 5  # ~1.1 m at the equator, finer than a GPS watch resolves

STREAM_TYPES = [strava for _, strava, _ in SCALAR_FIELDS] + ["latlng"]

# Without these there is nothing to analyse. latlng is deliberately not required: a
# treadmill run still has a perfectly good pace/HR trace, it just gets no map.
REQUIRED_FIELDS = ("t", "dist", "speed")

# Metadata copied from the activity summary onto the stream document, so the page can
# label a chart without also loading the index.
META_FIELDS = (
    "id", "name", "type", "sport_type", "start_date", "start_date_local",
    "distance", "moving_time", "elapsed_time", "total_elevation_gain",
    "average_heartrate", "max_heartrate", "average_speed",
)


def _clean(value, decimals):
    """Round for storage, mapping anything non-finite to None (JSON null)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if decimals == 0:
        return int(round(x))
    return round(x, decimals)


def compact_streams(raw):
    """Strava's stream objects -> ``{"t": [...], "speed": [...], ...}``, or None.

    `raw` is whatever ``client.get_activity_streams`` returned: a mapping of stream name
    to an object with a ``.data`` list. Streams Strava did not supply are simply absent
    from the result. Every returned series is truncated to the shortest one, because
    Strava occasionally returns series that differ by a sample or two.
    """
    if not raw:
        return None

    def data_of(name):
        stream = raw.get(name)
        data = getattr(stream, "data", None) if stream is not None else None
        return data if data else None

    out = {}
    for field, strava_name, decimals in SCALAR_FIELDS:
        data = data_of(strava_name)
        if data is None:
            continue
        if decimals is None:
            out[field] = [1 if bool(v) else 0 for v in data]
        else:
            out[field] = [_clean(v, decimals) for v in data]

    latlng = data_of("latlng")
    if latlng:
        out["lat"] = [_clean(p[0] if p else None, LATLNG_DECIMALS) for p in latlng]
        out["lng"] = [_clean(p[1] if p else None, LATLNG_DECIMALS) for p in latlng]

    if any(field not in out for field in REQUIRED_FIELDS):
        return None

    n = min(len(v) for v in out.values())
    if n < 2:
        return None
    return {k: v[:n] for k, v in out.items()}


def downsample(streams, seconds):
    """Thin `streams` to roughly one sample per `seconds`, keeping the first and last.

    Halves the payload at 2 s with no visible effect on a pace trace. Off by default:
    streams are fetched once and never again, so the cache keeps full resolution and
    only the published copy is thinned, if at all.
    """
    if not seconds or not streams:
        return streams
    t = streams["t"]
    keep = [0]
    last = t[0]
    for i in range(1, len(t)):
        if t[i] is not None and last is not None and t[i] - last >= seconds:
            keep.append(i)
            last = t[i]
    if keep[-1] != len(t) - 1:
        keep.append(len(t) - 1)
    return {k: [v[i] for i in keep] for k, v in streams.items()}


def build_document(activity_meta, streams, sample_seconds=None):
    """Wrap compact `streams` with the activity metadata the page needs."""
    doc = {"schema": SCHEMA}
    for field in META_FIELDS:
        if field in activity_meta:
            doc[field] = activity_meta[field]
    doc["sample_seconds"] = sample_seconds
    doc["n"] = len(streams["t"])
    doc["streams"] = streams
    return doc


def fetch(client, activity_id, resolution="high"):
    """Fetch and compact one activity's streams. None if Strava has none to give."""
    try:
        raw = client.get_activity_streams(
            activity_id, types=STREAM_TYPES, resolution=resolution
        )
    except TypeError:
        # Older/newer stravalib signatures without `resolution`.
        raw = client.get_activity_streams(activity_id, types=STREAM_TYPES)
    return compact_streams(raw)


def load(document):
    """Stored document -> numpy arrays, in SI units, ready for the decoupling math.

    Missing series come back as all-NaN arrays of the right length (all-False for
    `moving`), so callers never have to check which streams an activity happened to
    have. This is the single loading path shared by the notebook and ``update_plots``;
    ``decoupling.js`` does the equivalent in the browser.
    """
    schema = document.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported stream schema {schema!r} (expected {SCHEMA})")

    streams = document["streams"]
    n = len(streams["t"])

    def series(name):
        values = streams.get(name)
        if values is None:
            return np.full(n, np.nan)
        return np.array([np.nan if v is None else float(v) for v in values])

    out = {
        "n": n,
        "t": series("t"),
        "dist": series("dist"),
        "speed": series("speed"),
        "hr": series("hr"),
        "cad": series("cad"),
        "alt": series("alt"),
        "grade": series("grade"),
        "lat": series("lat"),
        "lng": series("lng"),
        "has_latlng": "lat" in streams,
    }
    if "moving" in streams:
        out["moving"] = series("moving") > 0
    else:
        # Pre-2018 activities have no moving stream; standing still is the next best thing.
        out["moving"] = out["speed"] > 0
    return out
