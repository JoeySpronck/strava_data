"""One activity from a file you downloaded from Strava, as a stream document.

On an activity page, the ⋯ menu offers *Export GPX* (always there) and *Export Original*
(the file your watch uploaded, usually a ``.fit``). Either one turns into the same stream
document :mod:`strava_data.streams` describes, so the decoupling maths never needs to
know where its samples came from.

The two formats carry different things:

- **FIT** has the watch's own cumulative distance and speed (from the footpod/GPS
  fusion it does on the wrist), so those are used as recorded.
- **GPX** has only positions, times and extension values (heart rate, cadence), so
  distance is summed from the positions and speed is the distance covered over a short
  centred window.

Neither carries Strava's ``moving`` flag or ``grade_smooth``, so moving is "faster than a
slow walk" and grade comes from the altitude, via
:func:`strava_data.decoupling.grade_from_altitude`, as it already did when Strava had no
grade stream.

``web/activity_file.js`` is the browser twin of this module and
``tests/test_decoupling_parity.py`` checks both over the same GPX and FIT fixtures. The
derived series are therefore built with plain scalar arithmetic in a fixed order, the
same order as the JavaScript, rather than with vectorised numpy.
"""
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from strava_data.decoupling import MAX_SAMPLE_GAP_S
from strava_data.streams import SCHEMA

EARTH_RADIUS_M = 6371008.8   # mean Earth radius
SPEED_WINDOW_S = 10.0        # GPX speed: distance over a centred window this long
MOVING_SPEED_MS = 0.5        # below this you are standing, not walking
SEMICIRCLE_DEG = 180.0 / 2**31


# ============================================================
# ENTRY POINTS
# ============================================================

def parse(path):
    """Read a ``.gpx`` or ``.fit`` file into a stream document."""
    with open(path, "rb") as fh:
        data = fh.read()
    return parse_bytes(data, os.path.basename(path))


def parse_bytes(data, filename="activity"):
    """Like :func:`parse`, for file contents already in memory.

    The format is decided by content, not the extension: a FIT file says ``.FIT`` in its
    header, and anything else is tried as GPX.
    """
    fallback_name = os.path.splitext(filename)[0] or "activity"
    if len(data) >= 12 and data[8:12] == b".FIT":
        return _parse_fit(data, fallback_name)
    return _parse_gpx(data, fallback_name)


def is_run(document):
    """Whether cadence is one-leg running cadence (to double) and NGP's model applies."""
    return "run" in (document.get("sport") or "")


# ============================================================
# GPX
# ============================================================

def _local(tag):
    """'{namespace}hr' -> 'hr', so Garmin's gpxtpx:hr and a bare hr read the same."""
    return tag.rsplit("}", 1)[-1]


def _number(text):
    if text is None:
        return math.nan
    text = text.strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _first_text(element, name):
    for child in element.iter():
        if child is not element and _local(child.tag) == name:
            return child.text
    return None


def _epoch_seconds(text):
    """ISO time -> seconds since 1970, or None. A bare time is taken as UTC."""
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


def _parse_gpx(data, fallback_name):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"not a FIT file, and not readable as GPX either ({exc})") from None

    name, sport = None, None
    for trk in root.iter():
        if _local(trk.tag) != "trk":
            continue
        for child in trk:
            if _local(child.tag) == "name" and name is None:
                name = (child.text or "").strip() or None
            elif _local(child.tag) == "type" and sport is None:
                sport = (child.text or "").strip() or None
        break

    epoch, lat, lng, alt, hr, cad = [], [], [], [], [], []
    for point in root.iter():
        if _local(point.tag) != "trkpt":
            continue
        seconds = _epoch_seconds(_first_text(point, "time"))
        if seconds is None:
            continue  # an untimed point says nothing about pace
        epoch.append(seconds)
        lat.append(_number(point.get("lat")))
        lng.append(_number(point.get("lon")))
        alt.append(_number(_first_text(point, "ele")))
        hr.append(_number(_first_text(point, "hr")))
        cad.append(_number(_first_text(point, "cad")))

    if len(epoch) < 2:
        raise ValueError("the GPX file has no timed track points")

    dist = _cumulative_distance(lat, lng)
    t = [e - epoch[0] for e in epoch]
    return _document(
        name=name or fallback_name,
        sport=sport,
        start_epoch=epoch[0],
        source="gpx",
        t=t, dist=dist, speed=_windowed_speed(t, dist),
        hr=hr, cad=cad, alt=alt, lat=lat, lng=lng,
    )


# ============================================================
# FIT
# ============================================================

def _parse_fit(data, fallback_name):
    import io

    import fitdecode

    epoch, lat, lng, alt, hr, cad, dist, speed = [], [], [], [], [], [], [], []
    sport, start = None, None

    def value(frame, *names):
        for name in names:
            if frame.has_field(name):
                v = frame.get_value(name)
                if v is not None:
                    return v
        return None

    def num(v):
        return math.nan if v is None else float(v)

    with fitdecode.FitReader(io.BytesIO(data)) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "record":
                when = value(frame, "timestamp")
                if when is None:
                    continue
                epoch.append(when.timestamp())
                raw_lat, raw_lng = value(frame, "position_lat"), value(frame, "position_long")
                lat.append(math.nan if raw_lat is None else raw_lat * SEMICIRCLE_DEG)
                lng.append(math.nan if raw_lng is None else raw_lng * SEMICIRCLE_DEG)
                alt.append(num(value(frame, "enhanced_altitude", "altitude")))
                hr.append(num(value(frame, "heart_rate")))
                cad.append(num(value(frame, "cadence")))
                dist.append(num(value(frame, "distance")))
                speed.append(num(value(frame, "enhanced_speed", "speed")))
            elif frame.name == "session" and sport is None:
                sport = value(frame, "sport")
                start = value(frame, "start_time")
            elif frame.name == "sport" and sport is None:
                sport = value(frame, "sport")

    if len(epoch) < 2:
        raise ValueError("the FIT file has no timed records")

    t = [e - epoch[0] for e in epoch]
    # A watch without GPS or footpod data still logs time; fall back to what GPX does.
    if not any(math.isfinite(d) for d in dist):
        dist = _cumulative_distance(lat, lng)
    if not any(math.isfinite(s) for s in speed):
        speed = _windowed_speed(t, dist)

    return _document(
        name=fallback_name,
        sport=None if sport is None else str(sport),
        start_epoch=start.timestamp() if start is not None else epoch[0],
        source="fit",
        t=t, dist=dist, speed=speed,
        hr=hr, cad=cad, alt=alt, lat=lat, lng=lng,
    )


# ============================================================
# DERIVED SERIES  (kept step-for-step identical to activity_file.js)
# ============================================================

def _haversine(lat1, lng1, lat2, lng2):
    rad = math.pi / 180.0
    p1 = lat1 * rad
    p2 = lat2 * rad
    sin_dp = math.sin((lat2 - lat1) * rad / 2.0)
    sin_dl = math.sin((lng2 - lng1) * rad / 2.0)
    a = sin_dp * sin_dp + math.cos(p1) * math.cos(p2) * sin_dl * sin_dl
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(a, 1.0)))


def _round_cm(x):
    # Rounded to a centimetre so that a last-bit difference between two maths libraries'
    # sin/cos cannot reach anything downstream; floor(x + 0.5) rather than round(), whose
    # tie-breaking differs between Python and JavaScript.
    return math.floor(x * 100.0 + 0.5) / 100.0


def _cumulative_distance(lat, lng):
    """Metres along the track, summed between consecutive positioned points."""
    total = 0.0
    out = [0.0]
    prev = 0 if math.isfinite(lat[0]) and math.isfinite(lng[0]) else None
    for i in range(1, len(lat)):
        if math.isfinite(lat[i]) and math.isfinite(lng[i]):
            if prev is not None:
                total += _haversine(lat[prev], lng[prev], lat[i], lng[i])
            prev = i
        out.append(_round_cm(total))
    return out


def _windowed_speed(t, dist):
    """m/s over a centred SPEED_WINDOW_S window that never spans a recording gap.

    Point-to-point GPS speed swings by a minute per km from one second to the next; this
    takes the edge off without blurring the pace changes an interval is meant to catch.
    A gap longer than MAX_SAMPLE_GAP_S is a pause, and the window stops at it.
    """
    n = len(t)
    segment = [0] * n
    for i in range(1, n):
        segment[i] = segment[i - 1] + (1 if t[i] - t[i - 1] > MAX_SAMPLE_GAP_S else 0)

    half = SPEED_WINDOW_S / 2.0
    out = []
    for i in range(n):
        lo = i
        while lo > 0 and segment[lo - 1] == segment[i] and t[i] - t[lo - 1] <= half:
            lo -= 1
        hi = i
        while hi < n - 1 and segment[hi + 1] == segment[i] and t[hi + 1] - t[i] <= half:
            hi += 1
        span = t[hi] - t[lo]
        out.append((dist[hi] - dist[lo]) / span if span > 0 else 0.0)
    return out


def _series(values):
    """Floats -> JSON-safe list (NaN as None), or None when there is nothing in it."""
    out = [v if math.isfinite(v) else None for v in values]
    return out if any(v is not None for v in out) else None


def _document(*, name, sport, start_epoch, source, t, dist, speed, hr, cad, alt, lat, lng):
    moving = [1 if math.isfinite(s) and s > MOVING_SPEED_MS else 0 for s in speed]
    streams = {
        "t": t,
        "dist": _series(dist),
        "speed": _series(speed),
        "moving": moving,
        "hr": _series(hr),
        "cad": _series(cad),
        "alt": _series(alt),
    }
    if _series(lat) is not None and _series(lng) is not None:
        streams["lat"] = _series(lat)
        streams["lng"] = _series(lng)
    streams = {k: v for k, v in streams.items() if v is not None}
    if "dist" not in streams or "speed" not in streams:
        raise ValueError("the file has neither distance nor positions to measure pace from")

    start = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    return {
        "schema": SCHEMA,
        "name": name,
        "sport": (sport or "").strip().lower() or None,
        "start_date": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "n": len(t),
        "streams": streams,
    }
