"""The stream document: one activity's samples, in the shape every consumer reads.

``strava_data.activity_file`` builds one from a GPX or FIT export; the notebook, and the
browser on the decoupling page (``web/decoupling.js:loadStreams``), read it back into
exactly the same arrays. A document looks like::

    {"schema": 1, "name": "Morning Run", "sport": "running", ...,
     "streams": {"t": [...], "dist": [...], "speed": [...], ...}}

Series, all per sample: ``t`` seconds since the start, ``dist`` cumulative metres,
``speed`` m/s, ``moving`` 0/1, ``hr`` bpm, ``cad`` one-leg spm (as Strava and the watch
report it), ``alt`` metres, ``grade`` percent, ``lat``/``lng`` degrees. Only ``t``,
``dist`` and ``speed`` are required.

`schema` is there so the page can refuse a document it does not understand instead of
drawing nonsense from it. Non-finite samples are stored as ``null``: ``json.dumps`` would
otherwise emit a bare ``NaN``, which is valid Python but not valid JSON, and
``JSON.parse`` in the browser rejects it.
"""
import numpy as np

SCHEMA = 1


def load(document):
    """Stored document -> numpy arrays, in SI units, ready for the decoupling math.

    Missing series come back as all-NaN arrays of the right length (all-False for
    `moving`), so callers never have to check which streams an activity happened to
    have. This is the single loading path on the Python side;
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
