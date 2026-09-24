"""The Python and JavaScript decoupling maths must agree, number for number.

``strava_data/decoupling.py`` and ``web/decoupling.js`` are deliberate twins, and so are
``strava_data/activity_file.py`` and ``web/activity_file.js``: the notebook analyses an
exported activity file with the Python, the web page analyses the same file with the
JavaScript, and a runner would have no way of knowing which one to believe if they drifted
apart. This test runs both over identical input and fails on the first number that
disagrees beyond floating-point noise.

It always checks ``tests/fixtures/sample_run.json`` — synthetic, but shaped to hit every
branch: a pause, a GPS speed spike that sigma-clipping must drop, dropped heart-rate
samples, rolling grade, and a recording gap long enough to exercise the moving-time cap.
Three variants are derived from it in memory to reach the fallback paths (no ``grade``
stream, no ``moving`` stream, no heart rate).

It also parses that run as a Strava export both ways — ``synthetic_run.gpx`` (*Export
GPX*) and ``synthetic_run.fit`` (*Export Original*), written by
``tests/make_export_fixtures.py`` — in each language, requires the two parsers to produce
identical stream documents sample for sample, and then compares the maths on them. Python
reads FIT with fitdecode and JavaScript with its own decoder, so this is also what keeps
that decoder honest. Needs Python, numpy, fitdecode and node.

Real exports never go in this repo (they are someone's GPS trace), but you can check your
own files locally by passing them as arguments; they are read and nothing is written::

    python tests/test_decoupling_parity.py ~/Downloads/Morning_Run.fit ~/Downloads/Morning_Run.gpx

Run it with::

    python tests/test_decoupling_parity.py
"""
import copy
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from strava_data import decoupling as dc
from strava_data import activity_file
from strava_data import streams as streams_mod

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "sample_run.json")
RUNNER = os.path.join(HERE, "decoupling_parity.mjs")
EXPORTS = [
    os.path.join(HERE, "fixtures", "synthetic_run.gpx"),
    os.path.join(HERE, "fixtures", "synthetic_run.fit"),
]

# Both sides do the same arithmetic in the same order, but numpy sums pairwise where the
# JS sums straight through, so the last couple of bits can differ on a 10,000-sample
# activity. A relative tolerance of 1e-9 is far tighter than any of these numbers is
# meaningful to, and far looser than that noise.
RTOL = 1e-9
ATOL = 1e-12

# analyze_interval / comparison_metrics fields, Python name -> JavaScript name.
INTERVAL_FIELDS = {
    "start": "start", "end": "end", "duration": "duration", "hr": "hr",
    "cadence": "cadence", "speed": "speed", "pace": "pace", "ngp_speed": "ngpSpeed",
    "ngp": "ngp", "altitude": "altitude", "ef": "ef", "distance": "distance",
    "samples": "samples",
}
METRIC_FIELDS = {
    "hr_change": "hrChange", "pace_change": "paceChange",
    "ngp_change": "ngpChange", "decoupling": "decoupling",
}

# Interval pairs probed on every document, as fractions of the activity's time span.
# The last two pairs are the degenerate cases: a zero-length interval, and interval 1
# running past the start of interval 2.
INTERVAL_FRACTIONS = [
    (0.05, 0.25), (0.60, 0.90),
    (0.20, 0.40), (0.60, 0.80),
    (0.00, 0.50), (0.50, 1.00),
    (0.30, 0.30), (0.70, 0.75),
    (0.10, 0.70), (0.40, 0.90),
]


class ParityError(AssertionError):
    pass


def _finite(value):
    """None (JSON's stand-in for NaN) and NaN are the same absence of a number."""
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else value


def _compare(label, py_value, js_value):
    py, js = _finite(py_value), _finite(js_value)
    if py is None or js is None:
        if py is not js:
            raise ParityError(f"{label}: python={py_value!r} javascript={js_value!r}")
        return
    if not math.isclose(py, js, rel_tol=RTOL, abs_tol=ATOL):
        raise ParityError(
            f"{label}: python={py!r} javascript={js!r} "
            f"(difference {abs(py - js):.3e})"
        )


def _compare_series(label, py_array, js_list):
    py_array = np.asarray(py_array, dtype=float)
    if len(py_array) != len(js_list):
        raise ParityError(f"{label}: length {len(py_array)} vs {len(js_list)}")
    for i, (p, j) in enumerate(zip(py_array, js_list)):
        _compare(f"{label}[{i}]", p, j)


def run_javascript(source, intervals, double_cadence):
    """`source` is a stream document, or the path of a file for JavaScript to parse."""
    key = "file" if isinstance(source, str) else "doc"
    job = json.dumps({key: source, "intervals": intervals, "doubleCadence": double_cadence})
    result = subprocess.run(
        ["node", RUNNER], input=job, capture_output=True, text=True, cwd=HERE
    )
    if result.returncode != 0:
        raise ParityError(f"node runner failed:\n{result.stderr}")
    return json.loads(result.stdout)


def check_parsed(py_doc, js_doc, name):
    """The two parsers must produce the same document, down to every sample."""
    for key in py_doc:
        if key != "streams" and py_doc[key] != js_doc.get(key):
            raise ParityError(f"{name}: {key}: python={py_doc[key]!r} javascript={js_doc.get(key)!r}")
    py_streams, js_streams = py_doc["streams"], js_doc["streams"]
    if sorted(py_streams) != sorted(js_streams):
        raise ParityError(f"{name}: streams {sorted(py_streams)} vs {sorted(js_streams)}")
    for key in py_streams:
        _compare_series(f"{name}: streams.{key}",
                        [math.nan if v is None else v for v in py_streams[key]], js_streams[key])


def check_document(doc, name, double_cadence=True, path=None):
    """Analyse `doc` in both languages and compare everything they produce.

    With `path`, JavaScript parses that file itself instead of being handed `doc`, and
    its parse is compared with `doc` (Python's parse of the same file) first.
    """
    data = dc.prepare(streams_mod.load(doc), double_cadence=double_cadence)

    span_start = float(np.nanmin(data["time_min"]))
    span = float(np.nanmax(data["time_min"])) - span_start
    intervals = [
        (span_start + span * lo, span_start + span * hi)
        for lo, hi in INTERVAL_FRACTIONS
    ]

    js = run_javascript(path or doc, [list(i) for i in intervals], double_cadence)
    if path:
        check_parsed(doc, js["doc"], name)

    _compare(f"{name}: sample count", data["time_min"].size, js["n"])
    _compare(f"{name}: analysis samples", int(data["analysis_mask"].sum()), js["analysisSamples"])
    _compare(f"{name}: moving samples", int(data["moving"].sum()), js["movingSamples"])
    _compare_series(f"{name}: time_min", data["time_min"], js["timeMin"])
    _compare_series(f"{name}: pace_plot", data["pace_plot"], js["pacePlot"])
    _compare_series(f"{name}: grade", data["grade"], js["grade"])
    _compare_series(f"{name}: graded_speed", data["graded_speed_kmh"], js["gradedSpeedKmh"])
    _compare_series(f"{name}: cadence", data["cadence"], js["cadence"])
    _compare_series(f"{name}: smoothed pace", data["pace_smooth"], js["paceSmooth"])

    py_axis = dc.pace_axis_range(data["pace_plot"])
    js_axis = js["paceAxisRange"]
    if (py_axis is None) != (js_axis is None):
        raise ParityError(f"{name}: pace axis {py_axis!r} vs {js_axis!r}")
    if py_axis is not None:
        for i, (p, j) in enumerate(zip(py_axis, js_axis)):
            _compare(f"{name}: pace_axis_range[{i}]", p, j)

    py_intervals = [dc.analyze_interval(data, start, end) for start, end in intervals]
    for index, (py, js_interval) in enumerate(zip(py_intervals, js["intervals"])):
        where = f"{name}: interval {index}"
        if (py is None) != (js_interval is None):
            raise ParityError(f"{where}: python={py!r} javascript={js_interval!r}")
        if py is None:
            continue
        for py_field, js_field in INTERVAL_FIELDS.items():
            _compare(f"{where}.{py_field}", py[py_field], js_interval[js_field])

    for pair_index, js_pair in enumerate(js["pairs"]):
        first, second = py_intervals[2 * pair_index], py_intervals[2 * pair_index + 1]
        metrics = dc.comparison_metrics(first, second)
        where = f"{name}: pair {pair_index}"
        for py_field, js_field in METRIC_FIELDS.items():
            _compare(f"{where}.{py_field}", metrics[py_field], js_pair["metrics"][js_field])
        py_label = dc.decoupling_label(metrics["decoupling"])
        if py_label != js_pair["label"]:
            raise ParityError(f"{where}.label: python={py_label!r} javascript={js_pair['label']!r}")
        py_valid = dc.intervals_are_valid(*intervals[2 * pair_index], *intervals[2 * pair_index + 1])
        if bool(py_valid) != bool(js_pair["valid"]):
            raise ParityError(f"{where}.valid: python={py_valid!r} javascript={js_pair['valid']!r}")

    return len(intervals)


def fixture_documents():
    """The committed fixture, plus variants that reach the fallback paths."""
    with open(FIXTURE) as fh:
        base = json.load(fh)

    yield base, "fixture"

    # No grade_smooth: the grade has to be reconstructed from altitude, which is a very
    # different code path (unique distances, interpolation over a 20 m window).
    no_grade = copy.deepcopy(base)
    del no_grade["streams"]["grade"]
    yield no_grade, "fixture/no-grade"

    # No moving stream, as on pre-2018 activities: moving falls back to speed > 0.
    no_moving = copy.deepcopy(base)
    del no_moving["streams"]["moving"]
    yield no_moving, "fixture/no-moving"

    # No heart rate: EF and decoupling must come out as NaN rather than something wrong.
    no_hr = copy.deepcopy(base)
    del no_hr["streams"]["hr"]
    yield no_hr, "fixture/no-heartrate"


def exported_documents(extra_paths=()):
    """The synthetic run's exports, plus any files given on the command line."""
    for path in EXPORTS:
        yield activity_file.parse(path), f"export/{os.path.basename(path)}", path
    for path in extra_paths:
        path = os.path.abspath(os.path.expanduser(path))
        yield activity_file.parse(path), f"local/{os.path.basename(path)}", path


def main(extra_paths=()):
    checked = 0
    for doc, name in fixture_documents():
        count = check_document(doc, name)
        checked += 1
        print(f"  ok  {name}  ({doc['n']} samples, {count} intervals)")

    # Cycling cadence is already a whole-crank rpm, so the page turns the doubling off
    # for a ride. Both sides have to agree about that too.
    with open(FIXTURE) as fh:
        check_document(json.load(fh), "fixture/single-leg-cadence", double_cadence=False)
    checked += 1
    print("  ok  fixture/single-leg-cadence  (double_cadence=False)")

    exported = 0
    for doc, name, path in exported_documents(extra_paths):
        count = check_document(doc, name, double_cadence=activity_file.is_run(doc), path=path)
        exported += 1
        print(f"  ok  {name}  ({doc['n']} samples, parsed identically, {count} intervals)")

    print(f"\nPython and JavaScript agree on {checked} fixture(s) and {exported} exported file(s).")


# Discovered by pytest when it is installed; the module runs standalone without it.
def test_decoupling_parity():
    main()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except ParityError as exc:
        print(f"\nPARITY FAILURE\n{exc}", file=sys.stderr)
        sys.exit(1)
