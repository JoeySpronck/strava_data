"""Write synthetic Strava exports — a GPX and a FIT — for the tests.

The parser tests need files shaped like what Strava hands you from *Export GPX* and
*Export Original*, but a real export is a real person's GPS trace, and nothing personal
belongs in this repo. So both files are generated from ``fixtures/sample_run.json``, the
synthetic run the maths tests already use, carrying its pause, GPS spike and dropped
heart-rate samples along. Its positions are redrawn as laps of a 400 m-radius loop at
0°N 0°E that follow its distance stream, because a GPX has no speed of its own: pace is
measured from the positions, so they have to agree with the distance.

The FIT deliberately uses more of the format than a watch usually needs, so the
JavaScript decoder in ``web/activity_file.js`` is exercised on it: compressed-timestamp
record headers, a big-endian definition, invalid-value sentinels, and two local message
types sharing one global message.

Re-run after changing ``sample_run.json``::

    python tests/make_export_fixtures.py
"""
import json
import math
import os
import struct
from datetime import datetime, timedelta, timezone

from fitdecode.utils import compute_crc

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
SOURCE = os.path.join(FIXTURES, "sample_run.json")
NAME = "synthetic_run"

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)
DEG_TO_SEMICIRCLE = 2**31 / 180.0
LOOP_RADIUS_M = 400.0
METRES_PER_DEGREE = math.pi / 180.0 * 6371008.8  # at the equator

# (field number, size, base type) — base types from the FIT profile.
UINT8, SINT32, UINT32, ENUM = 0x02, 0x85, 0x86, 0x00
INVALID = {UINT8: 0xFF, SINT32: 0x7FFFFFFF, UINT32: 0xFFFFFFFF, ENUM: 0xFF}
FORMAT = {UINT8: "B", SINT32: "i", UINT32: "I", ENUM: "B"}

RECORD_FIELDS = [
    (0, 4, SINT32),   # position_lat, semicircles
    (1, 4, SINT32),   # position_long
    (5, 4, UINT32),   # distance, m * 100
    (73, 4, UINT32),  # enhanced_speed, m/s * 1000
    (78, 4, UINT32),  # enhanced_altitude, (m + 500) * 5
    (3, 1, UINT8),    # heart_rate
    (4, 1, UINT8),    # cadence, one leg
]
TIMESTAMP_FIELD = (253, 4, UINT32)


def _samples():
    with open(SOURCE) as fh:
        doc = json.load(fh)
    start = datetime.fromisoformat(doc["start_date"])
    s = doc["streams"]
    n = len(s["t"])
    get = lambda key, i: s[key][i] if key in s else None
    samples = [
        {key: get(key, i) for key in ("t", "dist", "speed", "hr", "cad", "alt", "lat", "lng")}
        for i in range(n)
    ]
    for p in samples:
        # Keep the fixture's missing-GPS samples missing; move the rest onto the loop.
        if p["lat"] is None or p["lng"] is None or p["dist"] is None:
            p["lat"] = p["lng"] = None
            continue
        angle = p["dist"] / LOOP_RADIUS_M
        p["lat"] = LOOP_RADIUS_M * math.sin(angle) / METRES_PER_DEGREE
        p["lng"] = LOOP_RADIUS_M * (1.0 - math.cos(angle)) / METRES_PER_DEGREE
    return doc, start, samples


# ---------------------------------------------------------------- GPX

def write_gpx(path):
    doc, start, samples = _samples()
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx creator="StravaGPX" version="1.1" xmlns="http://www.topografix.com/GPX/1/1"'
        ' xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">',
        f' <metadata><time>{start:%Y-%m-%dT%H:%M:%SZ}</time></metadata>',
        " <trk>",
        f"  <name>{doc['name']} &amp; friends</name>",
        "  <type>running</type>",
        "  <trkseg>",
    ]
    for p in samples:
        if p["t"] is None:
            continue
        when = start + timedelta(seconds=p["t"])
        position = (f' lat="{p["lat"]:.7f}" lon="{p["lng"]:.7f}"'
                    if p["lat"] is not None and p["lng"] is not None else "")
        out.append(f"   <trkpt{position}>")
        if p["alt"] is not None:
            out.append(f"    <ele>{p['alt']:.1f}</ele>")
        out.append(f"    <time>{when:%Y-%m-%dT%H:%M:%SZ}</time>")
        ext = [f"<gpxtpx:{tag}>{p[key]:.0f}</gpxtpx:{tag}>"
               for tag, key in (("hr", "hr"), ("cad", "cad")) if p[key] is not None]
        if ext:
            out.append("    <extensions><gpxtpx:TrackPointExtension>"
                       + "".join(ext) + "</gpxtpx:TrackPointExtension></extensions>")
        out.append("   </trkpt>")
    out += ["  </trkseg>", " </trk>", "</gpx>", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


# ---------------------------------------------------------------- FIT

def _definition(local, global_num, fields, big_endian=False):
    arch = ">" if big_endian else "<"
    body = struct.pack(arch + "BBHB", 0, 1 if big_endian else 0, global_num, len(fields))
    for num, size, base in fields:
        body += struct.pack("BBB", num, size, base)
    return bytes([0x40 | local]) + body


def _values(fields, values, big_endian=False):
    arch = ">" if big_endian else "<"
    out = b""
    for (_, _, base), value in zip(fields, values):
        out += struct.pack(arch + FORMAT[base], INVALID[base] if value is None else value)
    return out


def _scaled(value, scale, offset=0.0):
    return None if value is None else int(round((value + offset) * scale))


def write_fit(path):
    _, start, samples = _samples()
    start_ts = int((start - FIT_EPOCH).total_seconds())

    body = b""
    # local 0: file_id (type=activity), so the file is a well-formed activity.
    file_id = [(0, 1, ENUM), (4, 4, UINT32)]
    body += _definition(0, 0, file_id) + bytes([0]) + _values(file_id, [4, start_ts])

    # local 0 again: record with a full timestamp. local 1: record without one, for the
    # compressed-timestamp headers (which can only address local types 0-3).
    full = [TIMESTAMP_FIELD] + RECORD_FIELDS
    body += _definition(0, 20, full) + _definition(1, 20, RECORD_FIELDS)

    last_full = None
    for i, p in enumerate(samples):
        if p["t"] is None:
            continue
        ts = start_ts + int(p["t"])
        values = [
            _scaled(p["lat"], DEG_TO_SEMICIRCLE),
            _scaled(p["lng"], DEG_TO_SEMICIRCLE),
            _scaled(p["dist"], 100),
            _scaled(p["speed"], 1000),
            _scaled(p["alt"], 5, 500),
            None if p["hr"] is None else int(p["hr"]),
            None if p["cad"] is None else int(p["cad"]),
        ]
        # Every tenth record, and after any gap a 5-bit offset cannot span, gets a full
        # timestamp; the rest use the compressed header.
        if last_full is None or i % 10 == 0 or ts - last_full >= 31:
            body += bytes([0]) + _values(full, [ts] + values)
            last_full = ts
        else:
            body += bytes([0x80 | (1 << 5) | (ts & 0x1F)]) + _values(RECORD_FIELDS, values)
            last_full = ts

    # local 2, big-endian: session with sport = running and start_time.
    session = [(5, 1, ENUM), (2, 4, UINT32), (253, 4, UINT32)]
    end_ts = start_ts + int(samples[-1]["t"])
    body += (_definition(2, 18, session, big_endian=True) + bytes([2])
             + _values(session, [1, start_ts, end_ts], big_endian=True))

    header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(body), b".FIT")
    header += struct.pack("<H", compute_crc(header))
    data = header + body
    data += struct.pack("<H", compute_crc(data))
    with open(path, "wb") as fh:
        fh.write(data)


if __name__ == "__main__":
    write_gpx(os.path.join(FIXTURES, f"{NAME}.gpx"))
    write_fit(os.path.join(FIXTURES, f"{NAME}.fit"))
    print(f"wrote fixtures/{NAME}.gpx and fixtures/{NAME}.fit from {os.path.basename(SOURCE)}")
