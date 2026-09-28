"""Write the example run that the decoupling page loads from "use example activity".

A showcase needs a run that looks like a real long run — steady pace, a gentle warm-up,
heart rate creeping up in the second half — without being anyone's real GPS trace, since
nothing personal belongs in this repo. So it is generated: 75 minutes at an easy, steady
pace, once around a big loop centred on the Amsterdamse Bos, with cardiac drift that puts the
decoupling a little over the 5% line, plus a little sensor noise so the charts look recorded
rather than drawn.

Deterministic (fixed seed). Re-run after changing it::

    python tests/make_example_run.py
"""
import math
import os
import random
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "web", "example_run.gpx")

START = datetime(2026, 3, 8, 8, 15, tzinfo=timezone.utc)
DURATION_S = 75 * 60
STEP_S = 3
CENTRE = (52.3150, 4.8390)           # lat, lon
METRES_PER_DEG_LAT = 111_320.0
METRES_PER_DEG_LON = METRES_PER_DEG_LAT * math.cos(math.radians(CENTRE[0]))


def loop_shape(theta):
    """A wobbly closed loop of unit size, so the route isn't a compass-drawn circle."""
    r = 1 + 0.18 * math.sin(3 * theta) + 0.07 * math.cos(5 * theta + 1)
    return r * math.cos(theta), 0.75 * r * math.sin(theta)


def arc_table(length_m, samples=4000):
    """The loop scaled to ``length_m`` around, with cumulative distance along it."""
    pts = [loop_shape(2 * math.pi * i / samples) for i in range(samples + 1)]
    unit = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    pts = [(x * length_m / unit, y * length_m / unit) for x, y in pts]
    cum = [0.0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
    return pts, cum


def position(dist, pts, cum):
    lap = cum[-1]
    d = min(dist, lap)
    lo, hi = 0, len(cum) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum[mid] <= d:
            lo = mid
        else:
            hi = mid
    f = (d - cum[lo]) / (cum[hi] - cum[lo])
    x = pts[lo][0] + f * (pts[hi][0] - pts[lo][0])
    y = pts[lo][1] + f * (pts[hi][1] - pts[lo][1])
    return x, y


def main():
    rng = random.Random(42)
    times = range(0, DURATION_S + 1, STEP_S)

    # Pace first, so the loop can be sized to the distance it adds up to: one lap, with
    # the finish back on the start.
    speeds, dists = [], []
    dist = 0.0
    for t in times:
        minutes = t / 60
        # Easy pace (~5:35/km) after a 5-minute build, with a slow wobble.
        speed = 2.98 + 0.03 * math.sin(minutes / 2.5)
        if minutes < 5:
            speed *= 0.88 + 0.12 * minutes / 5
        speed += rng.gauss(0, 0.04)
        if t:
            dist += speed * STEP_S
        speeds.append(speed)
        dists.append(dist)
    pts, cum = arc_table(dist)
    lap = cum[-1]

    hr_noise = 0.0
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx creator="StravaGPX" version="1.1" xmlns="http://www.topografix.com/GPX/1/1"'
        ' xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">',
        f'<metadata><time>{START:%Y-%m-%dT%H:%M:%SZ}</time></metadata>',
        "<trk><name>Example long run</name><type>running</type><trkseg>",
    ]
    for t, speed, dist in zip(times, speeds, dists):
        minutes = t / 60
        # Gentle rolling terrain along the loop.
        phase = 2 * math.pi * dist / lap
        ele = 2.0 + 2.0 * math.sin(2 * phase) + 1.0 * math.sin(5 * phase + 0.4)

        # Heart rate: settles to ~141 after the warm-up, then drifts ~17 bpm by the end.
        settle = 141 - 23 * math.exp(-minutes / 3.0)
        drift = 17 * max(0.0, minutes - 15) ** 1.2 / 60 ** 1.2
        hr_noise = 0.9 * hr_noise + rng.gauss(0, 0.6)
        hr = settle + drift + 1.2 * math.sin(2 * phase) + hr_noise

        cad = 85 + 0.6 * (speed - 2.98) * 10 + rng.gauss(0, 0.7)

        x, y = position(dist, pts, cum)
        x += rng.gauss(0, 0.8)
        y += rng.gauss(0, 0.8)
        lat = CENTRE[0] + y / METRES_PER_DEG_LAT
        lon = CENTRE[1] + x / METRES_PER_DEG_LON

        when = START + timedelta(seconds=t)
        out.append(
            f'<trkpt lat="{lat:.7f}" lon="{lon:.7f}"><ele>{ele:.1f}</ele>'
            f'<time>{when:%Y-%m-%dT%H:%M:%SZ}</time><extensions><gpxtpx:TrackPointExtension>'
            f'<gpxtpx:hr>{hr:.0f}</gpxtpx:hr><gpxtpx:cad>{cad:.0f}</gpxtpx:cad>'
            '</gpxtpx:TrackPointExtension></extensions></trkpt>'
        )
    out += ["</trkseg></trk></gpx>", ""]
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"wrote {os.path.relpath(OUT)}: {dist / 1000:.2f} km in one loop")


if __name__ == "__main__":
    main()
