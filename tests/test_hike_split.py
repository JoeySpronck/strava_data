"""The run/hike split, the multisport links and the link markers on the stacked plots.

All data here is synthetic: a run built in memory as 1 Hz streams (run, a low-cadence
climb, run again), shaped like what Strava returns for a tagged trail run.

Run it with::

    python tests/test_hike_split.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from strava_data import hike_split as hs  # noqa: E402
from strava_data import visualization as vis  # noqa: E402

RUN_SPEED, HIKE_SPEED = 3.0, 1.0   # m/s
RUN_CAD, HIKE_CAD = 85.0, 55.0     # one-leg spm


def synthetic_streams(run1=1200, hike=900, run2=1200, pause=60, cadence=True, seed=0):
    """Seconds of running, hiking, running; a standing pause inside the hike."""
    rng = np.random.default_rng(seed)
    speed = np.r_[np.full(run1, RUN_SPEED), np.full(hike, HIKE_SPEED), np.full(run2, RUN_SPEED)]
    cad = np.r_[np.full(run1, RUN_CAD), np.full(hike, HIKE_CAD), np.full(run2, RUN_CAD)]
    cad = cad + rng.normal(0, 3, cad.size)
    cad[rng.choice(cad.size, 40, replace=False)] = 0      # sensor dropouts
    cad[run1 + 100: run1 + 110] = RUN_CAD                  # a few running steps on the climb
    moving = np.ones(speed.size, dtype=bool)
    p0 = run1 + hike // 2
    speed[p0:p0 + pause] = 0
    moving[p0:p0 + pause] = False
    cad[p0:p0 + pause] = 0
    t = np.arange(speed.size, dtype=float)
    dist = np.r_[0, np.cumsum(speed[1:])]
    out = {"time": t.tolist(), "distance": dist.tolist(), "moving": moving.tolist(),
           "velocity_smooth": speed.tolist(), "heartrate": (140 + 10 * (speed > 2)).tolist()}
    if cadence:
        out["cadence"] = cad.tolist()
    return out


def summary_row(streams, **kw):
    d = streams["distance"][-1]
    moving_t = float(sum(streams["moving"][1:]))
    row = dict(id=1, type="Run", sport_type="TrailRun", name="Hilly one", distance=d,
               moving_time=moving_t, elapsed_time=float(len(streams["time"])),
               average_speed=d / moving_t, average_heartrate=145.0,
               total_elevation_gain=300.0, start_date=pd.Timestamp("2025-06-04 07:00", tz="UTC"),
               start_date_local=pd.Timestamp("2025-06-04 09:00"),
               description="Legs heavy, 20% hike", private_note=None)
    row.update(kw)
    return row


def test_parse_tag():
    assert hs.parse_hike_percent("hard one, 30% hike") == 30
    assert hs.parse_hike_percent(None, "", "~25 % hiking up the hill") == 25
    assert hs.parse_hike_percent("40%hiked") == 40
    assert hs.parse_hike_percent("walked the uphills") is None
    assert hs.parse_hike_percent("90% effort, hiked a bit") is None
    assert hs.parse_hike_percent("100% hike") is None      # >2 digits: not a partial hike
    assert hs.parse_hike_percent("0% hike") is None


def test_multisport_tag():
    for text in ["multisport", "Multi sport day", "brick, multi-sport", "MULTISPORT"]:
        assert hs.has_multisport_tag(None, text), text
    for text in ["multisports", "multi  sport", "sport", "multi", None]:
        assert not hs.has_multisport_tag(text), text


def test_cadence_split():
    s = synthetic_streams()
    r = hs.classify_streams(s, hike_percent=20)
    assert r["method"] == "cadence"
    assert 60 <= r["threshold"] <= 78
    # 900 s of hiking at 1 m/s minus the 60 s pause = 840 m of the 8040 m
    assert abs(r["hike_distance"] - 840) < 15, r
    assert abs(r["hike_time"] - 840) < 15, r
    assert abs(r["run_time"] - 2400) < 15, r
    assert abs(r["run_distance"] / r["run_time"] - RUN_SPEED) < 0.05
    assert r["run_hr"] > r["hike_hr"]


def test_split_run_keeps_totals_and_warns():
    s = synthetic_streams()
    row = summary_row(s)
    warnings = []
    run, hike = hs.split_run(row, 30, s, warn=warnings.append)
    assert abs(run["distance"] + hike["distance"] - row["distance"]) < 1e-6
    assert abs(run["moving_time"] + hike["moving_time"] - row["moving_time"]) < 1e-6
    assert hike["type"] == "Hike" and run["type"] == "Run"
    assert run["sport_type"] == "TrailRun"
    assert abs(run["average_speed"] - RUN_SPEED) < 0.05
    assert run["total_elevation_gain"] is None
    # 10% hiked vs a 30% tag: more than 15 %-points off -> one warning, split still used.
    assert len(warnings) == 1 and "tagged 30%" in warnings[0]
    assert hs.split_run(row, 12, s, warn=warnings.append) is not None
    assert len(warnings) == 1


def test_pace_fallback_without_cadence():
    s = synthetic_streams(cadence=False)
    r = hs.classify_streams(s, hike_percent=10)
    assert r["method"] == "pace"
    # The slowest 10% of distance is exactly the hiked stretch here.
    assert abs(r["hike_fraction"] - 0.10) < 0.01
    assert abs(r["run_distance"] / r["run_time"] - RUN_SPEED) < 0.05
    assert hs.classify_streams(s, hike_percent=None) is None


def test_percentage_fallback_without_stream():
    row = summary_row(synthetic_streams())
    warnings = []
    run, hike = hs.split_run(row, 25, None, warn=warnings.append)
    assert abs(hike["distance"] - 0.25 * row["distance"]) < 1e-6
    assert abs(run["average_speed"] - row["average_speed"]) < 1e-9   # keeps original pace
    assert abs(run["moving_time"] + hike["moving_time"] - row["moving_time"]) < 1e-6
    assert warnings and "no usable stream" in warnings[0]


def test_split_activities_and_markers():
    s = synthetic_streams()
    tagged = summary_row(s)
    untagged = summary_row(s, id=2, description="walked the uphills",
                           start_date=pd.Timestamp("2025-06-05 07:00", tz="UTC"),
                           start_date_local=pd.Timestamp("2025-06-05 09:00"))
    ride = dict(tagged, id=3, type="Ride", sport_type="Ride", description=None,
                private_note="multisport", start_date=pd.Timestamp("2025-06-06 07:00", tz="UTC"),
                start_date_local=pd.Timestamp("2025-06-06 09:00"))
    run_ms = dict(ride, id=4, type="Run", sport_type="Run",
                  start_date=pd.Timestamp("2025-06-06 09:00", tz="UTC"),
                  start_date_local=pd.Timestamp("2025-06-06 11:00"))
    lone_ms = dict(ride, id=5, start_date=pd.Timestamp("2025-06-07 09:00", tz="UTC"),
                   start_date_local=pd.Timestamp("2025-06-07 11:00"))
    next_week = dict(tagged, id=6, start_date=pd.Timestamp("2025-06-10 07:00", tz="UTC"),
                     start_date_local=pd.Timestamp("2025-06-10 09:00"))
    df = pd.DataFrame([tagged, untagged, ride, run_ms, lone_ms, next_week])

    out = hs.split_activities(df, {1: s, 6: s}, warn=lambda m: None)
    assert len(out) == 8
    assert list(out[out["id"] == 1]["type"]) == ["Run", "Hike"]
    assert (out["id"] == 2).sum() == 1           # no percentage -> untouched

    out = hs.assign_link_markers(out)
    m = dict(zip(zip(out["id"], out["type"]), out["link_marker"]))
    assert (m[(1, "Run")], m[(1, "Hike")]) == ("<H>", "<T>")   # split: both ways
    # 65 min apart, over the gap limit, but both tagged multisport.
    assert (m[(3, "Ride")], m[(4, "Run")]) == ("R>", "<B")
    assert m[(5, "Ride")] is None and m[(2, "Run")] is None
    assert (m[(6, "Run")], m[(6, "Hike")]) == ("<H>", "<T>")


def test_gap_links_and_chains():
    def act(aid, typ, start, minutes, sport_type=None, note=None):
        t = pd.Timestamp(f"2025-06-04 {start}", tz="UTC")
        return dict(id=aid, type=typ, sport_type=sport_type or typ, name="x", description=None,
                    private_note=note, start_date=t, start_date_local=t.tz_localize(None),
                    elapsed_time=minutes * 60.0, moving_time=minutes * 60.0)
    df = pd.DataFrame([
        act(1, "Ride", "07:00", 60),                     # ends 08:00
        act(2, "Run", "08:30", 30),                      # 30 min gap -> linked
        act(3, "Hike", "09:30", 60),                     # 60 min gap -> linked (limit)
        act(4, "Swim", "10:40", 20),                     # no letter: ignored
        act(5, "WeightTraining", "12:31", 10),           # 61 min after the hike -> not linked
        act(6, "Run", "18:00", 30, sport_type="TrailRun"),
    ])
    m = dict(zip(hs.assign_link_markers(df)["id"], hs.assign_link_markers(df)["link_marker"]))
    assert (m[1], m[2], m[3]) == ("R>", "H>", "<R")   # middle one points to the next
    assert m[4] is None and m[5] is None and m[6] is None
    # A trail run gets T; strength before it within the gap links.
    df.loc[df["id"] == 5, "start_date"] = pd.Timestamp("2025-06-04 17:30", tz="UTC")
    m = dict(zip(hs.assign_link_markers(df)["id"], hs.assign_link_markers(df)["link_marker"]))
    assert (m[5], m[6]) == ("T>", "<S")


def test_marker_placement():
    """Glyph sits mid-bar, its top a margin below the segment top; centred on short bars."""
    import matplotlib.pyplot as plt
    week = pd.Timestamp("2025-06-08")
    df = pd.DataFrame({"week": [week, week, week + pd.Timedelta(days=7)],
                       "km": [30.0, 0.3, 8.0], "c": [1.0, 2.0, 3.0],
                       "link_marker": ["H>", "<R>", None]})
    fig, ax = plt.subplots(figsize=(8, 4))
    _, _, segments = vis._draw_weekly_stacked(ax, df, "km", "c")
    fig.tight_layout()
    # A year of weeks on the axis, so bars are as narrow as in the real plots.
    ax.set_xlim(vis.mdates.date2num(week) - 180, vis.mdates.date2num(week) + 180)
    vis._draw_link_markers(ax, segments)
    fig.canvas.draw()
    marks = [p for p in ax.patches if p.get_gid() == "link_marker"]
    assert len(marks) == 2
    to_px = ax.transData.transform
    x0 = to_px((vis.mdates.date2num(week), 0))[0]
    bar_px = abs(to_px((vis.mdates.date2num(week) + vis.STYLE["bar_width"], 0))[0] - x0)
    # Each marker's path is centred on (0, 0), so its transform maps that to the centre.
    x_px, y_px = marks[0].get_transform().transform((0, 0))
    assert abs(x_px - x0) < 1e-6
    top_px = to_px((0, 30.0))[1]
    glyph_px = vis.STYLE["link_marker_size"] * bar_px
    expected = vis.STYLE["link_marker_top_margin"] * bar_px + glyph_px / 2
    assert abs((top_px - y_px) - expected) < 0.5
    y2_px = marks[1].get_transform().transform((0, 0))[1]
    assert abs(y2_px - to_px((0, 30.15))[1]) < 1e-6   # 0.3 km segment: vertical centre
    assert not marks[0].get_snap()
    arrows = [p for p in ax.patches if p.get_gid() == "link_arrow"]
    assert len(arrows) == 2
    # One- and two-headed arrows are equally tall, and sit above the letter.
    heights = [a.get_path().get_extents().height for a in arrows]
    assert abs(heights[0] - heights[1]) < 1e-9
    for mark, arrow in zip(marks, arrows):
        assert arrow.get_path().get_extents().y0 > mark.get_path().get_extents().y1
    plt.close(fig)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
