import importlib
import re
import strava_data
import strava_data.authentication
import strava_data.visualization
importlib.reload(strava_data)
importlib.reload(strava_data.authentication)
importlib.reload(strava_data.visualization)
from strava_data.authentication import login
import strava_data.visualization as vis
import pandas as pd
import numpy as np
import os

vis.SHOW_PLOTS = False

# --------------------------
# LOGIN & GET ACTIVITIES
# --------------------------
client = login()
activities_object = client.get_activities(limit=1000)
activities = list(activities_object)

def get_activity_data(activity):
    activity_dict = dict(activity)
    col_names = ['id','type', 'name', 'distance', 'moving_time', 'elapsed_time',
                 'total_elevation_gain', 'start_date', 'start_latlng', 'kilojoules',
                 'average_heartrate', 'max_heartrate', 'elev_high', 'elev_low',
                 'average_speed', 'max_speed']
    row = {k: activity_dict[k] for k in col_names}
    # sport_type distinguishes trail runs (type is the legacy 'Run' for both); start_date_local
    # gives the correct calendar day. Both are optional in the summary payload, so .get them.
    row['sport_type'] = activity_dict.get('sport_type')
    row['start_date_local'] = activity_dict.get('start_date_local')
    return row

df_activities = pd.DataFrame([get_activity_data(a) for a in activities])

# --------------------------
# RESET PLOT FOLDER
# --------------------------
# Wipe and recreate so stale plots (and the month_plots/ subfolder) never linger.
import shutil
if os.path.isdir('plots'):
    shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

# --------------------------
# PREPARE RUN DATA
# --------------------------
df_activities['start_date'] = pd.to_datetime(df_activities['start_date'], utc=True)

# Filter runs in 2025+
df_runs = df_activities[
    (df_activities['type'] == "Run") &
    (df_activities['start_date'].dt.year >= 2025)
].copy()

df_runs['distance_km'] = df_runs['distance'] / 1000
df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)

# --------------------------
# WEEKLY AGGREGATES
# --------------------------
df_weekly = df_runs.groupby('week')['distance_km'].agg(
    total_volume='sum',
    long_run='max'
).sort_index()

# Ensure current week exists
now = pd.Timestamp.now(tz=df_runs['start_date'].dt.tz)
this_week = now.to_period('W-SUN').end_time
if this_week not in df_weekly.index:
    df_weekly.loc[this_week] = {'total_volume': 0, 'long_run': 0}
df_weekly = df_weekly.sort_index()

# --------------------------
# TARGET CALCULATIONS
# --------------------------
# Last 4 completed weeks (excluding this week)
completed_weeks = df_weekly.loc[df_weekly.index < this_week]
recent_completed = completed_weeks.iloc[-4:]
base_completed = recent_completed['total_volume'].max() if len(recent_completed) > 0 else 0

# Recovery ceiling: highest mean of any 3 consecutive recorded weeks over the last
# half year (~26 weeks) = a previously sustained volume. Build-up grows at +25%/week
# while staying under it; once a +25% step would reach it, growth reverts to +10%.
half_year = completed_weeks.iloc[-26:]
recovery_ceiling = half_year['total_volume'].rolling(3).mean().max()
recovery_ceiling = None if pd.isna(recovery_ceiling) else float(recovery_ceiling)

# This week
this_week_volume = df_weekly.loc[this_week, 'total_volume']
this_week_target = vis.grow_target(base_completed, recovery_ceiling)
target_reached = this_week_volume >= this_week_target

# Target
week_target = this_week_target if not target_reached else max(this_week_volume, this_week_target)
week_target = round(float(week_target), 1)
week_ran = round(float(this_week_volume), 1)
if week_ran > week_target:
    week_target = round(vis.grow_target(week_ran, recovery_ceiling), 1)
    target_next_week = True
else:
    target_next_week = False

# --------------------------
# UPDATE PLOTS
# --------------------------
# Stacked plots
vis.plot_weekly(df_runs, col='distance', save_name='weekly_distance.png')
vis.plot_weekly(df_runs, col='pace', save_name='weekly_pace.png')
vis.plot_weekly(df_runs, col='risk', save_name='weekly_risk.png')

# Weekly volume progression
vis.plot_weekly_distance_targets(
    df_weekly,
    week_target=week_target,
    this_week=this_week,
    this_week_target=this_week_target,
    this_week_volume=this_week_volume,
    target_reached=target_reached,
    recovery_ceiling=recovery_ceiling,
    additional_weeks=4,
    last_weeks=7,
    save_name='weekly_distance_targets.png'
)

# Week plan plots
for runs in [3, 4, 5]:
    vis.plot_week_plan(week_target, runs, save_name=f'week_plan_{runs}_runs.png')

# Current week plan plots
for runs in [3, 4, 5]:
    vis.plot_current_week_plan(
        df_runs,
        week_target,
        runs=runs,
        target_next_week=target_next_week,
        save_name=f'current_week_plan_{runs}_runs.png'
    )

# --------------------------
# HIKE & STRENGTH SUPPORT
# --------------------------
# Descriptions / private notes need a per-activity get_activity call (the summary API
# omits them). fetch_text_fields caches results to .cache/ keyed by id, so re-runs only
# hit the API for new activities — this is what keeps us under the rate limit.
from strava_data.activity_cache import fetch_text_fields

KG_PATTERN = re.compile(r'(\d+)\s*kg', re.IGNORECASE)
VOLUME_PATTERN = re.compile(r'(\d{2,7})\s*kg\s*volume', re.IGNORECASE)


def _first_int_match(pattern, *texts):
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        m = pattern.search(t)
        if m:
            return int(m.group(1))
    return None


# --------------------------
# HIKING PLOT
# --------------------------
df_hikes = df_activities[
    (df_activities['type'] == 'Hike') &
    (df_activities['start_date'].dt.year >= 2025)
].copy()

if len(df_hikes) > 0:
    df_hikes['distance_km'] = df_hikes['distance'] / 1000
    df_hikes['week'] = df_hikes['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)

    hike_text = fetch_text_fields(client, df_hikes['id'].tolist())
    df_hikes = df_hikes.merge(hike_text, on='id', how='left')
    df_hikes['weight_kg'] = df_hikes.apply(
        lambda r: _first_int_match(KG_PATTERN, r['name'], r.get('description'), r.get('private_note')) or 0,
        axis=1,
    )

    vis.plot_weekly_stacked(
        df_hikes,
        stack_col='distance_km',
        color_col='weight_kg',
        stack_label='Distance (km)',
        color_label='Carried weight (kg)',
        title='Weekly Hiking Distance  |  Carried Weight',
        save_name='weekly_hike_weight.png',
    )

# --------------------------
# STRENGTH PLOT
# --------------------------
# Cap the volume-rate colour scale (kg/min) so a couple of very dense sessions don't push
# every other bar/circle to one end of the colormap. Sessions above this clamp to max colour.
STRENGTH_COLOR_MAX = 350

df_strength = df_activities[
    (df_activities['type'] == 'WeightTraining') &
    (df_activities['start_date'].dt.year >= 2025)
].copy()

if len(df_strength) > 0:
    df_strength['week'] = df_strength['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)
    df_strength['time_min'] = df_strength['moving_time'] / 60

    strength_text = fetch_text_fields(client, df_strength['id'].tolist())
    df_strength = df_strength.merge(strength_text, on='id', how='left')
    df_strength['volume_kg'] = df_strength.apply(
        lambda r: _first_int_match(VOLUME_PATTERN, r.get('description'), r.get('private_note')),
        axis=1,
    )

    # Drop early sessions where volume wasn't logged
    df_strength = df_strength[df_strength['volume_kg'].notna()].copy()

    if len(df_strength) > 0:
        df_strength['volume_kg'] = df_strength['volume_kg'].astype(int)
        # Volume rate (kg lifted per minute) is the strength analogue of run/cycle avg speed:
        # an effort-density metric, so colouring stays consistent with the other sports.
        df_strength['volume_per_min'] = df_strength['volume_kg'] / df_strength['time_min']

        vis.plot_weekly_stacked(
            df_strength,
            stack_col='volume_kg',
            color_col='volume_per_min',
            stack_label='Volume (kg)',
            color_label='Volume rate (kg/min)',
            title='Weekly Strength Volume  |  Volume Rate',
            color_vmax=STRENGTH_COLOR_MAX,
            save_name='weekly_strength_volume.png',
        )

# --------------------------
# CYCLING PLOT
# --------------------------
df_rides = df_activities[
    (df_activities['type'] == 'Ride') &
    (df_activities['start_date'].dt.year >= 2025)
].copy()

if len(df_rides) > 0:
    df_rides['distance_km'] = df_rides['distance'] / 1000
    df_rides['week'] = df_rides['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)
    df_rides['avg_speed_kmh'] = df_rides['average_speed'] * 3.6

    vis.plot_weekly_stacked(
        df_rides,
        stack_col='distance_km',
        color_col='avg_speed_kmh',
        stack_label='Distance (km)',
        color_label='Avg speed (km/h)',
        title='Weekly Cycling Distance  |  Avg Speed',
        save_name='weekly_ride_speed.png',
    )

# --------------------------
# ALL-SPORTS OVERVIEW
# --------------------------
df_runs_overview = df_runs.copy()
df_runs_overview['avg_speed_kmh'] = df_runs_overview['average_speed'] * 3.6
# Trail runs share the legacy type 'Run'; sport_type tells them apart so they can be hatched.
df_runs_overview['is_trail'] = df_runs_overview['sport_type'].astype(str).str.contains('Trail', case=False, na=False)

overview_panels = []
if len(df_runs_overview) > 0:
    overview_panels.append(dict(
        df=df_runs_overview,
        stack_col='distance_km', color_col='avg_speed_kmh',
        stack_label='Run km', color_label='km/h',
        title='Running  |  Avg Speed',
        hatch_col='is_trail',
    ))
if len(df_rides) > 0:
    overview_panels.append(dict(
        df=df_rides,
        stack_col='distance_km', color_col='avg_speed_kmh',
        stack_label='Ride km', color_label='km/h',
        title='Cycling  |  Avg Speed',
    ))
if len(df_hikes) > 0:
    overview_panels.append(dict(
        df=df_hikes,
        stack_col='distance_km', color_col='weight_kg',
        stack_label='Hike km', color_label='kg',
        title='Hiking  |  Carried Weight',
    ))
if len(df_strength) > 0:
    overview_panels.append(dict(
        df=df_strength,
        stack_col='volume_kg', color_col='volume_per_min',
        stack_label='Volume kg', color_label='kg/min',
        title='Strength  |  Volume Rate',
        color_vmax=STRENGTH_COLOR_MAX,
    ))

if overview_panels:
    vis.plot_weekly_stacked_multi(
        overview_panels,
        panel_height=1.5,
        save_name='weekly_overview_all_sports.png',
    )

# --------------------------
# MONTH CALENDAR (Strava-style)
# --------------------------
# One row per activity: circle COLOR = its sport's color metric through the shared
# overview colormap (normalized within the sport); circle SIZE = the same magnitude that
# drives the overview bar heights (distance_km, or volume_kg for strength). The (size, color)
# metrics below mirror the overview panels exactly so the calendar reads consistently.
import matplotlib.colors as mcolors

cal_cmap = vis.metric_colormap()  # shared dark→white→orange metric colormap


def _metric_colors(values, vmin=None, vmax=None):
    """RGBA per value via the shared colormap, normalized within this sport's range.

    vmin / vmax override the data-derived range to cap the scale so a couple of extreme
    activities don't push every other circle to one end of the colormap; out-of-range
    values clamp to the end colors (clip=True).
    """
    v = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return [cal_cmap(0.5)] * len(v)
    lo = float(finite.min()) if vmin is None else float(vmin)
    hi = float(finite.max()) if vmax is None else float(vmax)
    if hi <= lo:
        return [cal_cmap(0.5)] * len(v)
    norm = mcolors.Normalize(lo, hi, clip=True)
    return [cal_cmap(norm(x)) if np.isfinite(x) else cal_cmap(0.5) for x in v]


def _cal_date(row):
    """Local calendar day for an activity (start_date_local if present, else UTC start_date)."""
    d = row.get('start_date_local')
    if pd.isna(d):
        d = row['start_date']
    return pd.Timestamp(d).replace(tzinfo=None)


cal_rows = []

# Runs (incl. trail runs): size = distance_km, color = avg speed. Letter T for trail else R.
if len(df_runs) > 0:
    runs_cal = df_runs.copy()
    runs_cal['avg_speed_kmh'] = runs_cal['average_speed'] * 3.6
    is_trail = runs_cal['sport_type'].astype(str).str.contains('Trail', case=False, na=False)
    colors = _metric_colors(runs_cal['avg_speed_kmh'])
    for (_, row), color, trail in zip(runs_cal.iterrows(), colors, is_trail):
        cal_rows.append(dict(
            date=_cal_date(row), sport='trail' if trail else 'run',
            size_value=row['distance_km'], color=color,
        ))

# Cycling: size = distance_km, color = avg speed.
if len(df_rides) > 0:
    colors = _metric_colors(df_rides['avg_speed_kmh'])
    for (_, row), color in zip(df_rides.iterrows(), colors):
        cal_rows.append(dict(
            date=_cal_date(row), sport='bike',
            size_value=row['distance_km'], color=color,
        ))

# Hiking: size = distance_km, color = carried weight.
if len(df_hikes) > 0:
    colors = _metric_colors(df_hikes['weight_kg'])
    for (_, row), color in zip(df_hikes.iterrows(), colors):
        cal_rows.append(dict(
            date=_cal_date(row), sport='hike',
            size_value=row['distance_km'], color=color,
        ))

# Strength: size = volume_kg, color = volume rate (kg/min), capped so outliers don't skew.
if len(df_strength) > 0:
    colors = _metric_colors(df_strength['volume_per_min'], vmax=STRENGTH_COLOR_MAX)
    for (_, row), color in zip(df_strength.iterrows(), colors):
        cal_rows.append(dict(
            date=_cal_date(row), sport='strength',
            size_value=row['volume_kg'], color=color,
        ))

df_cal = pd.DataFrame(cal_rows)

if len(df_cal) > 0:
    df_cal['date'] = pd.to_datetime(df_cal['date'])
    # Non-empty months present in the data, oldest → newest (skip months with no activity).
    months = sorted({(d.year, d.month) for d in df_cal['date']})

    month_dir = os.path.join('plots', 'month_plots')
    os.makedirs(month_dir, exist_ok=True)

    # One calendar per non-empty month → plots/month_plots/YYYY-MM.png
    for (y, m) in months:
        vis.plot_month_calendar(
            df_cal, year=y, month=m,
            save_name=os.path.join('month_plots', f'{y}-{m:02d}.png'),
        )

    # Index README for the month_plots folder — latest month on top, oldest at the bottom.
    lines = [
        "# Monthly Activity Calendars",
        "",
        "Auto-generated by `update_plots.py`. Latest month on top.",
        "",
    ]
    for (y, m) in reversed(months):
        label = pd.Timestamp(year=y, month=m, day=1).strftime('%B %Y')
        lines += [f"### {label}", "", f"![{label}](plots/month_plots/{y}-{m:02d}.png)", ""]
    with open('CALENDAR_PLOTS.md', 'w') as f:
        f.write('\n'.join(lines).rstrip() + '\n')

    # Stable filenames for the root README: the two most recent non-empty months.
    latest_y, latest_m = months[-1]
    vis.plot_month_calendar(df_cal, year=latest_y, month=latest_m, save_name='month_calendar.png')
    if len(months) >= 2:
        prev_y, prev_m = months[-2]
        vis.plot_month_calendar(df_cal, year=prev_y, month=prev_m, save_name='month_calendar_prev.png')

print("All plots updated and saved to the plots/ folder.")
print("DONE")
