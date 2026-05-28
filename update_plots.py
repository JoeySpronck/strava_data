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
    return {k: activity_dict[k] for k in col_names}

df_activities = pd.DataFrame([get_activity_data(a) for a in activities])

# --------------------------
# RESET PLOT FOLDER
# --------------------------
for file in os.listdir('plots'):
    os.remove(os.path.join('plots', file))

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

# This week
this_week_volume = df_weekly.loc[this_week, 'total_volume']
this_week_target = base_completed * 1.1
target_reached = this_week_volume >= this_week_target

# Target
week_target = this_week_target if not target_reached else max(this_week_volume, this_week_target)
week_target = round(float(week_target), 1)
week_ran = round(float(this_week_volume), 1)
if week_ran > week_target:
    week_target = round(week_ran * 1.1, 1)
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
def fetch_text_fields(client, ids):
    """Per-id get_activity call → DataFrame of {id, description, private_note}.
    Required because Strava's summary API doesn't return `private_note`."""
    rows = []
    for aid in ids:
        d = dict(client.get_activity(aid))
        rows.append({
            'id': aid,
            'description': d.get('description'),
            'private_note': d.get('private_note'),
        })
    return pd.DataFrame(rows)


KG_PATTERN = re.compile(r'(\d+)\s*kg', re.IGNORECASE)
VOLUME_PATTERN = re.compile(r'(\d{2,7})\s*kg\s*volume', re.IGNORECASE)


def _first_int_match(pattern, *texts):
    for t in texts:
        if not t:
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

        vis.plot_weekly_stacked(
            df_strength,
            stack_col='volume_kg',
            color_col='time_min',
            stack_label='Volume (kg)',
            color_label='Session time (min)',
            title='Weekly Strength Volume  |  Session Time',
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

overview_panels = []
if len(df_runs_overview) > 0:
    overview_panels.append(dict(
        df=df_runs_overview,
        stack_col='distance_km', color_col='avg_speed_kmh',
        stack_label='Run km', color_label='km/h',
        title='Running  |  Avg Speed',
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
        stack_col='volume_kg', color_col='time_min',
        stack_label='Volume kg', color_label='min',
        title='Strength  |  Session Time',
    ))

if overview_panels:
    vis.plot_weekly_stacked_multi(
        overview_panels,
        panel_height=1.5,
        save_name='weekly_overview_all_sports.png',
    )

print("All plots updated and saved to the plots/ folder.")
print("DONE")
