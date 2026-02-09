import importlib
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
vis.plot_weekly(df_activities, col='distance', save_name='weekly_distance.png')
vis.plot_weekly(df_activities, col='pace', save_name='weekly_pace.png')
vis.plot_weekly(df_activities, col='risk', save_name='weekly_risk.png')

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

print("All plots updated and saved to the plots/ folder.")
print("DONE")
