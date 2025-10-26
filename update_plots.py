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

# LOGIN
client = login()

# GET ACTIVITIES
activities_object = client.get_activities(limit=1000)
activities = list(activities_object)

def get_activity_data(activity):
    activity_dict = dict(activity)
    col_names = ['id','type', 'name', 'distance', 'moving_time', 'elapsed_time',
                 'total_elevation_gain', 'start_date', 'start_latlng', 'kilojoules',
                 'average_heartrate', 'max_heartrate', 'elev_high', 'elev_low',
                 'average_speed', 'max_speed']
    activity_data = {k: activity_dict[k] for k in col_names}
    return activity_data

activities_data = [get_activity_data(a) for a in activities]
df_activities = pd.DataFrame(activities_data)

# RESET PLOT FOLDER
for file in os.listdir('plots'):
    os.remove(os.path.join('plots', file))

# UPDATE PLOTS
# Stacked plots
vis.plot_weekly(df_activities, col='distance', save_name='weekly_distance.png')
vis.plot_weekly(df_activities, col='pace', save_name='weekly_pace.png')
vis.plot_weekly(df_activities, col='risk', save_name='weekly_risk.png')

# Weekly volume progression
vis.plot_weekly_distance_targets(df_activities, additional_weeks=4, save_name=f'weekly_distance_targets.png')

# Prepare last weeks data
df_activities['start_date'] = pd.to_datetime(df_activities['start_date'], utc=True)
df_runs = df_activities[
    (df_activities['type'] == "Run") &
    (df_activities['start_date'].dt.year >= 2025)
].copy()

df_runs['distance_km'] = df_runs['distance'] / 1000
df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)   

weekly = df_runs.groupby('week')['distance_km'].agg(
    total_volume='sum',
    long_run='max'
).sort_index()

now = pd.Timestamp.now(tz=df_runs['start_date'].dt.tz)
this_week = now.to_period('W-SUN').end_time

if this_week not in weekly.index:
    weekly.loc[this_week] = {'total_volume': 0, 'long_run': 0}
weekly = weekly.sort_index()

week_target = round(float(weekly.iloc[-2]['total_volume'] * 1.1), 1)
week_ran = round(float(weekly.loc[this_week]['total_volume'].sum()), 1)

if week_ran > week_target:
    week_target = round(week_ran * 1.1, 1)
    target_next_week = True
else:
    target_next_week = False

# Week plan plots
for runs in [3, 4, 5]:
    vis.plot_week_plan(week_target, runs, save_name=f'week_plan_{runs}_runs.png')

# Current week plan plots
for runs in [3, 4, 5]:
    vis.plot_current_week_plan(df_activities, week_target, runs=runs, target_next_week=target_next_week, save_name=f'current_week_plan_{runs}_runs.png')

print("All plots updated and saved to the plots/ folder.")
print("DONE")