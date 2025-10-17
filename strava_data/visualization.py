import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import numpy as np

def plot_weekly_col_distance(df_activities: pd.DataFrame):
    # --- Prepare base data ---
    df_activities['start_date'] = pd.to_datetime(df_activities['start_date'], utc=True)
    df_runs = df_activities[(df_activities['type'] == "Run") &
                            (df_activities['start_date'].dt.year >= 2025)].copy()

    df_runs['distance_km'] = df_runs['distance'] / 1000
    df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)

    # --- Sequential colormap: short (dark gray) → medium (white) → long (orange) ---
    colors = ['#555555', '#FFFFFF', '#FC7B03']  # dark gray → white → orange
    cmap = mcolors.LinearSegmentedColormap.from_list('gray_white_orange', colors)

    # Linear normalization
    vmin = df_runs['distance_km'].min()
    vmax = df_runs['distance_km'].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- Plot styling ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # --- Plot stacked bars ---
    for week, group in df_runs.groupby('week'):
        bottom = 0
        for _, row in group.iterrows():
            color = cmap(norm(row['distance_km']))
            ax.bar(week, row['distance_km'], bottom=bottom, width=4,
                color=color, linewidth=0.6, edgecolor='black')
            bottom += row['distance_km']

    # --- Axes formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', colors='gray', rotation=45)
    ax.tick_params(axis='y', colors='gray')
    ax.set_xlabel('Month', color='gray', labelpad=8)
    ax.set_ylabel('Distance (km)', color='gray', labelpad=8)
    ax.set_title('Weekly Distance Stacked per Run  |  Distance',
                color='#FC7B03', fontsize=14, weight='bold', pad=15)
    ax.grid(axis='y', color='gray', alpha=0.2, linewidth=0.5)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Distance per run (km)', color='gray')
    cbar.ax.yaxis.set_tick_params(color='gray')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='gray')

    # --- Layout ---
    plt.tight_layout()
    plt.show()

def plot_weekly_col_pace(df_activities: pd.DataFrame):
    # --- Prepare base data ---
    df_activities['start_date'] = pd.to_datetime(df_activities['start_date'], utc=True)
    df_runs = df_activities[(df_activities['type'] == "Run") &
                            (df_activities['start_date'].dt.year >= 2025)].copy()

    df_runs['distance_km'] = df_runs['distance'] / 1000
    df_runs['pace_min_per_km'] = (1000 / df_runs['average_speed']) / 60
    df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)

    # --- Custom diverging colormap for pace ---
    colors = ['#FC7B03', '#FFFFFF', '#555555']  # slow -> neutral -> fast
    cmap = mcolors.LinearSegmentedColormap.from_list('gray_white_orange', colors)

    # Center at average pace
    avg_pace = df_runs['pace_min_per_km'].mean()
    vmin = df_runs['pace_min_per_km'].min()
    vmax = df_runs['pace_min_per_km'].max()
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=avg_pace, vmax=vmax)

    # --- Plot styling ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # --- Plot stacked bars ---
    for week, group in df_runs.groupby('week'):
        bottom = 0
        for _, row in group.iterrows():
            color = cmap(divnorm(row['pace_min_per_km']))
            ax.bar(week, row['distance_km'], bottom=bottom, width=5,
                color=color, linewidth=0.6, edgecolor='black')
            bottom += row['distance_km']

    # --- Axes formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', colors='gray', rotation=45)
    ax.tick_params(axis='y', colors='gray')
    ax.set_xlabel('Month', color='gray', labelpad=8)
    ax.set_ylabel('Distance (km)', color='gray', labelpad=8)
    ax.set_title('Weekly Distance Stacked per Run  |  Pace',
                color='#FC7B03', fontsize=14, weight='bold', pad=15)
    ax.grid(axis='y', color='gray', alpha=0.2, linewidth=0.5)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=divnorm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Pace (min/km)', color='gray')
    cbar.ax.yaxis.set_tick_params(color='gray')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='gray')

    # Format ticks as mm:ss
    def format_pace(x, pos):
        mins = int(x)
        secs = int((x - mins) * 60)
        return f"{mins}:{secs:02d}"

    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_pace))

    # --- Layout ---
    plt.tight_layout()
    plt.show()

def plot_weekly_col_risk(df_activities: pd.DataFrame):
    # --- Prepare base data ---
    df_activities['start_date'] = pd.to_datetime(df_activities['start_date'], utc=True)
    df_runs = df_activities[(df_activities['type'] == "Run") &
                            (df_activities['start_date'].dt.year >= 2025)].copy()

    df_runs['distance_km'] = df_runs['distance'] / 1000
    df_runs['pace_min_per_km'] = (1000 / df_runs['average_speed']) / 60
    df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)

    # --- Compute z-score risk metric ---
    scaler = StandardScaler()
    df_runs[['dist_z', 'speed_z']] = scaler.fit_transform(
        df_runs[['distance_km', 'average_speed']]
    )
    df_runs['risk_zscore'] = df_runs['dist_z'] + df_runs['speed_z']

    # --- Custom diverging colormap ---
    colors = ['#555555', '#FFFFFF', '#FC7B03']  # low, zero, high
    cmap = mcolors.LinearSegmentedColormap.from_list('gray_white_orange', colors)

    # Center normalization at 0
    vmin = df_runs['risk_zscore'].min()
    vmax = df_runs['risk_zscore'].max()
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # --- Plot styling ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12,6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # --- Plot stacked weekly bars ---
    for week, group in df_runs.groupby('week', sort=True):
        bottom = 0
        for _, row in group.iterrows():
            color = cmap(divnorm(row['risk_zscore']))
            ax.bar(week, row['distance_km'], bottom=bottom, width=5,
                color=color, linewidth=0.6, edgecolor='black')
            bottom += row['distance_km']

    # --- Axes formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', colors='gray', rotation=45)
    ax.tick_params(axis='y', colors='gray')
    ax.set_xlabel('Month', color='gray', labelpad=8)
    ax.set_ylabel('Distance (km)', color='gray', labelpad=8)
    ax.set_title('Weekly Distance Stacked per Run  |  Risk Z-score',
                color='#FC7B03', fontsize=14, weight='bold', pad=15)
    ax.grid(axis='y', color='gray', alpha=0.2, linewidth=0.5)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=divnorm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Risk Z-score', color='gray')
    cbar.ax.yaxis.set_tick_params(color='gray')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='gray')

    # --- Layout ---
    plt.tight_layout()
    plt.show()

def plot_weekly_distance_targets(df_activities: pd.DataFrame):
    # --- Weekly aggregation ---
    df_runs = df_activities[(df_activities['type'] == "Run") &
                            (df_activities['start_date'].dt.year >= 2025)].copy()
    df_runs['distance_km'] = df_runs['distance'] / 1000
    df_runs['start_date'] = pd.to_datetime(df_runs['start_date'], utc=True)
    df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)   

    weekly = df_runs.groupby('week')['distance_km'].agg(
        total_volume='sum',
        long_run='max'
    ).sort_index()

    # --- Identify current and next week ---
    now = pd.Timestamp.now(tz=df_runs['start_date'].dt.tz)
    this_week = now.to_period('W-SUN').end_time
    next_week = this_week + pd.Timedelta(days=7)

    # --- Ensure current week is present (even if 0 runs) ---
    if this_week not in weekly.index:
        weekly.loc[this_week] = {'total_volume': 0, 'long_run': 0}
    weekly = weekly.sort_index()

    # --- Determine last week and volumes ---
    last_week = weekly.index[-2] if len(weekly) >= 2 else None
    this_week_volume = weekly.loc[this_week, 'total_volume']
    last_week_volume = weekly.loc[last_week, 'total_volume'] if last_week else 0

    # --- 10% rule targets ---
    this_week_target = last_week_volume * 1.1
    next_week_target = this_week_target * 1.1

    # --- Prepare recent weeks ---
    last_x_weeks = 7
    recent_weeks = weekly.iloc[-last_x_weeks:].copy()

    # --- Plot ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # --- Plot past completed weeks ---
    past_weeks = recent_weeks.loc[recent_weeks.index < this_week]
    ax.bar(past_weeks.index, past_weeks['total_volume'],
        width=5, color='white', label='Completed Weeks', linewidth=0.6, edgecolor='black')
    # --- Plot current week ---
    ax.bar(this_week, this_week_volume, width=5, color='white', label='This Week (Progress)', linewidth=0.6, edgecolor='black')
    # --- Remaining target section ---
    remaining = max(this_week_target - this_week_volume, 0)
    if remaining > 0:
        ax.bar(this_week, remaining, bottom=this_week_volume, width=5,
            color='#fc7b03', label='Remaining to Target')
    # --- Next week target ---
    ax.bar(next_week, next_week_target, width=5,
        color='#fc7b03', alpha=0.9, label='Next Week Target', linewidth=0.6, edgecolor='black')

    # --- Add numeric labels ---
    def add_label(x, y, text, color='white', offset=0.5, inside=False, fontsize=9):
        va = 'top' if inside else 'bottom'
        y_pos = y - (offset + 0.1) if inside else y + offset
        ax.text(x, y_pos, text, ha='center', va=va, fontsize=fontsize, color=color, weight='bold')

    # Past weeks
    for week, row in past_weeks.iterrows():
        add_label(week, row['total_volume'], f"{row['total_volume']:.1f}", color='gray')

    # This week labels
    if this_week_volume > 0:
        add_label(this_week, this_week_volume, f"{this_week_volume:.1f}", inside=True, color='black')

    if remaining > 0:
        add_label(this_week, this_week_volume + remaining, f"{remaining:.1f}", inside=True, color='black')
        add_label(this_week, this_week_volume + remaining, f"{this_week_target:.1f}", inside=False, color='white')
    else:
        add_label(this_week, this_week_volume, f"{this_week_volume:.1f}", inside=False, color='white')

    # Next week label
    add_label(next_week, next_week_target, f"{next_week_target:.1f}", color='white')

    # --- Formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45, ha='right', color='gray', rotation_mode='anchor')
    plt.yticks(color='gray')
    plt.ylim(0, max(recent_weeks['total_volume'].max(), next_week_target) * 1.2)
    plt.ylabel('Weekly Distance (km)', color='gray')
    plt.title('Weekly Running Volume (10% Growth Targets)', color='#fc7b03', weight='bold')
    # plt.grid(axis='y', alpha=0.2, color='gray')
    # plt.legend(facecolor='black', labelcolor='white', loc='upper left')
    plt.tight_layout()
    plt.show()

def barplot(values, title=None, y_label="Distance (km)", highlight_list=None, sub_title=None):
    """
    Plots a Strava-styled bar chart with numeric labels.
    
    Parameters:
        values (array-like): numeric values for each bar
        title (str, optional): plot title
        y_label (str): y-axis label
        highlight_list (array-like, optional): same length as values,
            with 1 marking bars in orange, 0 in white
    """
    x = np.arange(len(values))
    total = np.sum(values)
    max_val = max(values) if len(values) > 0 else 0

    # --- Colors setup ---
    if highlight_list is None:
        colors = ['white'] * len(values)
    else:
        colors = ['#fc7b03' if h == 1 else 'white' for h in highlight_list]

    # --- Styling setup ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # --- Bars ---
    ax.bar(x, values, color=colors, width=0.7)

    # --- Labels on bars ---
    for i, v in enumerate(values):
        if v > 0:
            ax.text(
                i, v + max_val * 0.03, f"{v:.1f}",
                ha='center', va='bottom', color='gray', fontsize=9, weight='bold'
            )

    # --- Title and axes ---
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'd{i+1}' for i in x], color='gray')
    ax.set_ylabel(y_label, color='gray')
    if title is None:
        title = f"Week Total: {round(total, 1)} km"
    fig.suptitle(title, color='#fc7b03', weight='bold', fontsize=11)
    if sub_title is not None:
        ax.set_title(sub_title, color='gray', fontsize=10, weight='bold')

    # --- Grid & styling ---
    ax.tick_params(axis='y', colors='gray')
    plt.tight_layout()
    plt.show()

def target_to_proportions(target_km, target_proportions):
    return target_proportions / sum(target_proportions) * target_km

def plot_week_plan(target_km, runs=4):
    if runs==3:
        target_proportions = np.array([3. , 0. , 6. , 0. , 0. , 9. , 0. ])
    elif runs==4:
        target_proportions = np.array([3. , 6. , 0. , 4.5, 0. , 9. , 0. ])
    elif runs==5:
        target_proportions = np.array([ 4. ,  7.5,  0. ,  5.5,  3.5, 12. ,  0. ])
    else:
        raise ValueError("Only 3, 4 or 5 runs supported.")
    values = target_to_proportions(target_km, target_proportions)
    barplot(values, title=f"Weekly Plan for {target_km} km ({runs} runs)")

def remaining_week_kms(runs_ran, target_km, runs=4):
    # --- Planned distribution ---
    ### I got these from a youtube video for planning 3, 4 or 5 runs per week, numbers in the distribution represent miles
    if runs == 3:
        target_proportions = np.array([3., 0., 6., 0., 0., 9., 0.])
    elif runs == 4:
        target_proportions = np.array([3., 6., 0., 4.5, 0., 9., 0.])
    elif runs == 5:
        target_proportions = np.array([4., 7.5, 0., 5.5, 3.5, 12., 0.])
    else:
        raise ValueError("Only 3, 4 or 5 runs supported.")
    
    planned_runs = target_proportions[target_proportions > 0].copy()  # filter zeros

    # --- Match actual runs to closest planned runs ---
    remaining_planned = planned_runs.copy()
    for run in runs_ran:
        if len(remaining_planned) == 0:
            break
        idx = np.argmin(np.abs(remaining_planned - run))
        remaining_planned = np.delete(remaining_planned, idx)
    
    # --- Calculate remaining distance to hit target ---
    remaining_km = target_km - sum(runs_ran)
    if remaining_km <= 0 or len(remaining_planned) == 0:
        return np.zeros_like(remaining_planned)
    
    # --- Rescale remaining planned runs proportionally ---
    remaining_scaled = remaining_planned / remaining_planned.sum() * remaining_km
    return remaining_scaled

# def plot_current_week_plan(df_activities, week_target, runs=4, exclude_today=True):
#     """
#     Plot a weekly plan with completed and planned runs for the current week.

#     - Completed runs are shown in white.
#     - Planned (remaining) runs are shown in orange.
#     - Empty (rest) days remain visible.
#     - Only future days of the current week are used for remaining runs.
#     """

#     # --- Determine current week data ---
#     s_weeks = df_activities['start_date'].dt.to_period('W-SUN')
#     df_week = df_activities[s_weeks == s_weeks.max()].copy()
#     df_week['day_of_week'] = df_week['start_date'].dt.day_of_week  # Monday=0, Sunday=6

#     # --- Fill actual distances into 7-day array ---
#     current_week = np.zeros(7)
#     for _, row in df_week.iterrows():
#         current_week[int(row['day_of_week'])] += row['distance'] / 1000

#     # --- Identify current day ---
#     current_day = pd.Timestamp.now().day_of_week  # Monday=0
#     if exclude_today:
#         days_left = np.arange(current_day + 1, 7)
#     else:
#         days_left = np.arange(current_day, 7)
#     if len(days_left) == 0:
#         print("Week is over — no remaining days to plan.")
#         return

#     # --- Get actual runs ---
#     runs_ran = list(df_week['distance'] / 1000)
#     done_km = sum(runs_ran)

#     # --- Compute remaining runs using your matching logic ---
#     remaining = remaining_week_kms(runs_ran, week_target, runs=runs)

#     # --- Assign remaining runs to remaining days ---
#     plan = current_week.copy()
#     highlight = np.zeros(7)
#     remaining_days = list(days_left)
#     n_remaining_runs = len(remaining)
#     if len(remaining_days) < n_remaining_runs:
#         print(f"IMPOSSIBLE: Not enough days left in the week to schedule all {n_remaining_runs} remaining runs. Only {len(remaining_days)} days left.")
#         return

#     if n_remaining_runs > 0:
#         # choose spaced days within remaining ones
#         step = max(1, len(remaining_days) // n_remaining_runs)
#         assign_days = remaining_days[::step][:n_remaining_runs]

#         for i, day in enumerate(assign_days):
#             plan[day] = remaining[i]
#             highlight[day] = 1

#     # --- Combine actual and planned ---
#     highlight = np.where(plan > 0, highlight, 0)  # ensure only planned runs orange

#     # --- Plot ---
#     barplot(
#         plan,
#         title=f"Week Plan | {runs} runs, target {week_target} km)",
#         sub_title=f"{done_km:.1f} km done, {sum(remaining):.1f} km to go",
#         highlight_list=highlight
#     )

def plot_current_week_plan(df_activities, week_target, runs=4, exclude_today=True):
    """
    Plot a realistic weekly running plan considering:
    - Past runs (no back-to-back if avoidable)
    - Even spacing between future runs
    - Rest days in between if possible
    """

    # --- Determine current week data ---
    s_weeks = df_activities['start_date'].dt.to_period('W-SUN')
    df_week = df_activities[s_weeks == s_weeks.max()].copy()
    df_week['day_of_week'] = df_week['start_date'].dt.day_of_week  # Monday=0, Sunday=6

    # --- Fill actual distances into 7-day array ---
    current_week = np.zeros(7)
    for _, row in df_week.iterrows():
        current_week[int(row['day_of_week'])] += row['distance'] / 1000

    # --- Identify current day ---
    current_day = pd.Timestamp.now().day_of_week
    days_left = np.arange(current_day + 1 if exclude_today else current_day, 7)
    if len(days_left) == 0:
        print("Week is over — no remaining days to plan.")
        return

    # --- Compute current and remaining runs ---
    runs_ran = list(df_week['distance'] / 1000)
    done_km = sum(runs_ran)
    remaining = remaining_week_kms(runs_ran, week_target, runs=runs)
    n_remaining_runs = len(remaining)

    if n_remaining_runs == 0:
        print("No runs remaining — goal already met!")
        return
    if len(days_left) < n_remaining_runs:
        print(f"IMPOSSIBLE: Only {len(days_left)} days left, need {n_remaining_runs} runs.")
        return

    # --- Identify already-run days and possible rest days ---
    run_days = np.where(current_week > 0)[0]
    available_days = list(days_left)

    # --- Smart placement: prioritize spacing after last run ---
    plan = current_week.copy()
    highlight = np.zeros(7)

    # Determine the last day you ran (so we start planning after a rest if possible)
    last_run_day = run_days.max() if len(run_days) > 0 else -2
    min_gap = 1  # Prefer at least 1 rest day after each run

    assign_days = []

    # First, choose days separated by at least one rest day if possible
    for day in available_days:
        if len(assign_days) >= n_remaining_runs:
            break

        # Skip if this day is right after the last run or a newly scheduled one
        if any(abs(day - d) <= min_gap for d in list(run_days) + assign_days):
            continue

        assign_days.append(day)

    # Second, if still not enough runs scheduled, fill remaining slots from available days
    if len(assign_days) < n_remaining_runs:
        for day in available_days:
            if day not in assign_days:
                assign_days.append(day)
                if len(assign_days) >= n_remaining_runs:
                    break

    assign_days = sorted(assign_days[:n_remaining_runs])

    # --- Assign remaining runs ---
    for i, day in enumerate(assign_days):
        plan[day] = remaining[i]
        highlight[day] = 1

    # --- Plot ---
    barplot(
        plan,
        title=f"Week Plan | {runs} runs, target {week_target} km",
        sub_title=f"{done_km:.1f} km done, {sum(remaining):.1f} km to go",
        highlight_list=highlight
    )
