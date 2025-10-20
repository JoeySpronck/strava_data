import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import numpy as np
plt.rc('axes', axisbelow=True)

SHOW_PLOTS = True

# === GLOBAL PLOT STYLE SETTINGS ===
COLORS = {
    "background": "#000000",
    "main": "#FC5200", # strava orange
    "neutral": "#FFFFFF",
    "dark": "#555555",
}

STYLE = {
    "background_color": COLORS["background"],
    "text_color": COLORS["dark"],
    "highlight_color": COLORS["main"],
    "neutral_color": COLORS["neutral"],
    "bar_edge_color": COLORS["background"],
    "grid_color": COLORS["dark"],
    "grid_alpha": 0.2,
    "title_fontsize": 14,
    "label_fontsize": 11,
    "subtitle_fontsize": 10,
    "small_fontsize": 8,
    "title_weight": "bold",
    "font_family": "DejaVu Sans",
    "bar_width": 4,
    "bar_linewidth": 0.6,
    "width_small": 5,
    "width_large": 12,
    "height_large": 6,
    "height_small": 3,
    "color_seq_distance": [COLORS["dark"], COLORS["neutral"], COLORS["main"]],  # short → neutral → long
    "color_seq_pace": [COLORS["main"], COLORS["neutral"], COLORS["dark"]],      # fast → neutral → slow
    "color_seq_risk": [COLORS["dark"], COLORS["neutral"], COLORS["main"]],      # low → neutral → high
}

def setup_figure(width=STYLE["width_large"], height=STYLE["height_large"]):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(STYLE["background_color"])
    ax.set_facecolor(STYLE["background_color"])
    plt.rcParams["font.family"] = STYLE["font_family"]
    return fig, ax

def plot_weekly(df_activities: pd.DataFrame, col='risk'):
    """
    Plot weekly stacked bars colored by 'distance', 'pace', or 'risk'.

    Parameters:
        df_activities (pd.DataFrame): activity data with 'start_date', 'type', 'distance', 'average_speed'
        col (str): 'distance', 'pace', or 'risk'
    """
    # --- Filter runs ---
    df_activities['start_date'] = pd.to_datetime(df_activities['start_date'], utc=True)
    df_runs = df_activities[(df_activities['type'] == "Run") & (df_activities['start_date'].dt.year >= 2025)].copy()
    df_runs['distance_km'] = df_runs['distance'] / 1000
    df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)

    # --- Determine coloring ---
    if col == 'distance':
        cmap = mcolors.LinearSegmentedColormap.from_list('distance_map', STYLE["color_seq_distance"])
        vmin, vmax = df_runs['distance_km'].min(), df_runs['distance_km'].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        values = df_runs['distance_km']
        cbar_label = 'Distance per run (km)'

    elif col == 'pace':
        df_runs['pace_min_per_km'] = (1000 / df_runs['average_speed']) / 60
        cmap = mcolors.LinearSegmentedColormap.from_list('pace_map', STYLE["color_seq_pace"])
        avg_pace = df_runs['pace_min_per_km'].mean()
        vmin, vmax = df_runs['pace_min_per_km'].min(), df_runs['pace_min_per_km'].max()
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=avg_pace, vmax=vmax)
        values = df_runs['pace_min_per_km']
        cbar_label = 'Pace (min/km)'
        format_colorbar = lambda x, pos: f"{int(x)}:{int((x - int(x)) * 60):02d}"  # mm:ss format

    elif col == 'risk':
        scaler = StandardScaler()
        df_runs[['dist_z', 'speed_z']] = scaler.fit_transform(df_runs[['distance_km', 'average_speed']])
        df_runs['risk_zscore'] = df_runs['dist_z'] + df_runs['speed_z']
        cmap = mcolors.LinearSegmentedColormap.from_list('risk_map', STYLE["color_seq_risk"])
        vmin, vmax = df_runs['risk_zscore'].min(), df_runs['risk_zscore'].max()
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        values = df_runs['risk_zscore']
        cbar_label = 'Risk Z-score'

    else:
        raise ValueError("col must be 'distance', 'pace', or 'risk'")

    # --- Figure ---
    fig, ax = setup_figure()

    # --- Stacked bars per week ---
    for week, group in df_runs.groupby('week', sort=True):
        bottom = 0
        for _, row in group.iterrows():
            ax.bar(
                week,
                row['distance_km'],
                bottom=bottom,
                width=STYLE["bar_width"],
                color=cmap(norm(values.loc[row.name])),
                linewidth=STYLE["bar_linewidth"],
                edgecolor=STYLE["bar_edge_color"]
            )
            bottom += row['distance_km']

    # --- Axes formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', colors=STYLE["text_color"], rotation=45)
    ax.tick_params(axis='y', colors=STYLE["text_color"])
    ax.set_xlabel('Month', color=STYLE["text_color"], labelpad=8)
    ax.set_ylabel('Distance (km)', color=STYLE["text_color"], labelpad=8)

    ax.set_title(
        f'Weekly Distance Stacked per Run  |  {col.capitalize()}',
        color=STYLE["highlight_color"],
        fontsize=STYLE["title_fontsize"],
        weight=STYLE["title_weight"],
        pad=15
    )
    ax.grid(axis='y', color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], linewidth=0.5)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, color=STYLE["text_color"])
    cbar.ax.yaxis.set_tick_params(color=STYLE["text_color"])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=STYLE["text_color"])

    # Special formatting for pace colorbar
    if col == 'pace':
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_colorbar))

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()

def plot_weekly_distance_targets(df_activities: pd.DataFrame, additional_weeks: int = 1):
    # --- Prepare base data ---
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

    # --- Current week ---
    now = pd.Timestamp.now(tz=df_runs['start_date'].dt.tz)
    this_week = now.to_period('W-SUN').end_time

    if this_week not in weekly.index:
        weekly.loc[this_week] = {'total_volume': 0, 'long_run': 0}
    weekly = weekly.sort_index()

    last_week = weekly.index[-2] if len(weekly) >= 2 else None
    this_week_volume = weekly.loc[this_week, 'total_volume']
    last_week_volume = weekly.loc[last_week, 'total_volume'] if last_week else 0

    # --- Generate progressive weekly targets ---
    targets = []
    # prev_target = last_week_volume * 1.1
    prev_target = max((last_week_volume * 1.1), this_week_volume)   # this week target baseline, or this week's actual if higher
    for i in range(1, additional_weeks + 1):
        week_end = this_week + pd.Timedelta(days=7 * i)
        targets.append((week_end, prev_target * 1.1))
        prev_target *= 1.1

    # --- Recent weeks for plotting ---
    last_x_weeks = 7
    recent_weeks = weekly.iloc[-last_x_weeks:].copy()
    past_weeks = recent_weeks.loc[recent_weeks.index < this_week]

    # --- Figure ---
    fig, ax = setup_figure(width=5+additional_weeks*0.5, height=3+additional_weeks*0.5)

    # --- Bars ---
    # Past weeks
    ax.bar(
        past_weeks.index,
        past_weeks['total_volume'],
        width=STYLE["bar_width"],
        color=STYLE["neutral_color"],
        linewidth=STYLE["bar_linewidth"],
        edgecolor=STYLE["bar_edge_color"],
        label='Completed Weeks'
    )

    # Current week progress
    ax.bar(
        this_week,
        this_week_volume,
        width=STYLE["bar_width"],
        color=STYLE["neutral_color"],
        linewidth=STYLE["bar_linewidth"],
        edgecolor=STYLE["bar_edge_color"],
        label='This Week (Progress)'
    )

    # Remaining distance to target
    this_week_target = last_week_volume * 1.1
    remaining = max(this_week_target - this_week_volume, 0)
    if remaining > 0:
        ax.bar(
            this_week,
            remaining,
            bottom=this_week_volume,
            width=STYLE["bar_width"],
            color=STYLE["highlight_color"],
            label='Remaining to Target'
        )

    # --- Future week targets ---
    for week, target in targets:
        ax.bar(
            week,
            target,
            width=STYLE["bar_width"],
            color=STYLE["highlight_color"],
            alpha=0.9,
            linewidth=STYLE["bar_linewidth"],
            edgecolor=STYLE["bar_edge_color"],
            label='Future Target' if week == targets[0][0] else ""
        )

    # --- Labels on bars ---
    def add_label(x, y, text, color='white', offset=0.5, inside=False, fontsize=None):
        va = 'top' if inside else 'bottom'
        y_pos = y - (offset + 0.1) if inside else y + offset
        ax.text(
            x, y_pos, text,
            ha='center', va=va,
            fontsize=fontsize or STYLE["subtitle_fontsize" if not inside else "small_fontsize"],
            color=color, weight='bold'
        )

    for week, row in past_weeks.iterrows():
        add_label(week, row['total_volume'], f"{row['total_volume']:.1f}", color=STYLE["text_color"])
    
    if this_week_volume > 0:
        add_label(this_week, this_week_volume, f"{this_week_volume:.1f}", inside=True, color=STYLE["background_color"])
    if remaining > 0:
        add_label(this_week, this_week_volume + remaining, f"{remaining:.1f}", inside=True, color=STYLE["background_color"])
        add_label(this_week, this_week_volume + remaining, f"{this_week_target:.1f}", inside=False, color=STYLE["text_color"])
    else:
        add_label(this_week, this_week_volume, f"{this_week_volume:.1f}", inside=False, color=STYLE["text_color"])

    for week, target in targets:
        add_label(week, target, f"{target:.1f}", color=STYLE["text_color"])

    # --- Axes formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45, ha='right', color=STYLE["text_color"], rotation_mode='anchor')
    plt.yticks(color=STYLE["text_color"])
    ax.set_ylim(0, max(recent_weeks['total_volume'].max(), max(t for _, t in targets)) * 1.2)
    ax.set_ylabel('Weekly Distance (km)', color=STYLE["text_color"])
    ax.set_title(
        f'Weekly Running Volume (+10% Growth for {additional_weeks} Weeks)',
        color=STYLE["highlight_color"],
        weight=STYLE["title_weight"]
    )

    ax.grid(axis='y', color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], linewidth=0.5)

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()

def barplot(values, title=None, y_label="Distance (km)", highlight_list=None, sub_title=None):
    x = np.arange(len(values))
    total = np.sum(values)
    max_val = max(values) if len(values) > 0 else 0

    # --- Colors ---
    if highlight_list is None:
        colors = [STYLE["neutral_color"]] * len(values)
    else:
        colors = [STYLE["highlight_color"] if h == 1 else STYLE["neutral_color"] for h in highlight_list]

    # --- Figure ---
    fig, ax = setup_figure(width=5, height=3)

    # --- Bars ---
    ax.bar(x, values, color=colors, width=0.7, linewidth=STYLE["bar_linewidth"], edgecolor=STYLE["bar_edge_color"])

    # --- Labels on bars ---
    for i, v in enumerate(values):
        if v > 0:
            ax.text(
                i, v + max_val * 0.03, f"{v:.1f}",
                ha='center', va='bottom',
                color=STYLE["text_color"],
                fontsize=STYLE["subtitle_fontsize"],
                weight='bold'
            )

    # --- Axes ---
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'd{i+1}' for i in x], color=STYLE["text_color"])
    ax.set_ylabel(y_label, color=STYLE["text_color"])

    # --- Title and subtitle ---
    if title is None:
        title = f"Week Total: {round(total, 1)} km"
    fig.suptitle(title, color=STYLE["highlight_color"], weight=STYLE["title_weight"], fontsize=STYLE["title_fontsize"])
    if sub_title is not None:
        ax.set_title(sub_title, color=STYLE["text_color"], fontsize=STYLE["subtitle_fontsize"], weight='bold')

    # --- Grid and ticks ---
    ax.tick_params(axis='y', colors=STYLE["text_color"])
    ax.grid(axis='y', color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], linewidth=0.5)

    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()

def target_to_proportions(target_km, target_proportions):
    return target_proportions / sum(target_proportions) * target_km

def plot_week_plan(target_km, runs=4):
    if runs==3:
        target_proportions = np.array([5. , 0. , 6. , 0. , 0. , 9. , 0. ])
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
        target_proportions = np.array([5., 0., 6., 0., 0., 9., 0.])
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
    if remaining_km <= 0:
        return np.zeros_like(remaining_planned)

    if len(remaining_planned) == 0:
        # fallback: evenly split remaining km over remaining runs
        remaining_planned = np.ones(runs - len(runs_ran))
        
    remaining_scaled = remaining_planned / remaining_planned.sum() * remaining_km
    return remaining_scaled

def plot_current_week_plan(df_activities, week_target, runs=4, exclude_today=True):
    """
    Plot a realistic weekly running plan considering:
    - Past runs (no back-to-back if avoidable)
    - Even spacing between future runs
    - Rest days in between if possible
    """

    # --- Determine current week data ---
    s_weeks = df_activities['start_date'].dt.to_period('W-SUN')
    # df_week = df_activities[s_weeks == s_weeks.max()].copy()
    current_week_period = pd.Timestamp.now().to_period('W-SUN')
    df_week = df_activities[s_weeks == current_week_period].copy()
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
