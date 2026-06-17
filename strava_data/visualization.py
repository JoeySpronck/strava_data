import os
import calendar as _calendar
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyBboxPatch
from sklearn.preprocessing import StandardScaler
import numpy as np
plt.rc('axes', axisbelow=True)

SHOW_PLOTS = True
SAVE_FOLDER = "plots"

# === GLOBAL PLOT STYLE SETTINGS ===
COLORS = {
    "background": "#000000",
    "main": "#FB5200", # strava orange
    "neutral": "#FFFFFF",
    "dark": "#555555",
    "darker": "#2E2E2E",  # faint rings (e.g. empty calendar days) on the black background
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
    # Trail runs reuse the same distance/pace/risk colors as road runs but get diagonal
    # hatching overlaid so they're distinguishable at a glance. Hatch lines are drawn in the
    # background color (the same color as the segment separators) for consistent contrast.
    "trail_hatch": "////",
    "trail_hatch_linewidth": 0.7,
    "trail_hash_color": COLORS["dark"],  # fallback hatch color (e.g. legend swatch)
    # Trail hatch lines are a darkened version of each bar segment's own fill color so the
    # diagonals tint with the bar (dark/white/orange) instead of one fixed gray. The factor
    # scales HSV brightness: 1.0 = same color, lower = darker. Set to None to use the fixed
    # trail_hash_color for all hatched bars instead.
    "trail_hatch_darken": 0.7,
    "width_small": 5,
    "width_large": 12,
    "height_large": 6,
    "height_small": 3,
    "color_seq_distance": [COLORS["dark"], COLORS["neutral"], COLORS["main"]],  # short → neutral → long
    "color_seq_pace": [COLORS["main"], COLORS["neutral"], COLORS["dark"]],      # fast → neutral → slow
    "color_seq_risk": [COLORS["dark"], COLORS["neutral"], COLORS["main"]],      # low → neutral → high
}

# === MONTH CALENDAR SETTINGS ===
# Strava-style month overview. Circle COLOR reuses the shared dark→white→orange metric
# colormap (the exact one the multisport overview uses): each activity is colored by its
# sport's color metric (run/bike avg speed, hike carried weight, strength session time),
# normalized within that sport. Circle SIZE encodes the same magnitude that drives the
# overview bar heights (distance_km, or volume_kg for strength). The letter encodes sport.
SPORT_LETTERS = {
    "trail": "T",
    "run": "R",
    "hike": "H",
    "strength": "S",
    "bike": "B",
}

CALENDAR = {
    # Displayed circle radius bounds, in calendar-cell units (a day cell is 1×1, so keep
    # the max below 0.5 to stay inside the cell).
    "circle_min_radius": 0.16,
    "circle_max_radius": 0.40,
    # "sqrt" balances perceived area so big days don't dominate; "linear" maps straight.
    "size_scaling": "sqrt",
    # Per-sport magnitude (same metric as the overview bar height) mapping to the min / max
    # circle radius. Values below min clamp to the smallest circle, above max to the largest.
    "trail_size_min_value": 5,       "trail_size_max_value": 25,
    "run_size_min_value": 5,         "run_size_max_value": 25,
    "bike_size_min_value": 10,       "bike_size_max_value": 80,
    "hike_size_min_value": 5,        "hike_size_max_value": 20,
    "strength_size_min_value": 2000, "strength_size_max_value": 12000,
    # Same-day extra activities: smaller circles marching from the upper-right corner so
    # 3+ activities on one day still lay out cleanly.
    "secondary_size_mult": 0.55,
    "secondary_offset_x": 0.30,
    "secondary_offset_y": 0.30,
    "secondary_step_x": -0.20,
    "secondary_step_y": 0.0,
    # Outlined ring + day number for days without any activity (ring kept very dark).
    "empty_circle_radius": 0.34,
    "empty_circle_color": COLORS["darker"],
    "future_color": "#1C1C1C",  # days still to come: ring + number even darker than empty
    "today_highlight_color": COLORS["main"],  # rounded square behind today
    "today_corner_radius": 0.18,
    "day_number_fontsize": 9,  # day numbers on empty days stay small/subtle
    # Sport letters scale with their circle: fontsize = radius * ratio (floored so the
    # smallest circles stay legible). Bump the ratio for bigger letters.
    "letter_size_ratio": 62,
    "letter_min_fontsize": 7,
    "weekday_fontsize": 12,    # Mon/Tue/... column headers
    "legend_fontsize": 9,      # bottom legend
    "border_width": 1.4,
}


def setup_figure(width=STYLE["width_large"], height=STYLE["height_large"]):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(STYLE["background_color"])
    ax.set_facecolor(STYLE["background_color"])
    plt.rcParams["font.family"] = STYLE["font_family"]
    return fig, ax

def _darken(color, factor):
    """Return `color` darkened by scaling its HSV brightness by `factor` (0..1).

    Keeps hue and saturation, so orange stays orange and white becomes gray. Used to tint
    trail-run hatch lines a shade darker than the bar segment they sit on.
    """
    r, g, b, a = mcolors.to_rgba(color)
    h, s, v = mcolors.rgb_to_hsv((r, g, b))
    r2, g2, b2 = mcolors.hsv_to_rgb((h, s, v * factor))
    return (r2, g2, b2, a)


def metric_colormap(color_seq=None):
    """The shared dark→white→orange metric colormap used across the weekly plots.

    Centralized so the month calendar colors activities through the exact same colormap
    as the multisport overview. Defaults to STYLE['color_seq_distance'] (dark→neutral→main).
    """
    if color_seq is None:
        color_seq = STYLE["color_seq_distance"]
    return mcolors.LinearSegmentedColormap.from_list('weekly_map', color_seq)


def _draw_weekly_stacked(ax, df, stack_col, color_col, color_seq=None, norm_center=None,
                         color_vmin=None, color_vmax=None, hatch_col=None):
    """Draw stacked weekly bars onto an existing axes. Returns (cmap, norm).

    color_vmin / color_vmax override the data-derived color range. Use them to cap the
    scale so a couple of extreme activities don't compress everything else into one end
    of the colormap; out-of-range values clamp to the end colors (clip=True).

    hatch_col: optional name of a boolean column; rows that are True get diagonal hatching
    overlaid on their segment (used to mark trail runs apart from road runs).
    """
    # Hatch lines inherit the patch edgecolor; keep them thin so they read as texture, not noise.
    plt.rcParams['hatch.linewidth'] = STYLE["trail_hatch_linewidth"]
    cmap = metric_colormap(color_seq)
    values = df[color_col]
    vmin = values.min() if color_vmin is None else color_vmin
    vmax = values.max() if color_vmax is None else color_vmax
    if norm_center is not None:
        # TwoSlopeNorm requires strict vmin < vcenter < vmax
        eps = 1e-9
        lo = min(vmin, norm_center - eps)
        hi = max(vmax, norm_center + eps)
        norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=norm_center, vmax=hi)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    for week, group in df.groupby('week', sort=True):
        bottom = 0
        for _, row in group.iterrows():
            height = row[stack_col]
            fill = cmap(norm(values.loc[row.name]))
            is_trail = bool(hatch_col and row[hatch_col])
            # Base bar: just the fill, no border yet (the black border is drawn last so it
            # sits ON TOP of any hatch and the hatch can't protrude past the bar edges).
            ax.bar(
                week,
                height,
                bottom=bottom,
                width=STYLE["bar_width"],
                color=fill,
                linewidth=0,
                zorder=1,
            )
            # Trail runs: overlay ONLY the hatch (between fill and border). The hatch takes the
            # patch edgecolor, so this patch is transparent (facecolor='none') with linewidth=0 —
            # no border drawn — while the diagonal lines get their own darkened-fill color
            # (hatch line width comes from the rcParam above).
            if is_trail:
                darken = STYLE["trail_hatch_darken"]
                hatch_color = _darken(fill, darken) if darken is not None else STYLE["trail_hash_color"]
                ax.bar(
                    week,
                    height,
                    bottom=bottom,
                    width=STYLE["bar_width"],
                    facecolor='none',
                    linewidth=0,
                    edgecolor=hatch_color,
                    hatch=STYLE["trail_hatch"],
                    zorder=2,
                )
            # Black segment border ("square" around each block) drawn last, on top of the hatch,
            # so the hatch is clipped to the bar and the border stays crisp and black.
            ax.bar(
                week,
                height,
                bottom=bottom,
                width=STYLE["bar_width"],
                facecolor='none',
                linewidth=STYLE["bar_linewidth"],
                edgecolor=STYLE["bar_edge_color"],
                zorder=3,
            )
            bottom += height
    return cmap, norm


def _cap_flags(values, color_vmin, color_vmax):
    """Whether the color scale is capped *and* data actually spills past the cap.

    Only then is a ``≥`` / ``≤`` boundary label meaningful: values beyond the cap clamp
    to the end color, so the boundary tick really means "this value or beyond".
    """
    cap_low = color_vmin is not None and float(values.min()) < color_vmin
    cap_high = color_vmax is not None and float(values.max()) > color_vmax
    return cap_low, cap_high


def _attach_colorbar(ax, cmap, norm, color_label, color_format_fn=None, label_fontsize=None,
                     tick_fontsize=None, cap_low=False, cap_high=False):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(color_label, color=STYLE["text_color"], fontsize=label_fontsize)
    cbar.ax.yaxis.set_tick_params(color=STYLE["text_color"], labelsize=tick_fontsize)

    if cap_low or cap_high:
        vmin, vmax = float(norm.vmin), float(norm.vmax)
        tol = max(abs(vmax - vmin), 1.0) * 1e-6
        # Keep the auto ticks but guarantee one lands exactly on each capped boundary,
        # so the "≥"/"≤" marker shows at the end of the bar rather than on the nearest tick.
        ticks = [t for t in cbar.get_ticks() if vmin - tol <= t <= vmax + tol]
        if cap_high and not any(abs(t - vmax) <= tol for t in ticks):
            ticks.append(vmax)
        if cap_low and not any(abs(t - vmin) <= tol for t in ticks):
            ticks.append(vmin)
        cbar.set_ticks(sorted(ticks))

        def _fmt(x, pos):
            base = color_format_fn(x, pos) if color_format_fn is not None else f"{x:g}"
            if cap_high and abs(x - vmax) <= tol:
                return f"≥{base}"
            if cap_low and abs(x - vmin) <= tol:
                return f"≤{base}"
            return base
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt))
    elif color_format_fn is not None:
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(color_format_fn))

    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=STYLE["text_color"])
    return cbar


def plot_weekly_stacked(
    df: pd.DataFrame,
    stack_col: str,
    color_col: str,
    stack_label: str,
    color_label: str,
    title: str,
    color_seq=None,
    norm_center=None,
    color_format_fn=None,
    color_vmin=None,
    color_vmax=None,
    hatch_col=None,
    save_name=None,
):
    """
    Plot weekly stacked bars where each segment is one activity.

    Each bar (one per week) stacks `stack_col` values; segments are colored by `color_col`.
    Works for any activity type as long as the dataframe carries a 'week' column.

    Parameters:
        df (pd.DataFrame): activities with a 'week' column plus `stack_col` and `color_col`.
        stack_col (str): column whose values determine bar segment heights.
        color_col (str): column whose values determine segment colors.
        stack_label (str): y-axis label.
        color_label (str): colorbar label.
        title (str): figure title.
        color_seq (list[str] | None): colors for the colormap (low → high). Defaults to dark→neutral→main.
        norm_center (float | None): if set, use a TwoSlopeNorm centered here; otherwise linear.
        color_format_fn: optional matplotlib tick formatter callable for the colorbar.
        color_vmin / color_vmax (float | None): cap the color scale instead of using the
            data min/max, so extreme outliers don't skew the colormap. Values beyond the
            cap clamp to the end colors.
        save_name (str | None): if set, saves the plot under SAVE_FOLDER.
    """
    fig, ax = setup_figure()
    cmap, norm = _draw_weekly_stacked(ax, df, stack_col, color_col, color_seq, norm_center,
                                      color_vmin=color_vmin, color_vmax=color_vmax,
                                      hatch_col=hatch_col)

    # --- Axes formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', colors=STYLE["text_color"], rotation=45)
    ax.tick_params(axis='y', colors=STYLE["text_color"])
    ax.set_xlabel('Month', color=STYLE["text_color"], labelpad=8)
    ax.set_ylabel(stack_label, color=STYLE["text_color"], labelpad=8)

    ax.set_title(
        title,
        color=STYLE["highlight_color"],
        fontsize=STYLE["title_fontsize"],
        weight=STYLE["title_weight"],
        pad=15
    )
    ax.grid(axis='y', color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], linewidth=0.5)

    # When trail runs are hatched, add a legend so the diagonal lines read as "trail run".
    if hatch_col is not None and df[hatch_col].any():
        from matplotlib.patches import Patch
        # Representative swatch: a neutral fill with the hatch drawn the same way as the bars
        # (darkened fill color, or the fixed fallback when darkening is disabled).
        darken = STYLE["trail_hatch_darken"]
        legend_hatch_color = (_darken(STYLE["neutral_color"], darken)
                              if darken is not None else STYLE["trail_hash_color"])
        trail_patch = Patch(facecolor=STYLE["neutral_color"], edgecolor=legend_hatch_color,
                            hatch=STYLE["trail_hatch"], label='Trail run')
        legend = ax.legend(handles=[trail_patch], loc='upper left',
                           facecolor=STYLE["background_color"], edgecolor=STYLE["text_color"],
                           fontsize=STYLE["subtitle_fontsize"])
        plt.setp(legend.get_texts(), color=STYLE["text_color"])

    cap_low, cap_high = _cap_flags(df[color_col], color_vmin, color_vmax)
    _attach_colorbar(ax, cmap, norm, color_label, color_format_fn=color_format_fn,
                     cap_low=cap_low, cap_high=cap_high)

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(SAVE_FOLDER, save_name), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_weekly_stacked_multi(
    panels,
    save_name=None,
    panel_height=1.6,
    width=None,
    suptitle=None,
):
    """
    Render multiple weekly-stacked sport panels stacked vertically in one figure.

    Each panel renders the same way as `plot_weekly_stacked`, but with a slimmer
    y-axis and a shared x-axis so the weeks line up across sports.

    Parameters:
        panels (list[dict]): one dict per panel. Required keys:
            df, stack_col, color_col, stack_label, color_label, title.
          Optional keys:
            color_seq, norm_center, color_format_fn, color_vmin, color_vmax.
        panel_height (float): height in inches per panel (default 1.6).
        width (float | None): figure width; defaults to STYLE['width_large'].
        suptitle (str | None): figure-level title above all panels.
        save_name (str | None): if set, saves the plot under SAVE_FOLDER.
    """
    n = len(panels)
    if width is None:
        width = STYLE["width_large"]

    plt.style.use('dark_background')
    fig, axes = plt.subplots(n, 1, figsize=(width, panel_height * n + 0.6), sharex=True)
    fig.patch.set_facecolor(STYLE["background_color"])
    plt.rcParams["font.family"] = STYLE["font_family"]
    if n == 1:
        axes = [axes]

    # Shared x-range across all panels so weeks align.
    all_weeks = pd.concat([p['df']['week'] for p in panels if len(p['df']) > 0])
    if len(all_weeks) > 0:
        xmin, xmax = all_weeks.min(), all_weeks.max()
    else:
        xmin = xmax = pd.Timestamp.now()

    for i, panel in enumerate(panels):
        ax = axes[i]
        ax.set_facecolor(STYLE["background_color"])

        df = panel['df']
        if len(df) > 0:
            hatch_col = panel.get('hatch_col')
            cmap, norm = _draw_weekly_stacked(
                ax, df,
                panel['stack_col'], panel['color_col'],
                color_seq=panel.get('color_seq'),
                norm_center=panel.get('norm_center'),
                color_vmin=panel.get('color_vmin'),
                color_vmax=panel.get('color_vmax'),
                hatch_col=hatch_col,
            )
            cap_low, cap_high = _cap_flags(df[panel['color_col']],
                                           panel.get('color_vmin'), panel.get('color_vmax'))
            _attach_colorbar(
                ax, cmap, norm, panel['color_label'],
                color_format_fn=panel.get('color_format_fn'),
                label_fontsize=STYLE["small_fontsize"],
                tick_fontsize=STYLE["small_fontsize"],
                cap_low=cap_low, cap_high=cap_high,
            )
            # Legend marking the hatched series (e.g. trail runs) on panels that use it.
            if hatch_col is not None and df[hatch_col].any():
                from matplotlib.patches import Patch
                darken = STYLE["trail_hatch_darken"]
                legend_hatch_color = (_darken(STYLE["neutral_color"], darken)
                                      if darken is not None else STYLE["trail_hash_color"])
                trail_patch = Patch(facecolor=STYLE["neutral_color"], edgecolor=legend_hatch_color,
                                    hatch=STYLE["trail_hatch"], label='Trail run')
                legend = ax.legend(handles=[trail_patch], loc='upper left',
                                   facecolor=STYLE["background_color"], edgecolor=STYLE["text_color"],
                                   fontsize=STYLE["small_fontsize"])
                plt.setp(legend.get_texts(), color=STYLE["text_color"])

        ax.set_ylabel(panel['stack_label'], color=STYLE["text_color"], fontsize=STYLE["small_fontsize"], labelpad=6)
        ax.tick_params(axis='y', colors=STYLE["text_color"], labelsize=STYLE["small_fontsize"])
        ax.set_title(
            panel['title'],
            color=STYLE["highlight_color"],
            fontsize=STYLE["subtitle_fontsize"],
            weight=STYLE["title_weight"],
            pad=4,
        )
        ax.grid(axis='y', color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], linewidth=0.5)
        ax.set_xlim(xmin - pd.Timedelta(days=5), xmax + pd.Timedelta(days=5))

    # Bottom axis: month tick labels only
    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    bottom_ax.xaxis.set_major_locator(mdates.MonthLocator())
    bottom_ax.tick_params(axis='x', colors=STYLE["text_color"], rotation=45, labelsize=STYLE["small_fontsize"])
    bottom_ax.set_xlabel('Month', color=STYLE["text_color"], labelpad=4, fontsize=STYLE["small_fontsize"])

    if suptitle:
        fig.suptitle(
            suptitle,
            color=STYLE["highlight_color"],
            fontsize=STYLE["title_fontsize"],
            weight=STYLE["title_weight"],
        )

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(SAVE_FOLDER, save_name), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_weekly(df_runs: pd.DataFrame, col='risk', save_name=None):
    """
    Plot weekly stacked run bars colored by 'distance', 'pace', or 'risk'.

    Thin wrapper over `plot_weekly_stacked` that sets the run-specific stack column
    (distance_km) and configures the chosen color dimension.

    Parameters:
        df_runs (pd.DataFrame): runs with 'week', 'distance_km', 'average_speed', 'sport_type'.
        col (str): 'distance', 'pace', or 'risk'.

    Trail runs (sport_type containing 'Trail') are drawn with the same color metric as road
    runs but overlaid with diagonal hatching so they stand out.
    """
    df_runs = df_runs.copy()
    # Trail runs share the legacy type 'Run'; sport_type is what tells them apart.
    df_runs['is_trail'] = df_runs['sport_type'].astype(str).str.contains('Trail', case=False, na=False)

    if col == 'distance':
        plot_weekly_stacked(
            df_runs,
            stack_col='distance_km',
            color_col='distance_km',
            stack_label='Distance (km)',
            color_label='Distance per run (km)',
            title='Weekly Distance Stacked per Run  |  Distance',
            color_seq=STYLE["color_seq_distance"],
            hatch_col='is_trail',
            save_name=save_name,
        )
    elif col == 'pace':
        df_runs['pace_min_per_km'] = (1000 / df_runs['average_speed']) / 60
        plot_weekly_stacked(
            df_runs,
            stack_col='distance_km',
            color_col='pace_min_per_km',
            stack_label='Distance (km)',
            color_label='Pace (min/km)',
            title='Weekly Distance Stacked per Run  |  Pace',
            color_seq=STYLE["color_seq_pace"],
            norm_center=df_runs['pace_min_per_km'].mean(),
            color_format_fn=lambda x, pos: f"{int(x)}:{int((x - int(x)) * 60):02d}",
            hatch_col='is_trail',
            save_name=save_name,
        )
    elif col == 'risk':
        scaler = StandardScaler()
        df_runs[['dist_z', 'speed_z']] = scaler.fit_transform(df_runs[['distance_km', 'average_speed']])
        df_runs['risk_zscore'] = df_runs['dist_z'] + df_runs['speed_z']
        plot_weekly_stacked(
            df_runs,
            stack_col='distance_km',
            color_col='risk_zscore',
            stack_label='Distance (km)',
            color_label='Risk Z-score',
            title='Weekly Distance Stacked per Run  |  Risk',
            color_seq=STYLE["color_seq_risk"],
            norm_center=0,
            hatch_col='is_trail',
            save_name=save_name,
        )
    else:
        raise ValueError("col must be 'distance', 'pace', or 'risk'")

def grow_target(prev: float, recovery_ceiling=None) -> float:
    """Next weekly volume target.

    The 10% rule mostly serves to set new limits, but when building back up you can
    usually grow faster. So grow at +25% while a boosted step still stays under the
    recovery ceiling (a previously sustained volume); once +25% would reach/exceed it,
    fall back to the conservative +10%. With no ceiling, always +10%.
    """
    boosted = prev * 1.25
    if recovery_ceiling is not None and boosted < recovery_ceiling:
        return boosted
    return prev * 1.1


def plot_weekly_distance_targets(df_weekly: pd.DataFrame, week_target: float,
                                 this_week: pd.Timestamp, this_week_target: float, this_week_volume: float, target_reached: bool,
                                 recovery_ceiling=None,
                                 additional_weeks: int = 4, last_weeks: int = 7, save_name=None):
    # # --- Prepare base data ---
    # df_activities['start_date'] = pd.to_datetime(df_activities['start_date'], utc=True)
    # df_runs = df_activities[
    #     (df_activities['type'] == "Run") &
    #     (df_activities['start_date'].dt.year >= 2025)
    # ].copy()
    
    # df_runs['distance_km'] = df_runs['distance'] / 1000
    # df_runs['week'] = df_runs['start_date'].dt.to_period('W-SUN').apply(lambda r: r.end_time)

    # weekly = df_runs.groupby('week')['distance_km'].agg(
    #     total_volume='sum',
    #     long_run='max'
    # ).sort_index()

    # # --- Current week ---
    # now = pd.Timestamp.now(tz=df_runs['start_date'].dt.tz)
    # this_week = now.to_period('W-SUN').end_time

    # if this_week not in weekly.index:
    #     weekly.loc[this_week] = {'total_volume': 0, 'long_run': 0}
    # weekly = weekly.sort_index()

    # # --- This week and future targets using last 4 completed weeks ---
    # this_week_volume = weekly.loc[this_week, 'total_volume']

    # # --- Last 4 completed weeks (exclude this week) ---
    # completed_weeks = weekly.loc[weekly.index < this_week]
    # recent_completed = completed_weeks.iloc[-4:]
    # base_completed = recent_completed['total_volume'].max() if len(recent_completed) > 0 else 0

    # # --- This week target ---
    # this_week_target = base_completed * 1.1
    # this_week_volume = weekly.loc[this_week, 'total_volume']
    # target_reached = this_week_volume >= this_week_target

    # # --- Base for future growth ---
    # # If target reached, use max(this week, base_completed); else, start future growth from this_week_target
    # base_for_growth = this_week_target if not target_reached else max(this_week_volume, this_week_target)

    # # --- Generate progressive future targets ---
    # targets = []
    # prev_target = base_for_growth
    # for i in range(1, additional_weeks + 1):
    #     week_end = this_week + pd.Timedelta(days=7 * i)
    #     prev_target *= 1.1
    #     targets.append((week_end, prev_target))

    # --- Generate progressive future targets ---
    targets = []
    prev_target = week_target
    for i in range(1, additional_weeks + 1):
        week_end = this_week + pd.Timedelta(days=7 * i)
        prev_target = grow_target(prev_target, recovery_ceiling)
        targets.append((week_end, prev_target))

    # --- Recent weeks for plotting ---
    recent_weeks = df_weekly.iloc[-last_weeks:].copy()
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

    # Current week
    if target_reached:
        # Fully completed bar
        ax.bar(
            this_week,
            this_week_volume,
            width=STYLE["bar_width"],
            color=STYLE["neutral_color"],
            linewidth=STYLE["bar_linewidth"],
            edgecolor=STYLE["bar_edge_color"],
            label='This Week (Completed)'
        )
    else:
        remaining = max(this_week_target - this_week_volume, 0)
        # Completed portion
        ax.bar(
            this_week,
            this_week_volume,
            width=STYLE["bar_width"],
            color=STYLE["neutral_color"],
            linewidth=STYLE["bar_linewidth"],
            edgecolor=STYLE["bar_edge_color"],
            label='This Week (Progress)'
        )
        # Remaining portion
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

    # --- Labels ---
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

    if target_reached:
        add_label(this_week, this_week_volume, f"{this_week_volume:.1f}", color=STYLE["text_color"])
    else:
        remaining = max(this_week_target - this_week_volume, 0)
        if this_week_volume > 0:
            add_label(this_week, this_week_volume, f"{this_week_volume:.1f}", inside=True, color=STYLE["background_color"])
        if remaining > 0:
            add_label(this_week, this_week_volume + remaining, f"{remaining:.1f}", inside=True, color=STYLE["background_color"])
            add_label(this_week, this_week_volume + remaining, f"{this_week_target:.1f}", inside=False, color=STYLE["text_color"])
        else:
            add_label(this_week, this_week_volume, f"{this_week_volume:.1f}", color=STYLE["text_color"])

    for week, target in targets:
        add_label(week, target, f"{target:.1f}", color=STYLE["text_color"])

    # --- Recovery ceiling (previously sustained volume) ---
    # Below this line build-up runs at +25%/week; reaching it reverts growth to +10%.
    ceiling_drawn = recovery_ceiling is not None and np.isfinite(recovery_ceiling)
    if ceiling_drawn:
        ax.axhline(
            recovery_ceiling,
            color=STYLE["text_color"],  # dark gray
            linestyle='--',
            linewidth=1.3,
            alpha=0.9,
            zorder=0,
        )

    # --- Axes formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45, ha='right', color=STYLE["text_color"], rotation_mode='anchor')
    plt.yticks(color=STYLE["text_color"])
    ceiling_for_ylim = recovery_ceiling if ceiling_drawn else 0
    ax.set_ylim(0, max(recent_weeks['total_volume'].max(), max(t for _, t in targets), ceiling_for_ylim) * 1.2)
    ax.set_ylabel('Weekly Distance (km)', color=STYLE["text_color"])
    ax.set_title(
        f'Weekly Running Volume Progression (+10–25% Growth Target)',
        color=STYLE["highlight_color"],
        weight=STYLE["title_weight"]
    )

    ax.grid(axis='y', color=STYLE["grid_color"], alpha=STYLE["grid_alpha"], linewidth=0.5)

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(SAVE_FOLDER, save_name), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def barplot(values, title=None, y_label="Distance (km)", highlight_list=None, sub_title=None, save_name=None):
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
    if save_name:
        plt.savefig(os.path.join(SAVE_FOLDER, save_name), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def target_to_proportions(target_km, target_proportions):
    return target_proportions / sum(target_proportions) * target_km

def plot_week_plan(target_km, runs=4, save_name=None):
    if runs==3:
        target_proportions = np.array([5. , 0. , 6. , 0. , 0. , 9. , 0. ])
    elif runs==4:
        target_proportions = np.array([3. , 6. , 0. , 4.5, 0. , 9. , 0. ])
    elif runs==5:
        target_proportions = np.array([ 4. ,  7.5,  0. ,  5.5,  3.5, 12. ,  0. ])
    else:
        raise ValueError("Only 3, 4 or 5 runs supported.")
    values = target_to_proportions(target_km, target_proportions)
    barplot(values, title=f"Week Plan for {target_km} km ({runs} runs)", save_name=save_name)

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

# def plot_current_week_plan(df_runs, week_target, runs=4, save_name=None):
#     """
#     Plot a realistic weekly running plan considering:
#     - Past runs (no back-to-back if avoidable)
#     - Even spacing between future runs
#     - Rest days in between if possible
#     - Automatically excludes today if a run was already done today
#     - Properly includes Sunday in planning
#     """

#     # --- Determine current week data (ISO week: Monday–Sunday) ---
#     s_weeks = df_runs['start_date'].dt.isocalendar().week
#     s_years = df_runs['start_date'].dt.isocalendar().year

#     now = pd.Timestamp.now()
#     current_week = now.isocalendar().week
#     current_year = now.isocalendar().year

#     df_week = df_runs[
#         (s_years == current_year) & (s_weeks == current_week)
#     ].copy()

#     df_week['day_of_week'] = df_week['start_date'].dt.day_of_week  # Monday=0, Sunday=6

#     # --- Fill actual distances into 7-day array ---
#     current_week_km = np.zeros(7)
#     for _, row in df_week.iterrows():
#         current_week_km[int(row['day_of_week'])] += row['distance'] / 1000

#     # --- Identify current day ---
#     current_day = now.day_of_week
#     ran_today = current_week_km[current_day] > 0

#     # Automatically exclude today if already ran
#     days_left = np.arange(current_day + 1 if ran_today else current_day, 7)
#     if len(days_left) == 0:
#         print("Week is over — no remaining days to plan.")
#         return

#     # --- Compute remaining runs ---
#     runs_ran = list(df_week['distance'] / 1000)
#     done_km = sum(runs_ran)
#     remaining = remaining_week_kms(runs_ran, week_target, runs=runs)
#     n_remaining_runs = len(remaining)

#     if n_remaining_runs == 0:
#         print("No runs remaining — goal already met!")
#         return
#     if len(days_left) < n_remaining_runs:
#         print(f"IMPOSSIBLE: Only {len(days_left)} days left, need {n_remaining_runs} runs.")
#         return

#     # --- Identify already-run days ---
#     run_days = np.where(current_week_km > 0)[0]
#     available_days = list(days_left)

#     # --- Smart placement ---
#     plan = current_week_km.copy()
#     highlight = np.zeros(7)

#     last_run_day = run_days.max() if len(run_days) > 0 else -2
#     min_gap = 1  # Prefer at least 1 rest day

#     assign_days = []

#     # First pass: choose well-spaced days
#     for day in available_days:
#         if len(assign_days) >= n_remaining_runs:
#             break
#         if any(abs(day - d) <= min_gap for d in list(run_days) + assign_days):
#             continue
#         assign_days.append(day)

#     # Second pass: fill remaining slots if needed
#     if len(assign_days) < n_remaining_runs:
#         for day in available_days:
#             if day not in assign_days:
#                 assign_days.append(day)
#                 if len(assign_days) >= n_remaining_runs:
#                     break

#     assign_days = sorted(assign_days[:n_remaining_runs])

#     # --- Assign planned runs ---
#     for i, day in enumerate(assign_days):
#         plan[day] = remaining[i]
#         highlight[day] = 1

#     # --- Plot ---
#     barplot(
#         plan,
#         title=f"Week Plan | {runs} runs, target {week_target} km",
#         sub_title=f"{done_km:.1f} km done, {sum(remaining):.1f} km to go",
#         highlight_list=highlight,
#         save_name=save_name
#     )

def plot_current_week_plan(df_runs, week_target, runs=4, save_name=None, target_next_week=False):
    """
    Plot a realistic weekly running plan considering:
    - Past runs (no back-to-back if avoidable)
    - Even spacing between future runs
    - Rest days in between if possible
    - Automatically excludes today if a run was already done today
    - Properly includes Sunday in planning
    - If target_next_week=True, plans next week's schedule instead
    """

    # --- Determine current or next week ---
    now = pd.Timestamp.now()
    current_week = now.isocalendar().week
    current_year = now.isocalendar().year

    if target_next_week:
        # Shift to next ISO week
        next_week_date = now + pd.Timedelta(days=7)
        target_week = next_week_date.isocalendar().week
        target_year = next_week_date.isocalendar().year
        title_week_label = "Next Week"
    else:
        target_week = current_week
        target_year = current_year
        title_week_label = "Current Week"

    # --- Filter activities for selected week ---
    s_weeks = df_runs['start_date'].dt.isocalendar().week
    s_years = df_runs['start_date'].dt.isocalendar().year
    df_week = df_runs[
        (s_years == target_year) & (s_weeks == target_week)
    ].copy()

    df_week['day_of_week'] = df_week['start_date'].dt.day_of_week  # Monday=0, Sunday=6

    # --- Fill actual distances into 7-day array ---
    current_week_km = np.zeros(7)
    for _, row in df_week.iterrows():
        current_week_km[int(row['day_of_week'])] += row['distance'] / 1000

    # --- Identify current day ---
    current_day = now.day_of_week
    ran_today = current_week_km[current_day] > 0 if not target_next_week else False

    # --- Days left for planning ---
    if target_next_week:
        days_left = np.arange(0, 7)
    else:
        days_left = np.arange(current_day + 1 if ran_today else current_day, 7)

    if len(days_left) == 0:
        print("Week is over — no remaining days to plan.")
        return

    # --- Compute remaining runs ---
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

    # --- Identify already-run days ---
    run_days = np.where(current_week_km > 0)[0]
    available_days = list(days_left)

    # --- Improved smart placement ---
    plan = current_week_km.copy()
    highlight = np.zeros(7)

    min_gap = 1  # prefer at least 1 rest day between runs
    assign_days = []

    # Pass 1: schedule from Sunday backwards to maximize spacing
    for day in reversed(available_days):
        if len(assign_days) >= n_remaining_runs:
            break
        if any(abs(day - d) <= min_gap for d in list(run_days) + assign_days):
            continue
        assign_days.append(day)

    # Pass 2: fill remaining slots
    if len(assign_days) < n_remaining_runs:
        for day in available_days:
            if day not in assign_days:
                assign_days.append(day)
                if len(assign_days) >= n_remaining_runs:
                    break

    # Sort final days for plotting
    assign_days = sorted(assign_days[:n_remaining_runs])

    # --- Assign planned runs ---
    for i, day in enumerate(assign_days):
        plan[day] = remaining[i]
        highlight[day] = 1

    # --- Plot ---
    barplot(
        plan,
        title=f"{title_week_label} Plan | {runs} runs, target {week_target} km",
        sub_title=f"{done_km:.1f} km done, {sum(remaining):.1f} km to go",
        highlight_list=highlight,
        save_name=save_name
    )


def _calendar_radius(value, sport):
    """Map an activity magnitude to a circle radius using CALENDAR settings.

    Reuses the per-sport min/max *value* range and the global min/max *radius*, scaling
    either 'linear' or 'sqrt'. Values outside the range clamp to the end circle sizes, so
    a tiny activity is never smaller than circle_min_radius and a huge one never larger
    than circle_max_radius.
    """
    lo = CALENDAR[f"{sport}_size_min_value"]
    hi = CALENDAR[f"{sport}_size_max_value"]
    r_min = CALENDAR["circle_min_radius"]
    r_max = CALENDAR["circle_max_radius"]

    if value is None or not np.isfinite(value):
        return r_min
    if hi <= lo:
        frac = 1.0
    else:
        frac = (min(max(value, lo), hi) - lo) / (hi - lo)  # clamp to [0, 1]
    if CALENDAR["size_scaling"] == "sqrt":
        frac = np.sqrt(frac)
    return r_min + frac * (r_max - r_min)


def _text_color_for(rgba):
    """Pick black or white letter color for legibility on a given circle fill."""
    r, g, b = rgba[:3]  # matplotlib RGBA components are already 0..1
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 0.55 else STYLE["neutral_color"]


def plot_month_calendar(df_cal, year, month, title=None, save_name=None, first_weekday=0, today=None):
    """
    Strava-style month overview: a calendar grid where each day is a circle.

    Days without activities show an outlined ring with the day number centered. Days with
    activities show a filled circle whose COLOR comes from the shared metric colormap (the
    same one the multisport overview uses) and whose SIZE encodes the activity magnitude
    (the same metric as the overview bar heights), with the sport letter centered. Extra
    activities on the same day become smaller circles marching from the upper-right corner,
    so 3+ activities on one day still lay out cleanly.

    Parameters:
        df_cal (pd.DataFrame): one row per activity, with columns:
            'date'       — datetime-like; the day the activity falls on.
            'sport'      — key in SPORT_LETTERS ('run'/'trail'/'hike'/'strength'/'bike').
            'size_value' — magnitude metric (distance_km, or volume_kg for strength) → size.
            'color'      — RGBA tuple (from metric_colormap) → circle fill.
        year, month (int): month to render.
        title (str | None): figure title; defaults to "Month YYYY".
        first_weekday (int): 0 = Monday (default), 6 = Sunday.
        save_name (str | None): if set, saves under SAVE_FOLDER.
        today (datetime-like | None): reference "today" used to highlight the current day
            and dim still-to-come days; defaults to the current date.
    """
    today = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    today_tuple = (today.year, today.month)

    def _day_status(day):
        """Return 'today', 'future', or 'past' for a day in the rendered month."""
        if (year, month) == today_tuple:
            if day == today.day:
                return 'today'
            return 'future' if day > today.day else 'past'
        return 'future' if (year, month) > today_tuple else 'past'

    cal = _calendar.Calendar(firstweekday=first_weekday)
    weeks = cal.monthdayscalendar(year, month)  # list of weeks; each a list of 7 day nums (0 = padding)
    n_rows = len(weeks)

    # --- Index activities by day-of-month for the requested month ---
    by_day = {}
    if df_cal is not None and len(df_cal) > 0:
        dts = pd.to_datetime(df_cal['date'])
        mask = (dts.dt.year == year) & (dts.dt.month == month)
        for (_, row), day in zip(df_cal[mask].iterrows(), dts[mask].dt.day):
            by_day.setdefault(int(day), []).append(row)

    # --- Figure (square cells via equal aspect); extra bottom room for the legend table ---
    legend_h = 2.8  # data-unit height reserved below the grid for the legend table
    data_w, data_h = 7.0, n_rows + 1.0 + legend_h  # xlim spans 7; ylim spans this
    fig_w = 7.5
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * data_h / data_w))
    fig.patch.set_facecolor(STYLE["background_color"])
    ax.set_facecolor(STYLE["background_color"])
    plt.rcParams["font.family"] = STYLE["font_family"]

    grid_bottom = -(n_rows - 0.5)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(grid_bottom - legend_h, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    def _letter_fontsize(radius):
        """Font size that scales with the circle radius, floored for legibility."""
        return max(radius * CALENDAR["letter_size_ratio"], CALENDAR["letter_min_fontsize"])

    # Glyphs we want geometrically centered in their circle. We can't know a glyph's
    # rendered extent before layout, so we collect (text_artist, target_xy) here and
    # nudge each onto its target after a draw pass (see _recenter below).
    centered = []

    # --- Weekday headers ---
    for col in range(7):
        ax.text(
            col, 1.0, _calendar.day_abbr[(first_weekday + col) % 7],
            ha='center', va='center',
            color=STYLE["text_color"], fontsize=CALENDAR["weekday_fontsize"], weight='bold',
        )

    # --- Day cells ---
    for r, week in enumerate(weeks):
        cy = -r
        for col, day in enumerate(week):
            if day == 0:
                continue  # padding day from an adjacent month
            cx = col
            acts = by_day.get(day, [])
            status = _day_status(day)

            # Rounded orange square behind today's cell.
            if status == 'today':
                side = 0.88
                ax.add_patch(FancyBboxPatch(
                    (cx - side / 2, cy - side / 2), side, side,
                    boxstyle=f"round,pad=0,rounding_size={CALENDAR['today_corner_radius']}",
                    facecolor=CALENDAR["today_highlight_color"], edgecolor='none', zorder=1,
                ))

            if not acts:
                # Empty day: outlined ring + number. Future days are dimmed even darker;
                # today (always activity-less here) shows a readable number on the orange.
                if status == 'future':
                    ring_color = num_color = CALENDAR["future_color"]
                elif status == 'today':
                    ring_color = None
                    num_color = STYLE["neutral_color"]
                else:
                    ring_color = CALENDAR["empty_circle_color"]
                    num_color = STYLE["text_color"]
                if ring_color is not None:
                    ax.add_patch(Circle(
                        (cx, cy), CALENDAR["empty_circle_radius"],
                        facecolor='none', edgecolor=ring_color,
                        linewidth=CALENDAR["border_width"], zorder=2,
                    ))
                t = ax.text(
                    cx, cy, str(day), ha='center', va='center',
                    color=num_color, fontsize=CALENDAR["day_number_fontsize"], zorder=3,
                )
                centered.append((t, (cx, cy)))
                continue

            # Primary = largest magnitude; the rest become secondary circles.
            acts = sorted(
                acts,
                key=lambda a: a['size_value'] if pd.notna(a['size_value']) else 0,
                reverse=True,
            )
            primary = acts[0]
            r_primary = _calendar_radius(primary['size_value'], primary['sport'])
            ax.add_patch(Circle(
                (cx, cy), r_primary,
                facecolor=primary['color'], edgecolor=STYLE["bar_edge_color"],
                linewidth=CALENDAR["border_width"], zorder=2,
            ))
            t = ax.text(
                cx, cy, SPORT_LETTERS.get(primary['sport'], '?'),
                ha='center', va='center',
                color=_text_color_for(primary['color']),
                fontsize=_letter_fontsize(r_primary), weight='bold', zorder=3,
            )
            centered.append((t, (cx, cy)))

            # Secondary activities march from the upper-right corner (robust for 3+).
            for i, act in enumerate(acts[1:]):
                sx = cx + CALENDAR["secondary_offset_x"] + i * CALENDAR["secondary_step_x"]
                sy = cy + CALENDAR["secondary_offset_y"] + i * CALENDAR["secondary_step_y"]
                rs = _calendar_radius(act['size_value'], act['sport']) * CALENDAR["secondary_size_mult"]
                ax.add_patch(Circle(
                    (sx, sy), rs,
                    facecolor=act['color'], edgecolor=STYLE["background_color"],
                    linewidth=CALENDAR["border_width"] * 0.7, zorder=4,
                ))
                t = ax.text(
                    sx, sy, SPORT_LETTERS.get(act['sport'], '?'),
                    ha='center', va='center',
                    color=_text_color_for(act['color']),
                    fontsize=_letter_fontsize(rs), weight='bold', zorder=5,
                )
                centered.append((t, (sx, sy)))

    # --- Legend table (index letter | Activity | Size | Color) + metric colorbar ---
    fs = CALENDAR["legend_fontsize"]
    col_x = {"letter": 0.5, "activity": 1.1, "size": 2.9, "color": 4.2}
    rows_tbl = [
        ("T", "Trail Run", "Distance", "Speed"),
        ("R", "Run",       "Distance", "Speed"),
        ("H", "Hike",      "Distance", "Carried weight"),
        ("S", "Strength",  "Volume",   "Session time"),
        ("B", "Bike",      "Distance", "Speed"),
    ]
    row_step = 0.34
    table_top = grid_bottom - 0.55

    # Header row (the index/letter column header stays empty).
    for key, label in (("activity", "Activity"), ("size", "Size"), ("color", "Color")):
        ax.text(col_x[key], table_top, label, ha='left', va='center',
                color=STYLE["text_color"], fontsize=fs, weight='bold')
    # Faint rule under the header.
    ax.plot([col_x["letter"] - 0.25, 6.0], [table_top - row_step * 0.5] * 2,
            color=STYLE["grid_color"], alpha=0.5, linewidth=0.8, zorder=2)

    for i, (letter, activity, size_m, color_m) in enumerate(rows_tbl, start=1):
        y = table_top - i * row_step
        ax.text(col_x["letter"], y, letter, ha='center', va='center',
                color=STYLE["neutral_color"], fontsize=fs, weight='bold')
        ax.text(col_x["activity"], y, activity, ha='left', va='center',
                color=STYLE["neutral_color"], fontsize=fs)
        ax.text(col_x["size"], y, size_m, ha='left', va='center',
                color=STYLE["text_color"], fontsize=fs)
        ax.text(col_x["color"], y, color_m, ha='left', va='center',
                color=STYLE["text_color"], fontsize=fs)

    # Metric colorbar below the table (low → high), kept thin (~3x shorter than the rows).
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    bar_w, bar_h = 2.4, 0.09
    y_cbar = table_top - (len(rows_tbl) + 1) * row_step
    ax.imshow(
        grad, cmap=metric_colormap(), aspect='auto',
        extent=[3.0 - bar_w / 2, 3.0 + bar_w / 2, y_cbar - bar_h / 2, y_cbar + bar_h / 2],
        zorder=3,
    )
    ax.text(3.0 - bar_w / 2 - 0.15, y_cbar, "Color  low", ha='right', va='center',
            color=STYLE["text_color"], fontsize=fs)
    ax.text(3.0 + bar_w / 2 + 0.15, y_cbar, "high", ha='left', va='center',
            color=STYLE["text_color"], fontsize=fs)

    if title is None:
        title = f"{_calendar.month_name[month]} {year}"
    ax.set_title(
        title, color=STYLE["highlight_color"],
        fontsize=STYLE["title_fontsize"], weight=STYLE["title_weight"], pad=12,
    )

    # Lay out first, then geometrically center each circle glyph: after a draw we know its
    # rendered box, so we shift the anchor so the box center lands exactly on the circle
    # center. transData is affine, so a single pass is exact. Done after tight_layout so the
    # final axes scale (not the pre-layout one) is used.
    plt.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for t, (tx, ty) in centered:
        ax_disp = ax.transData.transform((tx, ty))
        bb = t.get_window_extent(renderer=renderer)
        box_center = ((bb.x0 + bb.x1) / 2.0, (bb.y0 + bb.y1) / 2.0)
        new_disp = (2 * ax_disp[0] - box_center[0], 2 * ax_disp[1] - box_center[1])
        t.set_position(inv.transform(new_disp))

    if save_name:
        plt.savefig(os.path.join(SAVE_FOLDER, save_name), dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
