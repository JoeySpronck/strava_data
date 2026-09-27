"""Manually logged injuries / sickness, drawn as an event timeline in the all-sports overview.

Add a row per injury: duration is in weeks, color a key of visualization.COLORS for the
circle. 'main' injuries also get an orange band over their duration in the sport panels.
Shared by update_plots.py and dev/playground.ipynb.
"""
import pandas as pd

INJURIES = [
    # start_date,  duration, abbreviation, full_name, color
    ('2025-06-01', 3, 'PF', 'Plantar Fasciitis', 'main'),
    ('2025-07-23', 3, 'PF', 'Plantar Fasciitis', 'main'),
    ('2025-08-18', 2, 'IT', 'IT Band', 'main'),
    ('2026-02-18', 3, 'SI', 'Sick', 'dark'),
    ('2026-04-13', 3, 'SS', 'Shin Splints', 'main'),
    ('2026-05-06', 4, 'BS', 'Bone Stress', 'main'),
    ('2026-06-21', 1, 'IT', 'IT Band', 'dark'),
    ('2026-08-19', 6, 'BS', 'Bone Stress', 'main'),
    ('2026-09-02', 1, 'TT', 'Tibial Tendinopathy', 'dark'),
    # ('2026-09-14', 4, 'BS', 'Bone Stress', 'main'),
]


def injuries_df():
    df = pd.DataFrame(INJURIES, columns=['start_date', 'duration', 'abbreviation', 'full_name', 'color'])
    df['start_date'] = pd.to_datetime(df['start_date'])
    return df


def injury_panel(df=None):
    """Events panel for vis.plot_weekly_stacked_multi."""
    df = injuries_df() if df is None else df
    return dict(
        kind='events',
        df=df.rename(columns={'start_date': 'date', 'abbreviation': 'label', 'full_name': 'description'}),
        title='Injuries',
    )
