"""Manually logged injuries / sickness, drawn as an event timeline in the all-sports overview.

Add a row per injury; end_date is kept for later use (not drawn yet). Shared by
update_plots.py and dev/playground.ipynb.
"""
import pandas as pd

INJURIES = [
    # start_date,  end_date, abbreviation, full_name
    ('2025-06-01', None, 'PF', 'Plantar Fasciitis'),
    ('2025-07-23', None, 'PF', 'Plantar Fasciitis'),
    ('2025-08-18', None, 'IT', 'IT Band'),
    ('2026-02-18', None, 'SI', 'Sick'),
    ('2026-04-13', None, 'SS', 'Shin Splints'),
    ('2026-05-06', None, 'BS', 'Bone Stress'),
    ('2026-06-21', None, 'IT', 'IT Band'),
    ('2026-08-19', None, 'BS', 'Bone Stress'),
    ('2026-09-02', None, 'TT', 'Tibial Tendinopathy'),
    ('2026-09-14', None, 'BS', 'Bone Stress'),
]


def injuries_df():
    df = pd.DataFrame(INJURIES, columns=['start_date', 'end_date', 'abbreviation', 'full_name'])
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    return df


def injury_panel(df=None):
    """Events panel for vis.plot_weekly_stacked_multi."""
    df = injuries_df() if df is None else df
    return dict(
        kind='events',
        df=df.rename(columns={'start_date': 'date', 'abbreviation': 'label', 'full_name': 'description'}),
        title='Injuries',
    )
