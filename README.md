# STRAVA DATA: Weekly Running Plans & Progress
###### Joey Spronck

A self-updating training dashboard built on the Strava API. It pulls my activities,
builds weekly volume targets and example week plans, and renders a set of progress
plots that refresh automatically. The plots live on the `plot_updates` branch so `main` stays clean.

<p align="left">
  <a href="https://github.com/JoeySpronck/strava_data/tree/plot_updates/README.md">
    <img src="https://img.shields.io/badge/📈_View_Live_Training_Plots-FB5200?style=flat" alt="View Live Training Plots" width="320">
  </a>
</p>

---

## How it works

- **`update_plots.py`** — fetches activities via the Strava API and regenerates every plot in `plots/`.
- **`strava_data/`** — the package: API client, data wrangling, and all the plotting/visualization logic.
- **`webhook/`** — a Cloudflare Worker that listens for Strava activity changes and triggers a plot refresh (debounced, so bursts of edits run once). Configuring this is optional.
- **GitHub Actions** (`.github/workflows/update_plots.yml`) — runs `update_plots.py` daily, on push to `main`, manually via github actions, and on webhook trigger (when an activity is added/edited on strava), then publishes the regenerated plots and dashboard to the `plot_updates` branch.
- **`dev/`** contains `playground.ipynb`, whis is a development notebook version of `update_plots.py`, and other development files. 

I generally view the `plot_updates` branch via the Github app on my phone.
The published `plot_updates` branch carries cache-busted image filenames (a per-run token)
so the GitHub app stops serving stale images — `main` keeps clean filenames for development.
