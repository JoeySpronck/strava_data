# STRAVA DATA: Weekly Running Plans & Progress
###### Joey Spronck

A self-updating training dashboard built on the Strava API. It pulls my activities,
builds weekly volume targets and example week plans, and renders a set of progress
plots that refresh automatically. The plots live on the `plot_updates` branch so `main` stays clean.

<p align="left">
  <a href="https://github.com/JoeySpronck/strava_data/tree/plot_updates/README.md">
    <img src="https://img.shields.io/badge/📈_View_Live_Training_Plots-FB5200?style=flat" alt="View Live Training Plots" width="320">
  </a>
  
  <a href="https://joeyspronck.github.io/strava_data/web/">
    <img src="https://img.shields.io/badge/🌐_Open_Web_Dashboard-1a1a1a?style=flat" alt="Open Web Dashboard" width="300">
  </a>
</p>

---

## How it works

- **`update_plots.py`** — fetches activities via the Strava API and regenerates every plot in `plots/`.
- **`strava_data/`** — the package: API client, data wrangling, and all the plotting/visualization logic.
- **`webhook/`** — a Cloudflare Worker that listens for Strava activity changes and triggers a plot refresh (debounced, so bursts of edits run once). Optional — see [`webhook/README.md`](webhook/README.md) to set it up.
- **`web/`** — a styled HTML/CSS dashboard published via GitHub Pages (the *Open Web Dashboard* button above), including the interactive **aerobic decoupling** page. Optional — see [`web/README.md`](web/README.md) to enable it.
- **GitHub Actions** (`.github/workflows/update_plots.yml`) — runs `update_plots.py` daily, on push to `main`, manually via github actions, and on webhook trigger (when an activity is added/edited on strava), then publishes the regenerated plots and dashboard to the `plot_updates` branch.
- **`dev/`** contains `playground.ipynb`, whis is a development notebook version of `update_plots.py`, `aerobic_decoupling.ipynb` (the notebook version of the decoupling page), and other development files. 

### Aerobic decoupling

`web/decoupling.html` analyses one run at a time from a file you download from Strava:
open the activity, ⋯ → **Export GPX**, or **Export Original** for the watch's own `.fit`
(better: it carries the watch's distance and speed instead of ones derived from GPS
points). Drop the file on the page, drag two intervals over the session, and read how far
efficiency (normalized graded pace per heartbeat) drifted between them. The file is
parsed and analysed in the browser and never uploaded, so no activity data is published
anywhere. `dev/aerobic_decoupling.ipynb` does the same from a file path.

- `strava_data/activity_file.py` / `web/activity_file.js` — GPX and FIT into one stream document.
- `strava_data/decoupling.py` / `web/decoupling.js` — Minetti graded pace, 30 s 4th-power normalization, efficiency factor.

Each JavaScript file is a port of its Python twin, and two tests (Python, numpy, fitdecode and node) keep them honest:

- `python tests/test_decoupling_parity.py` parses a synthetic run exported both ways (`tests/fixtures/synthetic_run.gpx` / `.fit`, written by `tests/make_export_fixtures.py`) in both languages and requires identical samples, then runs both decoupling implementations over it and fails on any number that disagrees. Pass your own exports as arguments to check them too; they are only read. Real exports are gitignored and never belong in the repo — they are your GPS trace.
- `node tests/test_page_smoke.mjs` drives the page against a stubbed DOM and Plotly: choose a GPX, drag each handle, switch basemap, drop a FIT, drop a file that is not an activity.

I generally view the `plot_updates` branch via the Github app on my phone.
The published `plot_updates` branch carries cache-busted image filenames (a per-run token)
so the GitHub app stops serving stale images — `main` keeps clean filenames for development.
