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

- **`update_plots.py`** — syncs the Strava mirror, then regenerates every plot in `plots/` and exports the web data.
- **`strava_data/`** — the package: API client, data wrangling, and all the plotting/visualization logic.
  - `store.py` / `sync.py` — a per-athlete mirror of the account in `.cache/`. Every activity's description and sample streams are fetched **exactly once**, newest first, inside a fixed request budget, so a run stays far under Strava's limits (100 reads / 15 min, 1000 / day) however big the history gets. The Action persists `.cache/` between runs; a first backfill spreads over a few days, after which a run costs a handful of requests.
  - `streams.py` / `decoupling.py` — stream cleanup into compact JSON, and the aerobic-decoupling maths (Minetti graded pace, 30 s 4th-power normalization, efficiency factor).
- **`webhook/`** — a Cloudflare Worker that listens for Strava activity changes and triggers a plot refresh (debounced, so bursts of edits run once). Optional — see [`webhook/README.md`](webhook/README.md) to set it up.
- **`web/`** — a styled HTML/CSS dashboard published via GitHub Pages (the *Open Web Dashboard* button above), including the interactive **aerobic decoupling** page. Optional — see [`web/README.md`](web/README.md) to enable it.
- **`tests/`** — two checks, needing only Python, numpy and node:
  - `python tests/test_decoupling_parity.py` runs the Python and JavaScript decoupling maths over the same activity and fails if any number disagrees, which is what lets the notebook and the web page claim to show the same thing.
  - `node tests/test_page_smoke.mjs` drives the decoupling page against a stubbed DOM and Plotly — load an activity, drag each handle, switch basemap. `node --check` only parses; this catches a helper lost to an edit or a map trace the Plotly bundle does not have.
- **GitHub Actions** (`.github/workflows/update_plots.yml`) — runs `update_plots.py` daily, on push to `main`, manually via github actions, and on webhook trigger (when an activity is added/edited on strava), then publishes the regenerated plots and dashboard to the `plot_updates` branch.
- **`dev/`** contains `playground.ipynb`, whis is a development notebook version of `update_plots.py`, `aerobic_decoupling.ipynb` (the interactive analyzer the web page is built from), and other development files. 

### Aerobic decoupling

`web/decoupling.html` is the one page here that is not a picture: pick a session, drag two
intervals over it, and read how far efficiency (normalized graded pace per heartbeat)
drifted between them. The Action exports each activity's sample streams as JSON next to
the plots and **all the analysis runs in the browser** — no server, no API call per
visitor, works on a phone. `web/decoupling.js` is a line-by-line port of
`strava_data/decoupling.py`, pinned to it by the parity test.

Only activities you have not marked *Only you* on Strava are exported, and descriptions
and private notes never leave `.cache/`. The exported JSON does include GPS traces, and
GitHub Pages is world-readable — `PUBLISHABLE_VISIBILITIES` in
[`strava_data/sync.py`](strava_data/sync.py) is the one place to change that.

I generally view the `plot_updates` branch via the Github app on my phone.
The published `plot_updates` branch carries cache-busted image filenames (a per-run token)
so the GitHub app stops serving stale images — `main` keeps clean filenames for development.
