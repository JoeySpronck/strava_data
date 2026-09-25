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

### Tags in activity notes

A few words in an activity's title, description or private note change how it's counted:

- **`<int>% hike`** on a run (e.g. `~30% hiked`) — the run is split into a run part and a
  hike part. The cadence stream decides which stretches were hiked; the percentage is only
  a sanity check (a warning is printed if they disagree by more than 15 percentage points).
  The run part keeps only the running distance and time, so volume and pace reflect the
  running; the hike part shows up in the hiking plots with 0 kg carried. Without cadence,
  the slowest `p%` of distance counts as hiked; without any stream, the distance is split
  by `p`. A note without a percentage ("walked the uphills") changes nothing.
- **`multisport`** (or `multi sport` / `multi-sport`) on two or more activities on the same day
  links them, whatever the time between them.
- **`<n>kg`** on a hike is the carried weight; **`<n>kg volume`** on a strength session its volume.

Same-day activities also link without a tag when one starts at most 60 minutes after the
previous one ends. Linked activities get the letter of the other sport (R run, T trail run,
H hike, S strength, B bike) on their bar in every weekly stacked plot, with a small arrow
above it: → that activity came after, ← it came before, ↔ the other part of a split run.
In a chain of three or more, the middle activities point to the next one.
The logic lives in `strava_data/hike_split.py`; `python tests/test_hike_split.py` tests it
on synthetic streams.

### Aerobic decoupling

`web/decoupling.html` analyses one run at a time from a file you download from Strava:
open the activity, ⋯ → **Export GPX**, or **Export Original** for the watch's own `.fit`
(better: it carries the watch's distance and speed instead of ones derived from GPS
points). Drop the file on the page, drag two intervals over the session, and read how far
efficiency (normalized graded pace per heartbeat) drifted between them. The file is
parsed and analysed in the browser and never uploaded, so no activity data is published
anywhere. The page is standalone (no links to the dashboard or repo), so its URL can be
shared on its own. `dev/aerobic_decoupling.ipynb` does the same from a file path.

- `strava_data/activity_file.py` / `web/activity_file.js` — GPX and FIT into one stream document.
- `strava_data/decoupling.py` / `web/decoupling.js` — Minetti graded pace, 30 s 4th-power normalization, efficiency factor.

Each JavaScript file is a port of its Python twin, and two tests (Python, numpy, fitdecode and node) keep them honest:

- `python tests/test_decoupling_parity.py` parses a synthetic run exported both ways (`tests/fixtures/synthetic_run.gpx` / `.fit`, written by `tests/make_export_fixtures.py`) in both languages and requires identical samples, then runs both decoupling implementations over it and fails on any number that disagrees. Pass your own exports as arguments to check them too; they are only read. Real exports are gitignored and never belong in the repo — they are your GPS trace.
- `node tests/test_page_smoke.mjs` drives the page against a stubbed DOM and Plotly: choose a GPX, drag each handle, switch basemap, drop a FIT, drop a file that is not an activity.

I generally view the `plot_updates` branch via the Github app on my phone.
The published `plot_updates` branch carries cache-busted image filenames (a per-run token)
so the GitHub app stops serving stale images — `main` keeps clean filenames for development.
