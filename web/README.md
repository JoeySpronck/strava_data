# Web dashboard (GitHub Pages)

> **This folder is optional.** It's just a nicer way to *view* the same plots.
>
> The plots already render as markdown on the `plot_updates` branch (handy in the
> GitHub mobile app). This folder adds a styled HTML/CSS dashboard you can publish
> with **GitHub Pages** to get a real URL that behaves like a small web app
> (add it to your phone's home screen and it opens fullscreen).

Once enabled, the site lives at:

```
https://<your-username>.github.io/<your-repo>/web/
```

---

## What's in here

| File | What it is |
|------|------------|
| `index.html` | The dashboard. Mirrors the sections of `PLOTS.md`, pulling images from `../plots/`. |
| `style.css` | Shared dark theme (black background, Strava orange). Used by both pages. |
| `calendar.html` | The full monthly-calendar archive. **Generated** by `update_plots.py` — don't hand-edit. |
| `decoupling.html` | Interactive aerobic-decoupling analyzer for one run: drop in a Strava export, drag two intervals, read the drift. Standalone and shareable: it links to nothing else on the site, so people you send it to don't land on the dashboard. |
| `activity_file.js` | Reads a `.gpx` or `.fit` into sample streams — a port of `strava_data/activity_file.py`. |
| `decoupling.js` | The maths behind that page — a port of `strava_data/decoupling.py`. Pure functions, no DOM. |
| `README.md` | This file. |

How it stays fresh: the same GitHub Action that regenerates the plots also
rewrites the image URLs here with a per-run cache-busting token and stamps the
"Updated …" time (see [`.github/helper_scripts/add_cache_token_to_image_names.py`](../.github/helper_scripts/add_cache_token_to_image_names.py)).
So Pages updates on the exact same daily / push / webhook cadence as the plots.

The decoupling page needs no data from the Action: you give it a file exported from
Strava (⋯ → *Export GPX*, or *Export Original* for a watch's `.fit`), and it is parsed and
analysed in the browser without being uploaded. `tests/test_decoupling_parity.py` pins
both JavaScript files to their Python twins; `node tests/test_page_smoke.mjs` covers the
page's wiring.

> **Why `web/` and not `docs/`?** GitHub Pages can serve from the branch root or a
> `docs/` folder. The plot images live at the repo root (`plots/`), so the site
> stays at the root too and these pages reference `../plots/...`. That's why the
> public URL ends in `/web/`. (`.nojekyll` sits at the **repo root**, not in here —
> Pages needs it at the served root to skip Jekyll.)

---

## Enable GitHub Pages (one-time)

Do this **after** the Action has run at least once and published the
`plot_updates` branch.

1. Go to your repo's **Settings → Pages**.
2. Under **Build and deployment → Source**, pick **Deploy from a branch**.
3. Set **Branch** to `plot_updates` and the folder to **`/ (root)`**, then **Save**.

Or, equivalently, from the CLI:

```bash
gh api -X POST repos/<your-username>/<your-repo>/pages \
  -f "source[branch]=plot_updates" \
  -f "source[path]=/"
```

Give Pages ~1 minute to build, then open
`https://<your-username>.github.io/<your-repo>/web/`.

That's it — nothing else to configure. Pages rebuilds automatically every time
the Action force-pushes `plot_updates`.

---

## Preview locally

From the repo root:

```bash
python update_plots.py        # regenerates plots + web/calendar.html
python -m http.server         # serve the repo
# open http://localhost:8000/web/
```

(The "Updated" stamp shows a literal `__BUILD_TOKEN__` placeholder locally; the
Action fills it in on publish.)

The decoupling page loads its scripts as ES modules, which browsers refuse over `file://`,
so open it through the server (http://localhost:8000/web/decoupling.html), not by
double-clicking the file.
