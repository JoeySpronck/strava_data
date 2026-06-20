"""Add a cache-busting token to every plot filename and patch the markdown references.

GitHub (especially the mobile app) caches images by URL, so identical plot filenames keep
serving the old image after an update. This appends a single per-run token to every plot
filename (e.g. weekly_distance.png -> weekly_distance.20260619161320.png) and rewrites the
markdown references to match, so every image gets a fresh URL.

It is meant to run ONLY in the Action's step that prepares the plot_updates branch, never in
update_plots.py — so `main` keeps clean, hand-editable filenames (nice for development) and
only the force-rebuilt plot_updates branch carries the token. Because plot_updates is recreated
from main on every run, the tokened files never accumulate in git history.

Run from the repo root (paths below are relative to it):
    python .github/helper_scripts/add_cache_token_to_image_names.py [TOKEN]

TOKEN defaults to a UTC timestamp (YYYYMMDDHHMMSS), so local runs work without arguments.
The Action passes the same timestamp so the token is readable as the generation time.
"""
import os
import re
import sys

PLOTS_DIR = "plots"
# Files whose plot image references get the cache-busting token. PLOTS.md is the dashboard
# the Action renames to README.md on the plot_updates branch; web/ holds the GitHub Pages site
# (its pages reference ../plots/..., which still contains the 'plots/...' substring we rewrite).
# (main keeps a short, repo-focused README.md untouched.)
PATCHED_FILES = ("PLOTS.md", "CALENDAR_PLOTS.md", "web/index.html", "web/calendar.html")

# HTML pages carry a __BUILD_TOKEN__ placeholder we replace with the human-readable
# generation time, so the site shows when it was last updated.
BUILD_PLACEHOLDER = "__BUILD_TOKEN__"


def _format_build_time(token):
    """Turn a YYYYMMDDHHMMSS token into 'YYYY-MM-DD HH:MM UTC'; fall back to the raw token."""
    if len(token) == 14 and token.isdigit():
        t = token
        return f"{t[0:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]} UTC"
    return token


def add_token(token, plots_dir=PLOTS_DIR, patched_files=PATCHED_FILES):
    # Keep only filename/URL-safe characters so the token can't break paths.
    token = re.sub(r"[^0-9A-Za-z_-]", "", token)
    if not token:
        raise ValueError("token is empty after sanitisation")
    build_time = _format_build_time(token)

    suffix = f".{token}.png"
    # renames: clean relative ref -> tokened relative ref
    #   'plots/weekly_distance.png' -> 'plots/weekly_distance.<token>.png'
    renames = {}
    for root, _, files in os.walk(plots_dir):
        for fname in files:
            if not fname.endswith(".png") or fname.endswith(suffix):
                continue
            tokened = fname[: -len(".png")] + suffix
            os.rename(os.path.join(root, fname), os.path.join(root, tokened))
            # forward slashes for markdown/URLs regardless of OS
            clean_ref = os.path.join(root, fname).replace(os.sep, "/")
            tokened_ref = os.path.join(root, tokened).replace(os.sep, "/")
            renames[clean_ref] = tokened_ref

    for fname in patched_files:
        if not os.path.exists(fname):
            continue
        with open(fname) as fh:
            text = fh.read()
        for clean_ref, tokened_ref in renames.items():
            # The full ref includes '.png', and that extension is the boundary: e.g.
            # 'plots/weekly_distance.png' can't match inside 'plots/weekly_distance_targets.png'.
            text = text.replace(clean_ref, tokened_ref)
        # Stamp the generation time into the HTML pages (no-op for the markdown files).
        text = text.replace(BUILD_PLACEHOLDER, build_time)
        with open(fname, "w") as fh:
            fh.write(text)

    print(f"Added token '{token}' to {len(renames)} plot(s).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        from datetime import datetime, timezone
        token = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    add_token(token)
