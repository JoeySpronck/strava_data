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
MARKDOWN_FILES = ("README.md", "CALENDAR_PLOTS.md")


def add_token(token, plots_dir=PLOTS_DIR, markdown_files=MARKDOWN_FILES):
    # Keep only filename/URL-safe characters so the token can't break paths.
    token = re.sub(r"[^0-9A-Za-z_-]", "", token)
    if not token:
        raise ValueError("token is empty after sanitisation")

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

    for md in markdown_files:
        if not os.path.exists(md):
            continue
        with open(md) as fh:
            text = fh.read()
        for clean_ref, tokened_ref in renames.items():
            # The full ref includes '.png', and that extension is the boundary: e.g.
            # 'plots/weekly_distance.png' can't match inside 'plots/weekly_distance_targets.png'.
            text = text.replace(clean_ref, tokened_ref)
        with open(md, "w") as fh:
            fh.write(text)

    print(f"Added token '{token}' to {len(renames)} plot(s).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        from datetime import datetime, timezone
        token = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    add_token(token)
