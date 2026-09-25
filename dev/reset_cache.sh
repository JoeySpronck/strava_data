#!/usr/bin/env bash
# Clear the on-disk Strava cache so the next run re-fetches from the API.
#
# Usage:
#   ./reset_cache.sh            # clear everything
#   ./reset_cache.sh details    # only descriptions / private notes
#   ./reset_cache.sh streams    # only run velocity streams (notebook speed plot)
#   ./reset_cache.sh split      # only streams of runs tagged "<int>% hike"
set -euo pipefail

# Run from the script's own folder so it works no matter where it's called from
cd "$(dirname "$0")"
CACHE_DIR="../.cache"  # the repo-root cache, shared with update_plots.py

remove() {
  if [ -f "$CACHE_DIR/$1" ]; then
    rm "$CACHE_DIR/$1"
    echo "Removed $CACHE_DIR/$1"
  else
    echo "Nothing to remove: $CACHE_DIR/$1 (already clear)"
  fi
}

case "${1:-all}" in
  details) remove "activity_details.json" ;;
  streams) remove "velocity_streams.json" ;;
  split) remove "split_streams.json" ;;
  all)
    remove "activity_details.json"
    remove "velocity_streams.json"
    remove "split_streams.json"
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Usage: ./reset_cache.sh [all|details|streams|split]" >&2
    exit 1
    ;;
esac

echo "Done. Next run will re-fetch the cleared data from Strava."
