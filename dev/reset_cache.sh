#!/usr/bin/env bash
# Clear the on-disk Strava cache so the next run re-fetches from the API.
#
# Usage:
#   ./reset_cache.sh            # clear everything
#   ./reset_cache.sh details    # only descriptions / private notes (hikes & strength)
#   ./reset_cache.sh streams    # only run velocity streams (notebook speed plot)
set -euo pipefail

# Run from the script's own folder so it works no matter where it's called from
cd "$(dirname "$0")"
CACHE_DIR=".cache"

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
  all)
    remove "activity_details.json"
    remove "velocity_streams.json"
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Usage: ./reset_cache.sh [all|details|streams]" >&2
    exit 1
    ;;
esac

echo "Done. Next run will re-fetch the cleared data from Strava."
