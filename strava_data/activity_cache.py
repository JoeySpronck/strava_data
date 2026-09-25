"""Persistent, id-keyed caches for per-activity Strava detail calls.

Strava's summary API (``get_activities``) omits ``description``/``private_note`` and
stream data, so each activity we want those for costs one extra ``get_activity`` /
``get_activity_streams`` request. With ~60 hikes+strength sessions and ~100 runs that
blows past the short-term rate limit (~100 req / 15 min) on a single run.

These helpers persist results to ``.cache/`` keyed by activity id. Since activity
details don't change, re-runs fetch only *new* activities. If a fetch is rate-limited
mid-run we stop early and keep what we got — the next run resumes from the cache.
With ``STRAVA_RATE_LIMIT_WAITS=n`` (set in CI) we instead wait for Strava's next
15-minute window and carry on, at most n times per process.

Delete ``.cache/*.json`` to force a full refresh (e.g. after editing descriptions).
"""
import json
import os
import time

import numpy as np
import pandas as pd

# Anchored to the repo root (not the CWD) so the notebook in dev/ and update_plots.py at
# the root share one cache, like SAVE_FOLDER in visualization.py does for plots/.
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
DETAILS_FILE = os.path.join(CACHE_DIR, "activity_details.json")
STREAMS_FILE = os.path.join(CACHE_DIR, "velocity_streams.json")
SPLIT_STREAMS_FILE = os.path.join(CACHE_DIR, "split_streams.json")

# How often one process may sleep through a rate limit instead of stopping early. Strava
# resets its 15-minute window on the quarter hour (:00, :15, :30, :45). 0 = stop at once
# (the default, so a notebook cell never blocks for 15 minutes); CI sets it via env.
MAX_RATE_LIMIT_WAITS = int(os.environ.get("STRAVA_RATE_LIMIT_WAITS", "0") or 0)
_waits_used = 0


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)  # atomic: never leave a half-written cache


def _is_rate_limit(error):
    """True for Strava's HTTP 429 (stravalib raises it as a plain Fault)."""
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 429


def _daily_limit_reached(error):
    """Whether the 429 is the *daily* limit, which waiting 15 minutes won't clear.

    Strava sends "15-min,daily" pairs in X-ReadRateLimit-* (reads) and X-RateLimit-*.
    """
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    for prefix in ("X-ReadRateLimit", "X-RateLimit"):
        try:
            limit = [int(x) for x in headers[f"{prefix}-Limit"].split(",")]
            usage = [int(x) for x in headers[f"{prefix}-Usage"].split(",")]
        except (KeyError, ValueError):
            continue
        if len(limit) > 1 and len(usage) > 1 and usage[1] >= limit[1]:
            return True
    return False


def _seconds_to_next_window(now=None):
    """Seconds until the next quarter hour, plus a small margin for clock skew."""
    now = time.time() if now is None else now
    return 900 - (now % 900) + 15


def _fetch_missing(missing, fetch_one, cache, cache_path, label, verbose):
    """Fill ``cache[str(id)] = fetch_one(id)`` for each id, saving as it goes.

    On an error: if it's a 15-minute rate limit and wait budget is left, save, sleep to the
    next window and retry the same id; otherwise stop early (next run resumes from cache).
    Returns True if everything was fetched.
    """
    global _waits_used
    fetched_any = False
    i = 0
    while i < len(missing):
        aid = missing[i]
        try:
            cache[str(aid)] = fetch_one(aid)
            fetched_any = True
            i += 1
            continue
        except Exception as e:
            can_wait = (_is_rate_limit(e) and not _daily_limit_reached(e)
                        and _waits_used < MAX_RATE_LIMIT_WAITS)
            if fetched_any:
                _save_json(cache_path, cache)  # keep progress even if the job dies while waiting
                fetched_any = False
            if not can_wait:
                if verbose:
                    print(f"  stopped early on activity {aid}: {e}. "
                          f"{len(missing) - i} {label} left — re-run later to fetch the rest.")
                return False
            _waits_used += 1
            wait = _seconds_to_next_window()
            if verbose:
                print(f"  rate limited on activity {aid}; waiting {wait / 60:.1f} min for the "
                      f"next window (wait {_waits_used}/{MAX_RATE_LIMIT_WAITS}), "
                      f"{len(missing) - i} {label} left...")
            time.sleep(wait)

    if fetched_any:
        _save_json(cache_path, cache)
    return True


def _describe(missing, cache, refresh, noun="activities"):
    """E.g. "5 activities (2 new, 3 refreshed)": new = not cached yet, refreshed = cached
    but refetched anyway because they're in refresh_ids (recent, or edited)."""
    refreshed = sum(1 for aid in missing if str(aid) in cache and str(aid) in refresh)
    return f"{len(missing)} {noun} ({len(missing) - refreshed} new, {refreshed} refreshed)"


def fetch_text_fields(client, ids, cache_path=DETAILS_FILE, verbose=True, refresh_ids=()):
    """DataFrame[id, description, private_note], fetching only uncached ids.

    ``refresh_ids`` are refetched even when cached, so edited notes are picked up
    (the webhook passes the ids it saw; update_plots adds the last few days).

    On a rate limit we wait (see MAX_RATE_LIMIT_WAITS) or stop early, persist what we
    have, and fall back to the cached value (or None) for the rest this run.
    """
    cache = _load_json(cache_path)
    refresh = {str(a) for a in refresh_ids}
    missing = [aid for aid in ids if str(aid) not in cache or str(aid) in refresh]
    if missing and verbose:
        print(f"Fetching details for {_describe(missing, cache, refresh)}, "
              f"{len(ids) - len(missing)} from cache...")

    def fetch_one(aid):
        d = dict(client.get_activity(aid))
        return {"description": d.get("description"), "private_note": d.get("private_note")}

    _fetch_missing(missing, fetch_one, cache, cache_path, "activities", verbose)

    empty = {"description": None, "private_note": None}
    rows = [{"id": aid, **cache.get(str(aid), empty)} for aid in ids]
    return pd.DataFrame(rows)


def fetch_velocity_streams(client, ids, cache_path=STREAMS_FILE, verbose=True):
    """{id: np.ndarray | None} velocity_smooth streams, fetching only uncached ids.

    A successful fetch with no stream is cached as None (won't re-fetch); a *failed*
    fetch is left uncached so it retries next run. Rate limits behave like above.
    """
    cache = _load_json(cache_path)
    missing = [aid for aid in ids if str(aid) not in cache]
    if missing and verbose:
        print(f"Fetching velocity streams for {len(missing)} new runs "
              f"({len(ids) - len(missing)} from cache)...")

    def fetch_one(aid):
        streams = client.get_activity_streams(aid, types=["velocity_smooth"])
        stream = streams.get("velocity_smooth") if streams else None
        return [float(x) for x in stream.data] if stream and stream.data else None

    _fetch_missing(missing, fetch_one, cache, cache_path, "runs", verbose)

    out = {}
    for aid in ids:
        data = cache.get(str(aid))  # None if uncached-this-run or genuinely streamless
        out[aid] = np.asarray(data, dtype=float) if data else None
    return out


def fetch_split_streams(client, ids, types, cache_path=SPLIT_STREAMS_FILE, verbose=True,
                        refresh_ids=()):
    """{id: {type: list} | None} streams for the run/hike split, fetching only uncached ids.

    Only called for the handful of runs tagged ``<int>% hike``. Caching and rate limits
    behave like ``fetch_velocity_streams``; ``refresh_ids`` are refetched even if cached.
    """
    cache = _load_json(cache_path)
    refresh = {str(a) for a in refresh_ids}
    missing = [aid for aid in ids if str(aid) not in cache or str(aid) in refresh]
    if missing and verbose:
        print(f"Fetching split streams for {_describe(missing, cache, refresh, 'tagged runs')}, "
              f"{len(ids) - len(missing)} from cache...")

    def fetch_one(aid):
        streams = client.get_activity_streams(aid, types=types) or {}
        data = {k: list(st.data) for k, st in streams.items() if st is not None and st.data}
        return data or None

    _fetch_missing(missing, fetch_one, cache, cache_path, "tagged runs", verbose)

    return {aid: cache.get(str(aid)) for aid in ids}
