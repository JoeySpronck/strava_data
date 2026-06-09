"""Persistent, id-keyed caches for per-activity Strava detail calls.

Strava's summary API (``get_activities``) omits ``description``/``private_note`` and
stream data, so each activity we want those for costs one extra ``get_activity`` /
``get_activity_streams`` request. With ~60 hikes+strength sessions and ~100 runs that
blows past the short-term rate limit (~100 req / 15 min) on a single run.

These helpers persist results to ``.cache/`` keyed by activity id. Since activity
details don't change, re-runs fetch only *new* activities. If a fetch is rate-limited
mid-run we stop early and keep what we got — the next run resumes from the cache.

Delete ``.cache/*.json`` to force a full refresh (e.g. after editing descriptions).
"""
import json
import os

import numpy as np
import pandas as pd

CACHE_DIR = ".cache"
DETAILS_FILE = os.path.join(CACHE_DIR, "activity_details.json")
STREAMS_FILE = os.path.join(CACHE_DIR, "velocity_streams.json")


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


def fetch_text_fields(client, ids, cache_path=DETAILS_FILE, verbose=True):
    """DataFrame[id, description, private_note], fetching only uncached ids.

    On a rate-limit (or any) error we stop fetching, persist what we have, and fall
    back to None for the rest this run; the next run picks up where we left off.
    """
    cache = _load_json(cache_path)
    missing = [aid for aid in ids if str(aid) not in cache]
    if missing and verbose:
        print(f"Fetching details for {len(missing)} new activities "
              f"({len(ids) - len(missing)} from cache)...")

    fetched_any = False
    for aid in missing:
        try:
            d = dict(client.get_activity(aid))
            cache[str(aid)] = {
                "description": d.get("description"),
                "private_note": d.get("private_note"),
            }
            fetched_any = True
        except Exception as e:
            if verbose:
                print(f"  stopped early on activity {aid}: {e}. "
                      f"Cached {len(cache)} so far — re-run later to fetch the rest.")
            break  # almost certainly rate-limited; further calls would fail too

    if fetched_any:
        _save_json(cache_path, cache)

    empty = {"description": None, "private_note": None}
    rows = [{"id": aid, **cache.get(str(aid), empty)} for aid in ids]
    return pd.DataFrame(rows)


def fetch_velocity_streams(client, ids, cache_path=STREAMS_FILE, verbose=True):
    """{id: np.ndarray | None} velocity_smooth streams, fetching only uncached ids.

    A successful fetch with no stream is cached as None (won't re-fetch); a *failed*
    fetch is left uncached so it retries next run. Stops early on error like above.
    """
    cache = _load_json(cache_path)
    missing = [aid for aid in ids if str(aid) not in cache]
    if missing and verbose:
        print(f"Fetching velocity streams for {len(missing)} new runs "
              f"({len(ids) - len(missing)} from cache)...")

    fetched_any = False
    for aid in missing:
        try:
            streams = client.get_activity_streams(aid, types=["velocity_smooth"])
            stream = streams.get("velocity_smooth") if streams else None
            cache[str(aid)] = [float(x) for x in stream.data] if stream and stream.data else None
            fetched_any = True
        except Exception as e:
            if verbose:
                print(f"  stopped early on activity {aid}: {e}. "
                      f"Cached {len([v for v in cache])} so far — re-run later for the rest.")
            break

    if fetched_any:
        _save_json(cache_path, cache)

    out = {}
    for aid in ids:
        data = cache.get(str(aid))  # None if uncached-this-run or genuinely streamless
        out[aid] = np.asarray(data, dtype=float) if data else None
    return out
