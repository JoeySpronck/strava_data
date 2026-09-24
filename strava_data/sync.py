"""Keep a local mirror of the Strava account, one activity fetched exactly once.

Strava's summary listing is cheap — a few requests for a whole history — but everything
interesting costs one request *per activity*: ``description``/``private_note`` come from
``get_activity``, the sample streams from ``get_activity_streams``. At ~1000 activities
that is far past the read limits (100 per 15 minutes, 1000 per day), so a run cannot
simply fetch what it needs.

Instead each run tops up a store that persists between runs (see :mod:`strava_data.store`):
the index is rebuilt every time, and anything per-activity is fetched only when it is
missing, newest first, within a fixed request budget. A first backfill therefore spreads
over a few runs; after that a run costs a handful of requests. Hitting the rate limit is
not an error — the run stops fetching, keeps what it got, and the next run carries on.

``export_web`` then writes the public subset for GitHub Pages. Private and
followers-only activities never leave the cache, and neither do descriptions or private
notes: the published JSON is world-readable.
"""
import datetime as _dt

from strava_data import streams as streams_mod
from strava_data.store import athlete_key

try:  # stravalib names this differently across versions; treat it as optional
    from stravalib.exc import RateLimitExceeded
except ImportError:  # pragma: no cover
    class RateLimitExceeded(Exception):
        pass

# Copied from each summary activity into the index. Everything the plots and the
# activity picker need, so neither has to open a stream file to build a list.
INDEX_FIELDS = (
    "id", "name", "type", "sport_type", "start_date", "start_date_local",
    "distance", "moving_time", "elapsed_time", "total_elevation_gain",
    "average_heartrate", "max_heartrate", "average_speed", "max_speed",
    "kilojoules", "elev_high", "elev_low", "has_heartrate", "manual",
    "private", "visibility",
)

# Sports whose description / private note the plots parse (carried hiking weight,
# strength volume). Fetching them for every activity would cost a request each for
# nothing, so the default stays narrow.
DETAIL_TYPES = ("Hike", "WeightTraining")

DEFAULT_MAX_REQUESTS = 80  # under Strava's 100-per-15-minutes read limit
DEFAULT_MAX_ERRORS = 5     # consecutive non-rate-limit failures before giving up


class Budget:
    """A countdown of API requests, so both fetch loops share one limit."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.rate_limited = False

    @property
    def left(self):
        return max(0, self.limit - self.used)

    def spend(self):
        self.used += 1

    def exhausted(self, reserve=0):
        """True when spending again would eat into `reserve` requests held back."""
        return self.rate_limited or self.left <= reserve


def _jsonable(value):
    # stravalib wraps enums such as `type` and `sport_type` in a pydantic RootModel;
    # str() on one of those gives "root='Run'", so unwrap before anything else.
    value = getattr(value, "root", value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return value.total_seconds()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


# Strava visibilities that may be exported to the world-readable Pages site. Note that
# "followers_only" is in here deliberately: on this account that is *every* activity, so
# excluding it would publish nothing at all. The consequence is real — the exported JSON
# carries full GPS traces, so routes starting at home are publicly fetchable. Dropping
# "followers_only" from this set is the one-line way to reverse that; excluding lat/lng
# instead is a matter of removing them in `export_web`.
PUBLISHABLE_VISIBILITIES = ("everyone", "followers_only")


def _is_public(row):
    """True when the activity may appear on a world-readable page.

    `private` is Strava's flag for "only you"; `visibility` is the newer, finer field.
    Either one saying no is enough to keep an activity in the cache and off the site.
    """
    if row.get("private"):
        return False
    visibility = row.get("visibility")
    return visibility is None or visibility in PUBLISHABLE_VISIBILITIES


def _wants_streams(row):
    """Manual entries have no samples, and gym sessions have no route worth storing."""
    return (
        _is_public(row)
        and not row.get("manual")
        and (row.get("distance") or 0) > 0
    )


def build_index(activities):
    """Summary activities -> the index rows written to ``activities.json``."""
    rows = []
    for activity in activities:
        data = dict(activity)
        row = {field: _jsonable(data.get(field)) for field in INDEX_FIELDS}
        row["public"] = _is_public(row)
        rows.append(row)
    # Newest first: the picker shows recent runs at the top, and the fetch loops walk
    # this same order so a fresh cache fills in with what you actually want to look at.
    rows.sort(key=lambda r: (r.get("start_date") or "", r.get("id") or 0), reverse=True)
    return rows


def _fetch_missing(ids, store, key_for, fetch_one, budget, label, verbose, reserve=0):
    """Fetch and store every id without a value yet, within `budget`. Returns a count.

    `reserve` holds back that many requests for whoever runs next, so the first loop
    cannot spend the whole run on itself and starve the second one indefinitely.
    """
    missing = [i for i in ids if not store.has(key_for(i))]
    if not missing:
        return 0
    if verbose:
        print(f"  {label}: {len(missing)} missing, "
              f"{max(0, budget.left - reserve)} request(s) available")

    fetched = errors = 0
    for activity_id in missing:
        if budget.exhausted(reserve):
            break
        budget.spend()
        try:
            store.write_json(key_for(activity_id), fetch_one(activity_id))
            fetched += 1
            errors = 0
        except RateLimitExceeded as exc:
            budget.rate_limited = True
            if verbose:
                print(f"  {label}: rate limited on {activity_id} ({exc}); stopping this run")
            break
        except Exception as exc:  # noqa: BLE001 - one bad activity must not end the backfill
            errors += 1
            if verbose:
                print(f"  {label}: failed on {activity_id}: {exc}")
            if errors >= DEFAULT_MAX_ERRORS:
                if verbose:
                    print(f"  {label}: {errors} failures in a row; stopping this run")
                break

    if verbose:
        print(f"  {label}: fetched {fetched}, {len(missing) - fetched} still missing")
    return fetched


def _prune(store, prefix, keep_ids, verbose):
    """Drop stored files for activities that are gone, or no longer publishable."""
    removed = 0
    for key in store.list(prefix):
        activity_id = key.rsplit("/", 1)[-1].removesuffix(".json")
        if activity_id not in keep_ids:
            store.delete(key)
            removed += 1
    if removed and verbose:
        print(f"  pruned {removed} stale file(s) under {prefix}")
    return removed


def sync(
    client,
    store,
    athlete_id=None,
    limit=1000,
    max_requests=DEFAULT_MAX_REQUESTS,
    detail_types=DETAIL_TYPES,
    verbose=True,
):
    """Top up `store` from Strava and return ``(athlete_id, index_rows)``.

    Rebuilds the activity index every run (it is cheap and activities get renamed), then
    spends what is left of `max_requests` on the per-activity data that is still missing.
    """
    budget = Budget(max_requests)

    # One request, every run: it resolves the athlete id when the caller did not pass one,
    # and it keeps a display name next to the data so the page can label an athlete
    # without the id being written into the HTML.
    athlete = client.get_athlete()
    budget.spend()
    athlete_id = athlete_id if athlete_id is not None else athlete.id
    store.write_json(athlete_key(athlete_id, "athlete.json"), {
        "id": athlete_id,
        "firstname": getattr(athlete, "firstname", None),
    })

    if verbose:
        print(f"Syncing athlete {athlete_id} (budget {max_requests} requests)...")

    # One request per page of 200; counted so a huge history cannot eat the whole budget.
    activities = list(client.get_activities(limit=limit))
    for _ in range(max(1, -(-len(activities) // 200))):
        budget.spend()

    rows = build_index(activities)
    store.write_json(athlete_key(athlete_id, "activities.json"), rows)
    if verbose:
        print(f"  index: {len(rows)} activities ({sum(r['public'] for r in rows)} public)")

    rows_by_id = {str(row["id"]): row for row in rows}

    detail_ids = [
        activity_id for activity_id, row in rows_by_id.items()
        if row.get("type") in detail_types or row.get("sport_type") in detail_types
    ]
    # Half the remaining budget is held back for streams. Without that, a first run on
    # a fresh cache would spend everything on descriptions — there are dozens of them —
    # and the decoupling page would stay empty for days.
    _fetch_missing(
        detail_ids,
        store,
        lambda i: athlete_key(athlete_id, "details", f"{i}.json"),
        lambda i: _fetch_details(client, i),
        budget,
        "details",
        verbose,
        reserve=budget.left // 2,
    )

    stream_ids = [
        activity_id for activity_id, row in rows_by_id.items() if _wants_streams(row)
    ]
    _fetch_missing(
        stream_ids,
        store,
        lambda i: athlete_key(athlete_id, "streams", f"{i}.json"),
        lambda i: _fetch_streams(client, i, rows_by_id[i]),
        budget,
        "streams",
        verbose,
    )

    _prune(store, athlete_key(athlete_id, "details") + "/", set(detail_ids), verbose)
    _prune(store, athlete_key(athlete_id, "streams") + "/", set(stream_ids), verbose)

    if verbose:
        print(f"  used {budget.used}/{max_requests} requests"
              + (" (rate limited)" if budget.rate_limited else ""))
    return athlete_id, rows


def _fetch_details(client, activity_id):
    detail = dict(client.get_activity(activity_id))
    return {
        "id": int(activity_id),
        "description": detail.get("description"),
        "private_note": detail.get("private_note"),
    }


def _fetch_streams(client, activity_id, row):
    """One activity's streams, or an ``empty`` marker so we never ask Strava twice."""
    compact = streams_mod.fetch(client, activity_id)
    if compact is None:
        return {"schema": streams_mod.SCHEMA, "id": int(activity_id), "empty": True}
    return streams_mod.build_document(row, compact)


def load_details(store, athlete_id, ids):
    """``{id: {"description": ..., "private_note": ...}}`` for the ids already cached."""
    out = {}
    for activity_id in ids:
        doc = store.read_json(athlete_key(athlete_id, "details", f"{activity_id}.json"))
        out[int(activity_id)] = doc or {"description": None, "private_note": None}
    return out


def export_web(store, athlete_id, out_dir, sample_seconds=None, verbose=True):
    """Write the public subset of the store under `out_dir`, for GitHub Pages.

    What ships: an index of the public activities that actually have streams, and those
    stream files. What does not: private and followers-only activities, and descriptions
    and private notes of any activity.
    """
    import json
    import os

    rows = store.read_json(athlete_key(athlete_id, "activities.json")) or []
    out_athlete = os.path.join(out_dir, "athletes", str(athlete_id))
    os.makedirs(os.path.join(out_athlete, "streams"), exist_ok=True)

    published = []
    for row in rows:
        if not _wants_streams(row):
            continue
        doc = store.read_json(athlete_key(athlete_id, "streams", f"{row['id']}.json"))
        if not doc or doc.get("empty") or not doc.get("streams"):
            continue
        if sample_seconds:
            doc = dict(doc)
            doc["streams"] = streams_mod.downsample(doc["streams"], sample_seconds)
            doc["sample_seconds"] = sample_seconds
            doc["n"] = len(doc["streams"]["t"])
        path = os.path.join(out_athlete, "streams", f"{row['id']}.json")
        with open(path, "w") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        # The index the page reads is deliberately thinner than the cached one: no
        # visibility flags, no fields the picker does not show.
        published.append({
            "id": row["id"],
            "name": row.get("name"),
            "type": row.get("type"),
            "sport_type": row.get("sport_type"),
            "start_date": row.get("start_date"),
            "start_date_local": row.get("start_date_local"),
            "distance": row.get("distance"),
            "moving_time": row.get("moving_time"),
            "total_elevation_gain": row.get("total_elevation_gain"),
            "average_heartrate": row.get("average_heartrate"),
            "n": doc.get("n"),
        })

    index = {
        "schema": streams_mod.SCHEMA,
        "athlete_id": athlete_id,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "activities": published,
    }
    with open(os.path.join(out_athlete, "activities.json"), "w") as fh:
        json.dump(index, fh, separators=(",", ":"))

    # The one file the page fetches without being told anything: which athletes exist.
    # Upserted rather than overwritten, so adding a second athlete later is just another
    # export_web call.
    roster_path = os.path.join(out_dir, "athletes.json")
    roster = {"schema": streams_mod.SCHEMA, "athletes": []}
    if os.path.exists(roster_path):
        with open(roster_path) as fh:
            roster = json.load(fh)
    profile = store.read_json(athlete_key(athlete_id, "athlete.json")) or {}
    entry = {
        "id": athlete_id,
        "name": profile.get("firstname") or str(athlete_id),
        "activities": len(published),
    }
    roster["athletes"] = [a for a in roster["athletes"] if a["id"] != athlete_id] + [entry]
    roster["athletes"].sort(key=lambda a: a["id"])
    roster["generated"] = index["generated"]
    with open(roster_path, "w") as fh:
        json.dump(roster, fh, separators=(",", ":"))

    # Stale stream files from a previous run whose activity is no longer published.
    keep = {f"{row['id']}.json" for row in published}
    for name in os.listdir(os.path.join(out_athlete, "streams")):
        if name not in keep:
            os.remove(os.path.join(out_athlete, "streams", name))

    if verbose:
        print(f"  exported {len(published)} activities to {out_athlete}")
    return published
