"""Rate-limit handling of the per-activity caches: wait for the next window, or stop early.

A fake client raises Strava-style HTTP 429s; ``time.sleep`` is stubbed so nothing waits.

Run it with::

    python tests/test_activity_cache.py
"""
import os
import sys
import tempfile

import requests
from stravalib import exc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strava_data import activity_cache as ac  # noqa: E402


def http_429(daily_used=500):
    response = requests.Response()
    response.status_code = 429
    response.headers["X-ReadRateLimit-Limit"] = "100,1000"
    response.headers["X-ReadRateLimit-Usage"] = f"101,{daily_used}"
    return exc.Fault("429 Client Error: Too Many Requests", response=response)


class FakeClient:
    """Allows ``per_window`` get_activity calls, then 429s until ``window_ends`` is called."""

    def __init__(self, per_window, daily_used=500):
        self.per_window, self.daily_used = per_window, daily_used
        self.used, self.calls = 0, []

    def window_ends(self):
        self.used = 0

    def get_activity(self, aid):
        if self.used >= self.per_window:
            raise http_429(self.daily_used)
        self.used += 1
        self.calls.append(aid)
        return {"description": f"desc {aid}", "private_note": None}


def run(client, ids, max_waits, refresh_ids=()):
    sleeps = []
    ac.MAX_RATE_LIMIT_WAITS, ac._waits_used = max_waits, 0
    real_sleep = ac.time.sleep
    ac.time.sleep = lambda s: (sleeps.append(s), client.window_ends())
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "details.json")
            df = ac.fetch_text_fields(client, ids, cache_path=path, verbose=False,
                                      refresh_ids=refresh_ids)
            return df, sleeps, ac._load_json(path)
    finally:
        ac.time.sleep = real_sleep


def test_waits_for_next_window_and_finishes():
    client = FakeClient(per_window=3)
    df, sleeps, cache = run(client, list(range(8)), max_waits=4)
    assert client.calls == list(range(8))           # nothing skipped, nothing fetched twice
    assert len(sleeps) == 2 and all(0 < s <= 915 for s in sleeps)
    assert df["description"].tolist() == [f"desc {i}" for i in range(8)]
    assert len(cache) == 8


def test_stops_when_wait_budget_is_used():
    client = FakeClient(per_window=3)
    df, sleeps, cache = run(client, list(range(8)), max_waits=1)
    assert len(sleeps) == 1 and len(cache) == 6      # 3 + 3, then stop instead of waiting again
    assert df["description"].isna().sum() == 2


def test_default_never_waits():
    client = FakeClient(per_window=3)
    _, sleeps, cache = run(client, list(range(8)), max_waits=0)
    assert sleeps == [] and len(cache) == 3


def test_daily_limit_is_not_waited_out():
    client = FakeClient(per_window=3, daily_used=1000)
    _, sleeps, cache = run(client, list(range(8)), max_waits=4)
    assert sleeps == [] and len(cache) == 3


def test_other_errors_stop_immediately():
    class Broken(FakeClient):
        def get_activity(self, aid):
            raise RuntimeError("network down")
    _, sleeps, cache = run(Broken(per_window=3), [1, 2], max_waits=4)
    assert sleeps == [] and cache == {}


def test_next_window():
    assert ac._seconds_to_next_window(now=900 * 10) == 915          # exactly on a boundary
    assert ac._seconds_to_next_window(now=900 * 10 + 890) == 25


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
