"""Key/value store for per-athlete Strava data.

Storage is kept separate from syncing on purpose. Today everything lives in a folder
(``.cache/``, restored from the GitHub Actions cache between runs); later, when the site
grows a login and more than one athlete, the same sync code can write to an object store
such as Cloudflare R2 by swapping :class:`LocalStore` for an ``R2Store``.

Keys are POSIX-style relative paths and always start with the athlete, even with a single
user, so nothing has to be renamed when a second one shows up::

    athletes/<athlete_id>/activities.json    index of every activity
    athletes/<athlete_id>/details/<id>.json  description / private note
    athletes/<athlete_id>/streams/<id>.json  the sample streams

Use :func:`athlete_key` to build them rather than formatting paths by hand.
"""
import json
import os
import posixpath
import shutil
from abc import ABC, abstractmethod


def athlete_key(athlete_id, *parts):
    """Key under one athlete, e.g. ``athlete_key(123, "streams", "987.json")``."""
    return posixpath.join("athletes", str(athlete_id), *(str(p) for p in parts))


def _check_key(key):
    """Reject anything that could escape the store root or is not a relative path."""
    if not key or key.endswith("/"):
        raise ValueError(f"store key must name a value, got {key!r}")
    return _check_prefix(key)


def _check_prefix(prefix):
    """Same checks, but a prefix may be empty or end in '/' — it names a subtree."""
    if prefix.startswith("/") or "\\" in prefix:
        raise ValueError(f"store key must be a relative POSIX path, got {prefix!r}")
    parts = prefix.split("/")
    if any(part in (".", "..") for part in parts):
        raise ValueError(f"store key must not contain '..' segments: {prefix!r}")
    if any(part == "" for part in parts[:-1]):
        raise ValueError(f"store key must not contain empty segments: {prefix!r}")
    return prefix


class Store(ABC):
    """Bytes in, bytes out, addressed by key.

    Backends only implement the five primitives; the JSON helpers below are shared.
    """

    @abstractmethod
    def has(self, key):
        """True if `key` holds a value."""

    @abstractmethod
    def read(self, key):
        """The bytes at `key`, or None if there is nothing there."""

    @abstractmethod
    def write(self, key, data):
        """Store `data` (bytes) at `key`, replacing anything already there."""

    @abstractmethod
    def delete(self, key):
        """Remove `key`. Deleting a missing key is not an error."""

    @abstractmethod
    def list(self, prefix=""):
        """Every key starting with `prefix`, sorted."""

    # -- JSON convenience -------------------------------------------------
    def read_json(self, key, default=None):
        raw = self.read(key)
        if raw is None:
            return default
        return json.loads(raw.decode("utf-8"))

    def write_json(self, key, obj, indent=None):
        # separators drop the space after ':' and ',' — on stream files that is a few
        # percent of the payload, and these are served over the wire to the browser.
        text = json.dumps(obj, indent=indent, separators=(",", ":") if indent is None else None)
        self.write(key, text.encode("utf-8"))


class LocalStore(Store):
    """A folder on disk. `root` is created on first write."""

    def __init__(self, root):
        self.root = root

    def _path(self, key):
        return os.path.join(self.root, *_check_key(key).split("/"))

    def has(self, key):
        return os.path.isfile(self._path(key))

    def read(self, key):
        path = self._path(key)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    def write(self, key, data):
        path = self._path(key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)  # atomic: a killed run never leaves half a file behind

    def delete(self, key):
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    def list(self, prefix=""):
        _check_prefix(prefix)
        keys = []
        for dirpath, _, filenames in os.walk(self.root):
            rel = os.path.relpath(dirpath, self.root)
            rel = "" if rel == "." else rel.replace(os.sep, "/")
            for name in filenames:
                if name.endswith(".tmp"):
                    continue
                key = f"{rel}/{name}" if rel else name
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def copy_tree(self, prefix, dest_root):
        """Copy every key under `prefix` into `dest_root`, keeping the key as the path.

        Used by the publish step to lift ``athletes/...`` out of the cache and into the
        web folder; an R2-backed store would instead serve those keys straight from a
        Worker, which is why this lives on the local backend only.
        """
        count = 0
        for key in self.list(prefix):
            dest = os.path.join(dest_root, *key.split("/"))
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            shutil.copyfile(self._path(key), dest)
            count += 1
        return count
