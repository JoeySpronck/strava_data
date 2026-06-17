"""Per-sport marker shapes for the month calendar.

Each sport gets a distinct outline (built with ``gemmini`` + ``shapely``) instead
of a plain circle. The vertices are normalized to **unit equal-area radius**: every
shape encloses an area of ``pi``, exactly like a unit circle. Scaling a shape by a
radius ``r`` therefore yields area ``pi * r**2`` — identical for every sport — so two
activities with the same magnitude render with the same visible area regardless of
which shape they use.

``sport_marker_vertices()`` returns one ``(N, 2)`` float array per sport, centered on
the centroid and ready to be scaled by ``r`` and offset to a cell center.
"""

from functools import lru_cache

import numpy as np
from gemmini import ConcaveStar, Circle, Gear
from shapely.geometry import Polygon


def _to_polygon(obj):
    """Convert a gemmini shape to a valid shapely Polygon (buffer(0) fixes self-intersections)."""
    poly = Polygon(np.asarray(obj.coords()))
    return poly if poly.is_valid else poly.buffer(0)


def _build_polygons():
    """Build the raw per-sport shapely polygons (matches make_shapes.ipynb)."""
    shapes = {
        "run": _to_polygon(Circle(r=10, n=80)),
        "trail": _to_polygon(ConcaveStar(s=10, v=10, n=2)),
        "strength": _to_polygon(Gear(r=10, c=6, n=2)),
        "bike": _to_polygon(Gear(r=10, c=16, n=2)),
        "hike": _to_polygon(ConcaveStar(s=10, v=6, n=2)),
    }
    # Rounding passes (buffer out/in) to soften the gear teeth and star points.
    shapes['strength'] = shapes['strength'].buffer(3).buffer(-3)
    shapes['bike'] = shapes['bike'].buffer(-0.4).buffer(0.7)
    shapes['hike'] = shapes['hike'].buffer(-1).buffer(6).buffer(-3)
    return shapes


@lru_cache(maxsize=1)
def sport_marker_vertices():
    """Return ``{sport: (N, 2) array}`` normalized to unit equal-area radius.

    Each array is centered on its centroid and scaled so the enclosed area equals ``pi``.
    Multiply by a target radius ``r`` and add the cell center to place a marker; the
    resulting area is ``pi * r**2`` for every sport.
    """
    out = {}
    for sport, poly in _build_polygons().items():
        xy = np.asarray(poly.exterior.coords)
        xy = xy - np.asarray(poly.centroid.coords)[0]  # center on centroid
        r_eq = np.sqrt(poly.area / np.pi)               # equal-area radius
        out[sport] = xy / r_eq
    return out
