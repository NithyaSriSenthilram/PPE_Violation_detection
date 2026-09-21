"""Zone geometry.

Zones are stored with **normalised** polygon coordinates (0..1) so one
definition survives a resolution change and the browser can draw it without
knowing the stream size. :class:`ResolvedZone` caches the pixel-space polygon
for a specific frame size and rebuilds only when that size changes.

Membership tests use the tracked person's **foot point** (bottom-centre of the
box), not the centroid: a floor-marked exclusion zone is about where someone is
standing, and a centroid test fires early for anyone leaning over the line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.inference.mojo.bridge import get_bridge


@dataclass
class ResolvedZone:
    """A zone prepared for hit-testing against a given frame size."""

    zone_id: str
    name: str
    zone_type: str
    #: Normalised polygon, [[x, y], ...] with x, y in 0..1.
    polygon_norm: list[list[float]]
    enabled: bool = True
    loitering_threshold: float | None = None
    crowd_threshold: int | None = None
    colour: str = "#f43f5e"

    _pixel_polygon: np.ndarray | None = field(default=None, repr=False)
    _cached_size: tuple[int, int] | None = field(default=None, repr=False)

    def pixel_polygon(self, width: int, height: int) -> np.ndarray:
        """Polygon in pixel coordinates for a `width` × `height` frame."""
        if self._cached_size != (width, height) or self._pixel_polygon is None:
            poly = np.asarray(self.polygon_norm, dtype=np.float32)
            scaled = np.empty_like(poly)
            scaled[:, 0] = poly[:, 0] * width
            scaled[:, 1] = poly[:, 1] * height
            self._pixel_polygon = scaled
            self._cached_size = (width, height)
        return self._pixel_polygon

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        """Is the pixel point (x, y) inside this zone?"""
        return bool(
            contains_points(
                np.array([[x, y]], dtype=np.float32),
                self.pixel_polygon(width, height),
            )[0]
        )

    @property
    def is_valid(self) -> bool:
        return len(self.polygon_norm) >= 3

    def area_fraction(self) -> float:
        """Zone area as a fraction of the frame (shoelace formula).

        Used by crowd analysis to turn a head count into a density.
        """
        poly = np.asarray(self.polygon_norm, dtype=np.float64)
        if len(poly) < 3:
            return 0.0
        x, y = poly[:, 0], poly[:, 1]
        return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def contains_points(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Batched point-in-polygon. Mojo-accelerated when that path is faster."""
    return get_bridge().points_in_polygon(points, polygon)


def zones_for_points(
    points: np.ndarray, zones: list[ResolvedZone], width: int, height: int
) -> list[list[str]]:
    """For each point, the ids of every enabled zone containing it.

    One call per zone over all points, rather than a Python double loop — this
    is the hot path when many people are tracked across several zones.
    """
    result: list[list[str]] = [[] for _ in range(len(points))]
    if len(points) == 0:
        return result
    for zone in zones:
        if not zone.enabled or not zone.is_valid:
            continue
        mask = contains_points(points, zone.pixel_polygon(width, height))
        for index in np.flatnonzero(mask):
            result[int(index)].append(zone.zone_id)
    return result


def polygon_centroid(polygon: list[list[float]]) -> tuple[float, float]:
    """Centroid of a normalised polygon — used to place zone labels."""
    poly = np.asarray(polygon, dtype=np.float64)
    if len(poly) == 0:
        return (0.0, 0.0)
    return (float(poly[:, 0].mean()), float(poly[:, 1].mean()))


def validate_polygon(polygon: list[list[float]]) -> tuple[bool, str]:
    """Check a polygon is usable. Returns ``(ok, reason)``."""
    if len(polygon) < 3:
        return False, "a zone needs at least 3 points"
    if len(polygon) > 64:
        return False, "a zone may have at most 64 points"
    for point in polygon:
        if len(point) != 2:
            return False, "each point must be [x, y]"
        x, y = point
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False, "points must be normalised to 0..1"
    poly = np.asarray(polygon, dtype=np.float64)
    x, y = poly[:, 0], poly[:, 1]
    area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0
    if area < 1e-6:
        return False, "polygon is degenerate (zero area)"
    return True, ""
