"""ByteTrack multi-object tracker.

ByteTrack's contribution is its use of *low*-confidence detections: rather
than discarding them, it matches them against tracks that the high-confidence
pass left unclaimed. That is what keeps an ID stable through a partial
occlusion instead of ending the track and issuing a fresh number when the
person reappears — which matters here because every event we raise is keyed to
a person ID.

Association order per frame:

1. Kalman-predict every track.
2. Match confirmed tracks ↔ high-confidence detections on IoU.
3. Match the leftover tracks ↔ low-confidence detections (the "BYTE" step).
4. Match still-unmatched tracks ↔ unconfirmed tracks' detections.
5. Promote surviving unmatched high-confidence detections to new tracks.
6. Retire tracks unseen for `max_age` frames.

Assignment uses an exact Hungarian solver implemented here so the project does
not take a scipy dependency for one function; scipy's compiled version is used
automatically when it happens to be installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from backend.config import settings
from backend.inference.base import Detection
from backend.inference.mojo.bridge import get_bridge
from backend.tracking.kalman import KalmanBoxTracker, xyah_to_xyxy, xyxy_to_xyah

# scipy is optional; the local solver below is exact and dependency-free.
try:  # pragma: no cover - depends on the host environment
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover
    _scipy_lsa = None


#: Per-frame velocity decay applied while a track is LOST. Tuned so a track
#: coasts briefly (carrying it through a short occlusion) then holds position.
LOST_VELOCITY_DAMPING: float = 0.7

#: Cost ceiling when re-associating a LOST track (IoU >= 0.1). Looser than
#: the live gate because the predicted box has necessarily drifted.
REASSOCIATION_COST_GATE: float = 0.9


class TrackState(Enum):
    """Lifecycle of a track."""

    TENTATIVE = auto()  # seen, not yet confirmed by `min_hits`
    TRACKED = auto()    # confirmed and matched this frame
    LOST = auto()       # confirmed but unmatched recently
    REMOVED = auto()    # retired


@dataclass
class Track:
    """One tracked person with a persistent ID."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    label: str = "person"

    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    frame_first_seen: int = 0
    frame_last_seen: int = 0

    mean: np.ndarray = field(default_factory=lambda: np.zeros(8, np.float32))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8, dtype=np.float32))

    #: Recent (timestamp, centre_x, centre_y, height) samples. Behaviour
    #: analysis (speed, fall, loitering) reads this rather than keeping its own
    #: history, so there is exactly one motion record per person.
    history: list[tuple[float, float, float, float]] = field(default_factory=list)
    max_history: int = 90

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.TRACKED, TrackState.LOST)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre — used for zone membership."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def height(self) -> float:
        return max(1e-3, self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        return max(1e-3, self.bbox[2] - self.bbox[0])

    @property
    def aspect_ratio(self) -> float:
        """Width / height. Above ~1.3 the box reads as horizontal."""
        return self.width / self.height

    def record(self, timestamp: float) -> None:
        """Append a motion sample, trimming the oldest."""
        cx, cy = self.centroid
        self.history.append((timestamp, cx, cy, self.height))
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]


# ══════════════════════════════════════════════════════════════════════════
#  Assignment
# ══════════════════════════════════════════════════════════════════════════
def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact minimum-cost assignment (O(n³) Jonker-Volgenant shortest paths).

    Returns matched ``(row_indices, col_indices)``. Handles rectangular input.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros(0, int), np.zeros(0, int)

    transposed = n_rows > n_cols
    if transposed:
        cost = cost.T
        n_rows, n_cols = cost.shape

    INF = np.inf
    # col_of_row[r] = assigned column; row_of_col[c] = assigned row (-1 = free)
    row_of_col = np.full(n_cols + 1, -1, dtype=int)
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        row_of_col[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, INF)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = row_of_col[j0]
            delta = INF
            j1 = -1
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == -1:
                break
            for j in range(n_cols + 1):
                if used[j]:
                    u[row_of_col[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if row_of_col[j0] == -1:
                break
        while j0:
            j1 = way[j0]
            row_of_col[j0] = row_of_col[j1]
            j0 = j1

    rows, cols = [], []
    for j in range(1, n_cols + 1):
        r = row_of_col[j]
        if r > 0:
            rows.append(r - 1)
            cols.append(j - 1)
    order = np.argsort(rows)
    rows_arr = np.asarray(rows, int)[order]
    cols_arr = np.asarray(cols, int)[order]
    if transposed:
        return cols_arr, rows_arr
    return rows_arr, cols_arr


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match rows to columns, rejecting pairs costlier than `max_cost`.

    Returns ``(matches, unmatched_rows, unmatched_cols)``.
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    solver = _scipy_lsa or _hungarian
    row_idx, col_idx = solver(cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_idx, col_idx, strict=False):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    return (
        matches,
        [r for r in range(n_rows) if r not in matched_rows],
        [c for c in range(n_cols) if c not in matched_cols],
    )


def _iou_cost(tracks: list[Track], detections: list[Detection]) -> np.ndarray:
    """1 − IoU cost matrix (Mojo-accelerated IoU where it wins)."""
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), np.float32)
    a = np.array([t.bbox for t in tracks], dtype=np.float32)
    b = np.array([d.bbox for d in detections], dtype=np.float32)
    return 1.0 - get_bridge().iou_matrix(a, b)


# ══════════════════════════════════════════════════════════════════════════
#  Tracker
# ══════════════════════════════════════════════════════════════════════════
class ByteTracker:
    """Multi-object tracker producing stable, persistent person IDs."""

    def __init__(
        self,
        high_threshold: float | None = None,
        low_threshold: float | None = None,
        match_iou: float | None = None,
        max_age: int | None = None,
        min_hits: int | None = None,
        target_labels: Iterable[str] = ("person",),
    ) -> None:
        self.high_threshold = (
            settings.track_high_threshold if high_threshold is None else high_threshold
        )
        self.low_threshold = (
            settings.track_low_threshold if low_threshold is None else low_threshold
        )
        self.match_iou = settings.track_match_iou if match_iou is None else match_iou
        self.max_age = settings.track_max_age_frames if max_age is None else max_age
        self.min_hits = settings.track_min_hits if min_hits is None else min_hits
        self.target_labels = set(target_labels)

        self.tracks: list[Track] = []
        self.frame_index = 0
        self._next_id = 1
        self._kf = KalmanBoxTracker()

    # ── helpers ──────────────────────────────────────────────────────────
    def _new_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _create(self, detection: Detection) -> Track:
        track = Track(
            track_id=self._new_id(),
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
            frame_first_seen=self.frame_index,
            frame_last_seen=self.frame_index,
        )
        track.mean, track.covariance = self._kf.initiate(xyxy_to_xyah(detection.bbox))
        # min_hits == 1 means "confirm immediately".
        if self.min_hits <= 1:
            track.state = TrackState.TRACKED
        return track

    def _apply(self, track: Track, detection: Detection) -> None:
        track.mean, track.covariance = self._kf.update(
            track.mean, track.covariance, xyxy_to_xyah(detection.bbox)
        )
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.label = detection.label
        track.class_id = detection.class_id
        track.hits += 1
        track.time_since_update = 0
        track.frame_last_seen = self.frame_index
        if track.state is TrackState.TENTATIVE and track.hits >= self.min_hits or track.state is TrackState.LOST:
            track.state = TrackState.TRACKED

    # ── main step ────────────────────────────────────────────────────────
    def update(
        self, detections: list[Detection], timestamp: float | None = None
    ) -> list[Track]:
        """Advance one frame. Returns the currently confirmed tracks."""
        import time as _time

        self.frame_index += 1
        now = _time.monotonic() if timestamp is None else timestamp

        relevant = [
            d
            for d in detections
            if (not self.target_labels or d.label in self.target_labels)
        ]
        high = [d for d in relevant if d.confidence >= self.high_threshold]
        low = [
            d
            for d in relevant
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        # 1. Predict. A lost track's velocity is damped each frame: a person
        #    behind an obstacle does not keep accelerating, and an
        #    unconstrained constant-velocity extrapolation overshoots badly
        #    over a long occlusion, which is what breaks re-association and
        #    burns a fresh person ID on reappearance.
        for track in self.tracks:
            if track.state is TrackState.LOST:
                track.mean[4:] *= LOST_VELOCITY_DAMPING
            track.mean, track.covariance = self._kf.predict(track.mean, track.covariance)
            track.bbox = xyah_to_xyxy(track.mean)
            track.age += 1
            track.time_since_update += 1

        confirmed = [t for t in self.tracks if t.is_confirmed]
        tentative = [t for t in self.tracks if t.state is TrackState.TENTATIVE]

        # 2. Confirmed tracks ↔ high-confidence detections.
        #    `match_iou` is a *cost* ceiling in ByteTrack's convention: a gate
        #    of 0.8 admits pairs with IoU >= 0.2, which is deliberately loose
        #    because the Kalman prediction has already narrowed the field.
        matches, unmatched_tracks, unmatched_high = linear_assignment(
            _iou_cost(confirmed, high), self.match_iou
        )
        for ti, di in matches:
            self._apply(confirmed[ti], high[di])

        remaining = [confirmed[i] for i in unmatched_tracks]

        # 3. The BYTE step: leftovers ↔ low-confidence detections. A looser
        #    IoU gate is appropriate — these boxes are noisier by definition.
        if remaining and low:
            low_matches, unmatched_remaining, _ = linear_assignment(
                _iou_cost(remaining, low), 0.5
            )
            for ti, di in low_matches:
                self._apply(remaining[ti], low[di])
            remaining = [remaining[i] for i in unmatched_remaining]

        # 3b. Re-association pass for LOST tracks against the high-confidence
        #     detections nobody claimed. Without this a person who reappears
        #     after an occlusion gets a brand-new ID, and every event keyed to
        #     the old ID becomes untraceable.
        leftover_high = [high[i] for i in unmatched_high]
        lost = [t for t in remaining if t.state is TrackState.LOST]
        if lost and leftover_high:
            lost_matches, unmatched_lost, unmatched_leftover = linear_assignment(
                _iou_cost(lost, leftover_high), REASSOCIATION_COST_GATE
            )
            for ti, di in lost_matches:
                self._apply(lost[ti], leftover_high[di])
            matched_lost = {id(lost[ti]) for ti, _ in lost_matches}
            remaining = [t for t in remaining if id(t) not in matched_lost]
            leftover_high = [leftover_high[i] for i in unmatched_leftover]

        # 4. Tentative tracks ↔ still-unmatched high-confidence detections.
        if tentative and leftover_high:
            t_matches, unmatched_tentative, unmatched_leftover = linear_assignment(
                _iou_cost(tentative, leftover_high), 0.3
            )
            for ti, di in t_matches:
                self._apply(tentative[ti], leftover_high[di])
            stale_tentative = [tentative[i] for i in unmatched_tentative]
            leftover_high = [leftover_high[i] for i in unmatched_leftover]
        else:
            stale_tentative = tentative

        # 5. Mark unmatched confirmed tracks lost.
        for track in remaining:
            if track.state is TrackState.TRACKED:
                track.state = TrackState.LOST

        # 6. New tracks from the remaining high-confidence detections.
        for detection in leftover_high:
            self.tracks.append(self._create(detection))

        # 7. Retire. A tentative track that misses a single frame is dropped
        #    immediately — it was probably a false positive.
        keep: list[Track] = []
        for track in self.tracks:
            if track in stale_tentative and track.time_since_update > 0:
                continue
            if track.time_since_update > self.max_age:
                track.state = TrackState.REMOVED
                continue
            keep.append(track)
        self.tracks = keep

        # 8. Record motion for confirmed, currently-visible tracks.
        active = [
            t for t in self.tracks if t.state is TrackState.TRACKED
        ]
        for track in active:
            track.record(now)
        return active

    # ── introspection ────────────────────────────────────────────────────
    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state is TrackState.TRACKED]

    def get(self, track_id: int) -> Track | None:
        return next((t for t in self.tracks if t.track_id == track_id), None)

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_index = 0
        self._next_id = 1
