"""Evidence capture: snapshots and short video clips.

Naming carries the provenance an operator needs to find a file without the
database: ``{camera_id}_{event_type}_{timestamp}_{event_id}``.

Only frames around an event are ever written — the pipeline never persists a
continuous stream. Clips are assembled from the rolling pre-event buffer plus
a short post-roll, and the encode runs on a worker thread so a slow disk
cannot stall video capture.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.logging_conf import get_logger
from backend.video.ring_buffer import BufferedFrame, FrameRingBuffer

logger = get_logger(__name__)

#: Everything outside this set is stripped from path components.
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]")


def sanitise(value: str, max_length: int = 48) -> str:
    """Make a string safe to embed in a filename."""
    cleaned = _SAFE_COMPONENT.sub("-", value or "unknown")
    return cleaned[:max_length].strip("-.") or "unknown"


def evidence_stem(camera_id: str, event_type: str, event_id: str, when: datetime) -> str:
    """Build the shared filename stem for an event's evidence."""
    stamp = when.astimezone(UTC).strftime("%Y%m%dT%H%M%S")
    return (
        f"{sanitise(camera_id, 36)}_{sanitise(event_type, 24)}_"
        f"{stamp}_{sanitise(event_id, 36)}"
    )


def resolve_evidence_path(relative: str) -> Path | None:
    """Resolve a stored evidence path, refusing anything outside EVIDENCE_DIR.

    Evidence paths are persisted relative to the **project root** (see
    :meth:`EvidenceWriter._relative`), so they must be resolved against the
    same base. Resolving them against ``evidence_root.parent`` instead happens
    to work only while ``EVIDENCE_DIR`` sits directly under the project root —
    with any other configured location every lookup fails.

    The containment check afterwards is the security boundary: it stops a
    crafted or corrupted database value from reading a file outside the
    evidence directory, even though clients never supply these paths directly.
    """
    if not relative:
        return None

    from backend.config import PROJECT_ROOT

    root = settings.evidence_root.resolve()
    candidate = Path(relative)
    absolute = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    try:
        resolved = absolute.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root):
        logger.warning("Rejected evidence path outside EVIDENCE_DIR: %s", relative)
        return None
    return resolved if resolved.is_file() else None


@dataclass(slots=True)
class EvidenceResult:
    """Paths produced for one event, relative to the project root."""

    snapshot_path: str | None = None
    clip_path: str | None = None


class EvidenceWriter:
    """Writes snapshots synchronously and clips on a background thread."""

    def __init__(self, max_workers: int = 2) -> None:
        settings.ensure_directories()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="evidence"
        )
        self._pending: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    # ── snapshots ────────────────────────────────────────────────────────
    def write_snapshot(
        self,
        frame: np.ndarray,
        camera_id: str,
        event_type: str,
        event_id: str,
        when: datetime | None = None,
    ) -> str | None:
        """Save a single JPEG. Returns a project-relative path."""
        when = when or datetime.now(UTC)
        stem = evidence_stem(camera_id, event_type, event_id, when)
        path = settings.snapshot_dir / f"{stem}.jpg"
        try:
            ok = cv2.imwrite(
                str(path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), settings.snapshot_jpeg_quality],
            )
        except Exception as exc:
            logger.error("Snapshot write failed for %s: %s", event_id, exc)
            return None
        if not ok:
            logger.error("Snapshot encode failed for %s", event_id)
            return None
        return self._relative(path)

    # ── clips ────────────────────────────────────────────────────────────
    def schedule_clip(
        self,
        buffer: FrameRingBuffer,
        camera_id: str,
        event_type: str,
        event_id: str,
        event_time: float,
        fps: float,
        when: datetime | None = None,
        post_seconds: float | None = None,
    ) -> str | None:
        """Queue clip assembly. Returns the path the clip *will* occupy.

        The path is recorded on the event immediately; the file appears once
        the post-roll has elapsed and the encode finishes. Callers should treat
        a missing file as "still being written" — the API does exactly that.
        """
        if not settings.clip_enabled:
            return None
        when = when or datetime.now(UTC)
        stem = evidence_stem(camera_id, event_type, event_id, when)
        path = settings.clip_dir / f"{stem}.mp4"
        post = settings.clip_post_seconds if post_seconds is None else post_seconds

        done = threading.Event()
        with self._lock:
            self._pending[event_id] = done

        self._pool.submit(
            self._write_clip, buffer, path, event_id, event_time, post, fps, done
        )
        return self._relative(path)

    def _write_clip(
        self,
        buffer: FrameRingBuffer,
        path: Path,
        event_id: str,
        event_time: float,
        post_seconds: float,
        fps: float,
        done: threading.Event,
    ) -> None:
        try:
            # Wait for the post-roll to accumulate in the ring buffer. Polling
            # keeps this independent of the capture thread's cadence.
            deadline = time.monotonic() + post_seconds + 1.0
            target = event_time + post_seconds
            while time.monotonic() < deadline:
                latest = buffer.latest()
                if latest and latest.timestamp >= target:
                    break
                time.sleep(0.05)

            start = event_time - settings.clip_buffer_seconds
            frames = [f for f in buffer.snapshot() if start <= f.timestamp <= target]
            if not frames:
                logger.warning(
                    "No buffered frames for clip %s — clip not written", event_id
                )
                return

            self._encode(frames, path, fps)
        except Exception as exc:
            logger.error("Clip write failed for %s: %s", event_id, exc)
        finally:
            done.set()
            with self._lock:
                self._pending.pop(event_id, None)

    @staticmethod
    def _encode(frames: list[BufferedFrame], path: Path, fps: float) -> None:
        """Encode buffered JPEGs into an MP4."""
        first = frames[0].decode()
        if first is None:
            logger.warning("Clip %s: first frame failed to decode", path.name)
            return
        height, width = first.shape[:2]

        # avc1 gives browser-playable H.264 where the OpenCV build supports it;
        # mp4v is the dependable fallback and still plays in most players.
        writer = None
        for codec in ("avc1", "mp4v"):
            candidate = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                max(1.0, fps),
                (width, height),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()

        if writer is None:
            logger.error("No usable video codec for %s", path.name)
            return

        written = 0
        try:
            for buffered in frames:
                image = buffered.decode()
                if image is None:
                    continue
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))
                writer.write(image)
                written += 1
        finally:
            writer.release()

        logger.info(
            "Evidence clip written: %s (%d frames, %.1fs)",
            path.name, written, written / max(1.0, fps),
        )

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _relative(path: Path) -> str:
        """Store paths relative to the project root, keeping the DB portable.

        :func:`resolve_evidence_path` resolves against the same base — the two
        must agree or evidence lookup silently fails for any non-default
        ``EVIDENCE_DIR``.
        """
        from backend.config import PROJECT_ROOT

        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    def wait_for_clips(self, timeout: float = 15.0) -> bool:
        """Block until queued clips finish. Used by tests and shutdown."""
        with self._lock:
            events = list(self._pending.values())
        deadline = time.monotonic() + timeout
        for event in events:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not event.wait(remaining):
                return False
        return True

    def shutdown(self) -> None:
        self.wait_for_clips(timeout=10.0)
        self._pool.shutdown(wait=True)


_writer: EvidenceWriter | None = None


def get_evidence_writer() -> EvidenceWriter:
    """Process-wide evidence writer."""
    global _writer
    if _writer is None:
        _writer = EvidenceWriter()
    return _writer
