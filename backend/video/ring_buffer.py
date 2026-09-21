"""Rolling pre-event frame buffer.

Frames are held **JPEG-encoded**, not raw. At 960×540 a raw BGR frame is
~1.5 MB, so an 8-second buffer at 12 fps would cost ~150 MB *per camera* —
unaffordable on a modest box running several streams. JPEG at quality 80 is
roughly 60 KB, bringing the same buffer to ~6 MB per camera. Decoding happens
once, when a clip is actually written.

The buffer is time-based rather than count-based so `CLIP_BUFFER_SECONDS`
means the same thing whatever the pipeline's frame rate.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class BufferedFrame:
    """One JPEG-encoded frame with its capture time."""

    timestamp: float
    jpeg: bytes
    width: int
    height: int

    def decode(self) -> np.ndarray | None:
        """Decode back to BGR. Returns None if the buffer was corrupted."""
        array = np.frombuffer(self.jpeg, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)


class FrameRingBuffer:
    """Thread-safe, time-bounded ring buffer of recent frames."""

    def __init__(self, seconds: float, jpeg_quality: int = 80) -> None:
        self.seconds = max(0.0, seconds)
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 100))
        self._frames: deque[BufferedFrame] = deque()
        self._lock = threading.Lock()
        self._bytes = 0

    def append(self, frame: np.ndarray, timestamp: float) -> None:
        """Encode and store a frame, evicting anything older than the window."""
        if self.seconds <= 0:
            return
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        height, width = frame.shape[:2]
        item = BufferedFrame(timestamp, encoded.tobytes(), width, height)
        with self._lock:
            self._frames.append(item)
            self._bytes += len(item.jpeg)
            cutoff = timestamp - self.seconds
            while self._frames and self._frames[0].timestamp < cutoff:
                self._bytes -= len(self._frames.popleft().jpeg)

    def snapshot(self, since: float | None = None) -> list[BufferedFrame]:
        """Copy out the buffered frames, optionally only those after `since`."""
        with self._lock:
            frames = list(self._frames)
        if since is None:
            return frames
        return [f for f in frames if f.timestamp >= since]

    def latest(self) -> BufferedFrame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._bytes = 0

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def memory_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            count = len(self._frames)
            span = (
                self._frames[-1].timestamp - self._frames[0].timestamp
                if count > 1
                else 0.0
            )
            return {
                "frames": count,
                "seconds_buffered": round(span, 2),
                "memory_mb": round(self._bytes / 1e6, 2),
                "window_seconds": self.seconds,
            }
