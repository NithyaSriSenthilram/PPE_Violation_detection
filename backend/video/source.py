"""Camera abstraction.

One class handles uploaded files, local webcams and RTSP/HTTP streams so the
pipeline never branches on source type. The differences that genuinely matter
are handled here:

* **Live sources** (webcam, RTSP) must stay *current*. OpenCV buffers frames
  internally, so a pipeline slower than the stream drifts further behind
  real time with every frame. Live sources therefore run a reader thread that
  keeps only the newest frame and drops the rest.
* **File sources** must not drop anything — every frame of an uploaded video
  should be analysed — and they end, which live sources do not.

Failures are contained: a bad URL, an unplugged camera or a corrupt file marks
*that* source unavailable with a readable reason. It never raises into the
pipeline loop, and a reconnect is retried with exponential backoff.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config import settings
from backend.logging_conf import get_logger, throttled

logger = get_logger(__name__)


def _configure_ffmpeg_timeouts() -> None:
    """Bound how long FFmpeg waits on an unreachable stream.

    OpenCV's default RTSP behaviour is to block for ~30 s before admitting a
    host is unreachable, which holds a pipeline thread hostage and makes a
    mistyped URL look like a hung application. FFmpeg only reads these options
    when a capture is constructed, and only from this environment variable.
    """
    micros = int(settings.stream_open_timeout_seconds * 1_000_000)
    options = [
        "rtsp_transport;tcp",       # TCP survives lossy networks; UDP stalls
        f"timeout;{micros}",        # socket timeout (current FFmpeg)
        f"stimeout;{micros}",       # same, for older FFmpeg builds
        "reconnect;1",
        "reconnect_streamed;1",
        f"analyzeduration;{min(micros, 2_000_000)}",
    ]
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "|".join(options))


_configure_ffmpeg_timeouts()


class SourceStatus(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    ENDED = "ended"


@dataclass(slots=True)
class FrameData:
    """One decoded frame with its timing metadata."""

    frame: np.ndarray
    index: int
    #: Monotonic clock — safe for measuring durations.
    monotonic: float
    #: Epoch seconds — for records shown to people.
    wall_time: float
    #: Position within the source video, when it has one.
    video_timestamp: float | None = None

    @property
    def width(self) -> int:
        return self.frame.shape[1]

    @property
    def height(self) -> int:
        return self.frame.shape[0]


class VideoSource:
    """A uniform, resilient reader over any supported video input."""

    LIVE_TYPES = frozenset({"rtsp", "webcam", "http"})

    def __init__(
        self,
        source_type: str,
        stream_url: str,
        camera_id: str = "",
        name: str = "",
        max_width: int | None = None,
        loop: bool | None = None,
    ) -> None:
        self.source_type = source_type
        self.stream_url = stream_url
        self.camera_id = camera_id
        self.name = name or camera_id or stream_url
        self.max_width = settings.max_stream_width if max_width is None else max_width
        # File sources rewind by default: replaying a recording as a camera is
        # an explicitly supported mode, and stopping at the last frame makes it
        # useless for that. Uploaded-video *analysis* is a separate path and is
        # never looped.
        self.loop = (source_type == "file") if loop is None else loop
        self.loops_completed = 0

        self.status: SourceStatus = SourceStatus.IDLE
        self.last_error: str = ""

        self._capture: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame_index = 0

        # Live-source reader thread state.
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest: FrameData | None = None
        self._latest_lock = threading.Lock()
        self._dropped = 0

        # Reconnect backoff.
        self._backoff = settings.reconnect_backoff_seconds
        self._next_retry_at = 0.0

        # Source properties, populated on open.
        self.source_fps = 0.0
        self.total_frames = 0
        self.width = 0
        self.height = 0

    # ── properties ───────────────────────────────────────────────────────
    @property
    def is_live(self) -> bool:
        return self.source_type in self.LIVE_TYPES

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def duration_seconds(self) -> float:
        if self.total_frames and self.source_fps:
            return self.total_frames / self.source_fps
        return 0.0

    # ── resolution of the underlying handle ──────────────────────────────
    def _resolve_target(self) -> int | str:
        """Translate the configured URL into something OpenCV accepts."""
        if self.source_type == "webcam":
            try:
                return int(self.stream_url)
            except ValueError as exc:
                raise ValueError(
                    f"webcam source needs a device index, got {self.stream_url!r}"
                ) from exc

        if self.source_type == "file":
            path = Path(self.stream_url)
            if not path.is_absolute():
                path = settings.upload_root / path.name
            resolved = path.resolve()
            # Containment check: a camera row must not be able to point the
            # decoder at an arbitrary file on disk.
            if not resolved.is_relative_to(settings.upload_root.resolve()):
                raise ValueError(
                    "file source must live inside the configured upload directory"
                )
            if not resolved.is_file():
                raise FileNotFoundError(f"video file not found: {resolved.name}")
            return str(resolved)

        return self.stream_url

    # ── lifecycle ────────────────────────────────────────────────────────
    def open(self) -> bool:
        """Open the source. Returns False and records `last_error` on failure."""
        with self._lock:
            if self.is_open:
                return True

            self.status = SourceStatus.CONNECTING
            try:
                target = self._resolve_target()
            except (ValueError, FileNotFoundError) as exc:
                self._fail(str(exc))
                return False

            try:
                if self.source_type in ("rtsp", "http"):
                    capture = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
                    # A small buffer keeps latency down on live streams.
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                else:
                    capture = cv2.VideoCapture(target)
            except Exception as exc:
                self._fail(f"could not open source: {exc}")
                return False

            if not capture.isOpened():
                capture.release()
                self._fail(
                    f"could not open {self.source_type} source — check the URL, "
                    "codec support and that the device is not already in use"
                )
                return False

            # Probe a frame: an "opened" capture can still fail to decode, which
            # is what a corrupt or zero-length file looks like.
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                capture.release()
                self._fail(
                    "source opened but produced no decodable frames "
                    "(empty, truncated, or unsupported codec)"
                )
                return False

            self.source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.height, self.width = frame.shape[:2]

            if self.source_type == "file":
                # Rewind past the probe frame so analysis starts at frame 0.
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self._capture = capture
            self.status = SourceStatus.ONLINE
            self.last_error = ""
            self._backoff = settings.reconnect_backoff_seconds
            self._frame_index = 0

        logger.info(
            "Camera connected: %s (%s, %dx%d @ %.1f fps%s)",
            self.name, self.source_type, self.width, self.height, self.source_fps,
            f", {self.total_frames} frames" if self.total_frames else "",
            extra={"camera": self.camera_id or self.name},
        )

        if self.is_live:
            self._start_reader()
        return True

    def _fail(self, reason: str) -> None:
        self.last_error = reason
        self.status = SourceStatus.ERROR
        logger.warning(
            "Camera unavailable: %s — %s", self.name, reason,
            extra={"camera": self.camera_id or self.name},
        )

    def close(self) -> None:
        """Release the source and stop any reader thread."""
        self._stop.set()
        reader = self._reader
        if reader and reader.is_alive() and reader is not threading.current_thread():
            reader.join(timeout=2.0)
        self._reader = None
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
        with self._latest_lock:
            self._latest = None
        if self.status is not SourceStatus.ERROR:
            self.status = SourceStatus.IDLE

    # ── live reader thread ───────────────────────────────────────────────
    def _start_reader(self) -> None:
        self._stop.clear()
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"reader-{self.camera_id or self.name}"[:32],
            daemon=True,
        )
        self._reader.start()

    def _read_loop(self) -> None:
        """Keep only the newest frame; stale frames are worthless when live."""
        consecutive_failures = 0
        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                throttled(
                    logger,
                    f"read-fail-{self.camera_id}",
                    f"Camera {self.name}: frame read failed "
                    f"({consecutive_failures} consecutive)",
                    level=30,
                    interval=15.0,
                    context={"camera": self.camera_id or self.name},
                )
                # ~1 s of solid failure at 30 fps means the stream is gone.
                if consecutive_failures >= 30:
                    self.status = SourceStatus.OFFLINE
                    self.last_error = "stream stopped delivering frames"
                    logger.warning(
                        "Camera disconnected: %s", self.name,
                        extra={"camera": self.camera_id or self.name},
                    )
                    break
                time.sleep(0.03)
                continue

            consecutive_failures = 0
            now = time.monotonic()
            self._frame_index += 1
            prepared = self._prepare(frame)
            data = FrameData(
                frame=prepared,
                index=self._frame_index,
                monotonic=now,
                wall_time=time.time(),
            )
            with self._latest_lock:
                if self._latest is not None:
                    self._dropped += 1
                self._latest = data

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Apply the decode-side downscale cap."""
        from backend.inference.preprocess import downscale_to_width

        return downscale_to_width(frame, self.max_width)

    # ── frame access ─────────────────────────────────────────────────────
    def read(self) -> FrameData | None:
        """Next frame, or None when unavailable/ended.

        Live sources return the most recent frame (never a backlog); file
        sources return the next sequential frame.
        """
        if self.is_live:
            with self._latest_lock:
                data, self._latest = self._latest, None
            if data is None and self.status is SourceStatus.OFFLINE:
                return None
            return data

        capture = self._capture
        if capture is None:
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            if self.loop and self.total_frames > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.loops_completed += 1
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.status = SourceStatus.ENDED
                    return None
            else:
                self.status = SourceStatus.ENDED
                return None

        self._frame_index += 1
        position = capture.get(cv2.CAP_PROP_POS_MSEC)
        return FrameData(
            frame=self._prepare(frame),
            index=self._frame_index,
            monotonic=time.monotonic(),
            wall_time=time.time(),
            video_timestamp=(position / 1000.0) if position and position > 0 else None,
        )

    # ── reconnection ─────────────────────────────────────────────────────
    def should_retry(self) -> bool:
        """True when the backoff has elapsed and a reconnect is due."""
        if self.status not in (SourceStatus.OFFLINE, SourceStatus.ERROR):
            return False
        return time.monotonic() >= self._next_retry_at

    def schedule_retry(self) -> float:
        """Arm the next reconnect attempt. Returns the delay applied."""
        delay = self._backoff
        self._next_retry_at = time.monotonic() + delay
        self._backoff = min(
            self._backoff * 2, settings.reconnect_max_backoff_seconds
        )
        return delay

    def reconnect(self) -> bool:
        """Close and reopen. Backoff is preserved on failure."""
        logger.info(
            "Reconnecting camera %s ...", self.name,
            extra={"camera": self.camera_id or self.name},
        )
        self.close()
        self.status = SourceStatus.CONNECTING
        return self.open()

    # ── introspection ────────────────────────────────────────────────────
    def info(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source_type": self.source_type,
            "status": self.status.value,
            "is_live": self.is_live,
            "width": self.width,
            "height": self.height,
            "source_fps": round(self.source_fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "frames_read": self._frame_index,
            "frames_dropped": self._dropped,
            "loop": self.loop,
            "loops_completed": self.loops_completed,
            "last_error": self.last_error,
        }


def probe_video_file(path: Path) -> dict[str, Any]:
    """Inspect a video file without starting a pipeline.

    Used to validate an upload before queueing analysis, so a corrupt file is
    rejected at the API boundary with a clear message.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"ok": False, "error": "file could not be opened by the decoder"}
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            return {
                "ok": False,
                "error": "file contains no decodable video frames "
                         "(empty, truncated, or unsupported codec)",
            }
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        height, width = frame.shape[:2]
        return {
            "ok": True,
            "width": width,
            "height": height,
            "fps": round(fps, 2) if fps > 0 else 0.0,
            "total_frames": frames,
            "duration_seconds": round(frames / fps, 2) if fps > 0 and frames else 0.0,
        }
    except Exception as exc:
        return {"ok": False, "error": f"probe failed: {exc}"}
    finally:
        capture.release()
