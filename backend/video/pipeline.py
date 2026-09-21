"""Per-camera processing pipeline.

One :class:`CameraPipeline` owns one camera and runs on its own thread —
OpenCV capture and ONNX inference both block, so they cannot share the API's
event loop. Per frame:

    read → throttle → detect → track → PPE → analyse → events → broadcast

Three performance decisions worth stating outright:

* **Detection runs every Nth frame** (`DETECT_EVERY_N_FRAMES`). The tracker is
  only advanced on detection frames; intermediate frames feed the clip buffer
  and keep the overlay smooth by reusing the last known boxes. Running the
  detector on every frame at 25 fps buys very little for pedestrian-speed
  motion and costs most of the CPU budget.
* **Frames are dropped, never queued** for live sources. The reader thread in
  :mod:`backend.video.source` keeps only the newest frame, so a slow analysis
  pass shows *current* footage rather than falling further behind.
* **Only normalised coordinates go over the wire.** The browser draws the
  overlay; the pipeline never re-encodes annotated video for the live view.

A failure in one camera is contained here — the loop catches, logs, backs off
and retries, so one dead RTSP stream cannot take the server down.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from backend.analysis import build_analysers
from backend.analysis.base import Analyser, FrameContext
from backend.analysis.movement import MovementAnalyser
from backend.analysis.zones import ResolvedZone, zones_for_points
from backend.config import settings
from backend.db.base import session_scope
from backend.db.models import Camera as CameraModel
from backend.events.bus import get_bus
from backend.events.engine import EventEngine, get_event_engine
from backend.inference.base import Detection, Detector
from backend.inference.ppe import PPEAssessment, PPEDetector, resolve_ppe_detector
from backend.inference.registry import resolve_detector
from backend.logging_conf import get_logger, throttled
from backend.tracking import ByteTracker, Track
from backend.tracking.recorder import TrackRecorder
from backend.video.ring_buffer import FrameRingBuffer
from backend.video.source import SourceStatus, VideoSource

logger = get_logger(__name__)

#: How often camera telemetry is flushed to the database. Per-frame writes
#: would dominate the SQLite write path for no benefit.
DB_FLUSH_INTERVAL = 3.0

#: How often zones are reloaded, so edits in the UI take effect on a live
#: camera without restarting it.
ZONE_RELOAD_INTERVAL = 5.0


@dataclass
class PipelineStats:
    """Live telemetry for one camera."""

    fps: float = 0.0
    people_count: int = 0
    active_tracks: int = 0
    inference_ms: float = 0.0
    processing_ms: float = 0.0
    frames_processed: int = 0
    frames_detected: int = 0
    frames_dropped: int = 0
    events_raised: int = 0
    started_at: float = 0.0
    backend: str = ""
    ppe_method: str = ""
    last_error: str = ""

    #: Rolling frame timestamps used to compute a smoothed fps.
    _samples: list[float] = field(default_factory=list, repr=False)

    def tick(self, now: float) -> None:
        self._samples.append(now)
        if len(self._samples) > 30:
            del self._samples[: len(self._samples) - 30]
        if len(self._samples) >= 2:
            span = self._samples[-1] - self._samples[0]
            self.fps = (len(self._samples) - 1) / span if span > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fps": round(self.fps, 2),
            "people_count": self.people_count,
            "active_tracks": self.active_tracks,
            "inference_ms": round(self.inference_ms, 2),
            "processing_ms": round(self.processing_ms, 2),
            "frames_processed": self.frames_processed,
            "frames_detected": self.frames_detected,
            "frames_dropped": self.frames_dropped,
            "events_raised": self.events_raised,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "backend": self.backend,
            "ppe_method": self.ppe_method,
            "last_error": self.last_error,
        }


class CameraPipeline:
    """Runs detection, tracking and analysis for a single camera."""

    def __init__(
        self,
        camera_id: str,
        name: str,
        source_type: str,
        stream_url: str,
        *,
        location: str = "",
        detector: Detector | None = None,
        ppe_detector: PPEDetector | None = None,
        engine: EventEngine | None = None,
        zones: Sequence[ResolvedZone] = (),
        profile: str = "thorough",
        analytics_enabled: bool = True,
        broadcast: bool = True,
        persist_telemetry: bool = True,
        loop: bool | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.location = location
        self.profile = profile
        self.analytics_enabled = analytics_enabled
        self.broadcast = broadcast
        self.persist_telemetry = persist_telemetry

        # `loop` may be set per camera in settings_json; file sources default
        # to looping (see VideoSource).
        self.source = VideoSource(
            source_type, stream_url, camera_id, name, loop=loop
        )
        self.detector = detector
        self.ppe_detector = ppe_detector
        self.engine = engine or get_event_engine()
        self.tracker = ByteTracker()
        self.analysers: list[Analyser] = build_analysers(profile)
        self.zones: list[ResolvedZone] = list(zones)
        self.buffer = FrameRingBuffer(settings.clip_buffer_seconds)
        self.stats = PipelineStats()
        # Durable per-person summaries behind the People Detected KPI and
        # historical person search.
        self.recorder = TrackRecorder(camera_id)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_tracks: list[Track] = []
        self._last_ppe: dict[int, PPEAssessment] = {}
        self._detect_counter = 0
        self._last_db_flush = 0.0
        self._last_zone_reload = 0.0
        self._latest_frame: np.ndarray | None = None
        self._zones_dirty = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.stats.started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name=f"pipeline-{self.camera_id[:8]}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to finish and wait for it.

        Clips still being assembled read from `self.buffer`, so the buffer is
        released only after they finish — clearing it first would silently
        truncate the evidence for the last events of a run.
        """
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self.source.close()

        from backend.events.evidence import get_evidence_writer

        if not get_evidence_writer().wait_for_clips(timeout=timeout):
            logger.warning(
                "Evidence clips still pending after %.0fs on %s; releasing buffer",
                timeout, self.name, extra={"camera": self.camera_id},
            )
        self.buffer.clear()
        self._thread = None

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def status(self) -> str:
        if not self.is_running:
            return "offline"
        return self.source.status.value

    def mark_zones_dirty(self) -> None:
        """Ask the worker to reload zones on its next iteration."""
        self._zones_dirty = True

    def set_zones(self, zones: Sequence[ResolvedZone]) -> None:
        with self._lock:
            self.zones = list(zones)

    def latest_frame(self) -> np.ndarray | None:
        """Most recent decoded frame — used by the MJPEG preview endpoint."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    # ── worker ───────────────────────────────────────────────────────────
    def _run(self) -> None:
        logger.info(
            "Pipeline starting for %s (profile=%s)", self.name, self.profile,
            extra={"camera": self.camera_id},
        )
        if self.detector is None:
            self.detector = resolve_detector()
        if self.ppe_detector is None:
            self.ppe_detector = resolve_ppe_detector()
        self.stats.backend = self.detector.name
        self.stats.ppe_method = self.ppe_detector.method

        frame_interval = 1.0 / settings.target_fps if settings.target_fps > 0 else 0.0
        next_frame_at = 0.0

        while not self._stop.is_set():
            try:
                # Connect / reconnect.
                needs_connection = not self.source.is_open or self.source.status in (
                    SourceStatus.OFFLINE,
                    SourceStatus.ERROR,
                )
                if needs_connection and not self._ensure_connected():
                    continue

                now = time.monotonic()
                if frame_interval and now < next_frame_at:
                    time.sleep(min(0.01, next_frame_at - now))
                    continue

                data = self.source.read()
                if data is None:
                    if self.source.status is SourceStatus.ENDED:
                        logger.info(
                            "Source ended for %s", self.name,
                            extra={"camera": self.camera_id},
                        )
                        self._publish_status("ended")
                        break
                    # Live source with nothing new yet.
                    time.sleep(0.005)
                    continue

                next_frame_at = time.monotonic() + frame_interval
                self._process(data.frame, data.monotonic, data.wall_time, data.index)

            except Exception as exc:
                # One bad frame, model hiccup or transient DB error must not
                # kill the camera thread.
                self.stats.last_error = str(exc)[:200]
                throttled(
                    logger,
                    f"pipeline-error-{self.camera_id}",
                    f"Pipeline error on {self.name}: {exc}",
                    level=40,
                    interval=10.0,
                    context={"camera": self.camera_id},
                )
                time.sleep(0.2)

        self.recorder.flush(force=True)
        self._flush_telemetry(force=True, status="offline")
        logger.info(
            "Pipeline stopped for %s", self.name, extra={"camera": self.camera_id}
        )

    def _ensure_connected(self) -> bool:
        """Open the source, honouring the backoff. Returns True when online."""
        if self.source.status is SourceStatus.IDLE and not self.source.is_open:
            if self.source.open():
                self._publish_status("online")
                return True
            self.source.schedule_retry()
            self._flush_telemetry(force=True, status="error")
            self._publish_status("error")
            return False

        if self.source.should_retry():
            if self.source.reconnect():
                self._publish_status("online")
                return True
            delay = self.source.schedule_retry()
            throttled(
                logger,
                f"reconnect-{self.camera_id}",
                f"Camera {self.name} still unreachable; next attempt in {delay:.0f}s",
                level=30,
                interval=30.0,
                context={"camera": self.camera_id},
            )
        # Sleep in small slices so stop() stays responsive.
        self._stop.wait(0.5)
        return False

    # ── per-frame work ───────────────────────────────────────────────────
    def _process(
        self, frame: np.ndarray, monotonic: float, wall_time: float, index: int
    ) -> None:
        started = time.perf_counter()
        height, width = frame.shape[:2]

        with self._lock:
            self._latest_frame = frame
        self.buffer.append(frame, monotonic)

        self._maybe_reload_zones(monotonic)

        # Detection cadence.
        self._detect_counter += 1
        should_detect = (self._detect_counter % settings.detect_every_n_frames) == 0

        tracks = self._last_tracks
        if should_detect and self.detector is not None:
            result = self.detector.infer(frame)
            self.stats.inference_ms = result.inference_ms
            self.stats.frames_detected += 1
            people: list[Detection] = [
                d for d in result.detections if d.label == "person"
            ]
            tracks = self.tracker.update(people, timestamp=monotonic)
            self._last_tracks = tracks

            # PPE assessment, on its own cadence.
            if (
                self.analytics_enabled
                and self.ppe_detector is not None
                and tracks
                and (self.stats.frames_detected % settings.ppe_every_n_detections) == 0
            ):
                try:
                    self._last_ppe = self.ppe_detector.assess(
                        frame, tracks, people, scope=str(self.camera_id)
                    )
                except Exception as exc:
                    throttled(
                        logger, f"ppe-error-{self.camera_id}",
                        f"PPE assessment failed on {self.name}: {exc}",
                        level=30, interval=30.0,
                        context={"camera": self.camera_id},
                    )

        self.stats.frames_processed += 1
        self.stats.people_count = len(tracks)
        self.stats.active_tracks = len(self.tracker.tracks)
        self.stats.frames_dropped = self.source.info()["frames_dropped"]
        self.stats.tick(monotonic)

        # Behaviour analysis + events, only on detection frames (analysers are
        # time-based, so running them on stale tracks would double-count dwell).
        if should_detect:
            zone_map = self._zone_map(tracks, width, height)
            self.recorder.observe(
                tracks,
                timestamp=monotonic,
                speeds=self._speeds(tracks),
                zone_map=zone_map,
                ppe=self._last_ppe,
            )
            if self.analytics_enabled:
                self._analyse(frame, tracks, monotonic, wall_time, index, width, height)
            # Retire summaries for tracks the tracker has dropped.
            self.recorder.flush({t.track_id for t in self.tracker.tracks})

        if self.broadcast:
            self._publish_detections(tracks, index, width, height, wall_time)

        self._flush_telemetry(now=monotonic)
        self.stats.processing_ms = (time.perf_counter() - started) * 1000.0

    def _analyse(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        monotonic: float,
        wall_time: float,
        index: int,
        width: int,
        height: int,
    ) -> None:
        with self._lock:
            zones = list(self.zones)

        context = FrameContext(
            camera_id=self.camera_id,
            frame_index=index,
            timestamp=monotonic,
            wall_time=wall_time,
            frame_width=width,
            frame_height=height,
            tracks=tracks,
            zones=zones,
            ppe=self._last_ppe,
        )

        candidates = []
        for analyser in self.analysers:
            try:
                candidates.extend(analyser.analyse(context))
            except Exception as exc:
                throttled(
                    logger, f"analyser-{analyser.name}-{self.camera_id}",
                    f"Analyser '{analyser.name}' failed on {self.name}: {exc}",
                    level=40, interval=30.0,
                    context={"camera": self.camera_id},
                )

        if not candidates:
            return

        # Annotate only when there is something to record — the overlay pass is
        # pure cost on a frame nobody will look at.
        from backend.video.annotate import annotate_frame

        speeds = self._speeds(tracks)
        annotated = annotate_frame(
            frame, tracks, zones, self._last_ppe, speeds,
            camera_name=self.name,
            stats={**self.stats.as_dict(), "people_count": len(tracks)},
            timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        accepted = self.engine.submit(
            candidates,
            camera_id=self.camera_id,
            frame=frame,
            annotated_frame=annotated,
            buffer=self.buffer,
            fps=max(1.0, self.stats.fps),
            monotonic_time=monotonic,
        )
        self.stats.events_raised += len(accepted)
        self.recorder.note_events([r.person_id for r in accepted])

    def _zone_map(
        self, tracks: list[Track], width: int, height: int
    ) -> dict[int, list[str]]:
        """track_id -> ids of the zones containing that person's foot point."""
        with self._lock:
            zones = list(self.zones)
        if not zones or not tracks:
            return {}
        points = np.array([t.foot_point for t in tracks], dtype=np.float32)
        memberships = zones_for_points(points, zones, width, height)
        return {t.track_id: ids for t, ids in zip(tracks, memberships, strict=False)}

    def _speeds(self, tracks: list[Track]) -> dict[int, float]:
        """Speeds from the movement analyser, if the profile enabled it."""
        for analyser in self.analysers:
            if isinstance(analyser, MovementAnalyser):
                return {t.track_id: analyser.speed_for(t) for t in tracks}
        return {}

    # ── publishing ───────────────────────────────────────────────────────
    def _publish_detections(
        self,
        tracks: list[Track],
        index: int,
        width: int,
        height: int,
        wall_time: float,
    ) -> None:
        """Push normalised boxes so the browser can draw the overlay."""
        zone_map = self._zone_map(tracks, width, height)
        speeds = self._speeds(tracks)
        boxes = []
        for track in tracks:
            assessment = self._last_ppe.get(track.track_id)
            x1, y1, x2, y2 = track.bbox
            boxes.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "confidence": round(track.confidence, 3),
                    # Normalised so the client scales to any rendered size.
                    "bbox": [
                        round(x1 / width, 5), round(y1 / height, 5),
                        round(x2 / width, 5), round(y2 / height, 5),
                    ],
                    "helmet": assessment.helmet if assessment else None,
                    "vest": assessment.vest if assessment else None,
                    "ppe_method": assessment.method if assessment else None,
                    "violation": bool(assessment and assessment.is_violation),
                    "speed": round(speeds.get(track.track_id, 0.0), 2),
                    "state": track.state.name.lower(),
                    "zones": zone_map.get(track.track_id, []),
                }
            )

        get_bus().publish(
            "detections",
            {
                "camera_id": self.camera_id,
                "frame_index": index,
                "timestamp": datetime.fromtimestamp(
                    wall_time, tz=UTC
                ).isoformat(),
                "people_count": len(tracks),
                "boxes": boxes,
                "fps": round(self.stats.fps, 2),
                "inference_ms": round(self.stats.inference_ms, 2),
                "processing_ms": round(self.stats.processing_ms, 2),
                "backend": self.stats.backend,
            },
            camera_id=self.camera_id,
        )

    def _publish_status(self, status: str) -> None:
        get_bus().publish(
            "camera_status",
            {
                "camera_id": self.camera_id,
                "name": self.name,
                "status": status,
                "last_error": self.source.last_error,
                **self.stats.as_dict(),
            },
            camera_id=self.camera_id,
        )

    # ── persistence ──────────────────────────────────────────────────────
    def _maybe_reload_zones(self, now: float) -> None:
        if not (
            self._zones_dirty or now - self._last_zone_reload > ZONE_RELOAD_INTERVAL
        ):
            return
        self._last_zone_reload = now
        self._zones_dirty = False
        try:
            from backend.db.repository import load_zones_for_camera

            self.set_zones(load_zones_for_camera(self.camera_id))
        except Exception as exc:
            throttled(
                logger, f"zone-reload-{self.camera_id}",
                f"Could not reload zones for {self.name}: {exc}",
                level=30, interval=60.0, context={"camera": self.camera_id},
            )

    def _flush_telemetry(
        self, now: float | None = None, force: bool = False, status: str | None = None
    ) -> None:
        """Write camera telemetry to the database, rate-limited."""
        if not self.persist_telemetry:
            return
        now = time.monotonic() if now is None else now
        if not force and (now - self._last_db_flush) < DB_FLUSH_INTERVAL:
            return
        self._last_db_flush = now
        try:
            with session_scope() as session:
                camera = session.get(CameraModel, self.camera_id)
                if camera is None:
                    return
                camera.status = status or self.source.status.value
                camera.fps = round(self.stats.fps, 2)
                camera.people_count = self.stats.people_count
                camera.last_error = self.source.last_error or None
                if self.source.status is SourceStatus.ONLINE:
                    camera.last_frame_at = datetime.now(UTC)
        except Exception as exc:
            throttled(
                logger, f"telemetry-{self.camera_id}",
                f"Could not persist telemetry for {self.name}: {exc}",
                level=30, interval=60.0, context={"camera": self.camera_id},
            )

    # ── introspection ────────────────────────────────────────────────────
    def runtime(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "status": self.status,
            **self.stats.as_dict(),
            "source": self.source.info(),
            "buffer": self.buffer.stats(),
            "zones": len(self.zones),
            "people_recorded": self.recorder.written,
            "profile": self.profile,
            "analytics_enabled": self.analytics_enabled,
        }
