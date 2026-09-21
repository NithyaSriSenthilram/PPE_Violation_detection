"""Multi-camera lifecycle manager.

Owns every :class:`CameraPipeline` and is the only thing the API talks to when
it needs to start, stop or inspect a camera. Pipelines share one detector
instance — model weights and the ONNX session are expensive, and ONNX Runtime
sessions are safe to call from multiple threads — while each camera keeps its
own tracker, analysers and clip buffer, because all of those are per-camera
state.

Every operation is defensive: starting a camera that cannot connect leaves the
others running, and `start_all` reports failures instead of aborting.
"""

from __future__ import annotations

import threading
from typing import Any

from backend.config import settings
from backend.db.base import session_scope
from backend.db.models import Camera as CameraModel
from backend.db.repository import list_cameras, load_zones_for_camera
from backend.inference.ppe import PPEDetector, resolve_ppe_detector
from backend.inference.registry import resolve_detector
from backend.logging_conf import get_logger
from backend.video.pipeline import CameraPipeline

logger = get_logger(__name__)


class CameraManager:
    """Registry and lifecycle controller for camera pipelines."""

    def __init__(self) -> None:
        self._pipelines: dict[str, CameraPipeline] = {}
        self._lock = threading.RLock()
        self._detector = None
        self._ppe_detector: PPEDetector | None = None

    # ── shared resources ─────────────────────────────────────────────────
    def _shared_detector(self):
        if self._detector is None:
            self._detector = resolve_detector()
        return self._detector

    def _shared_ppe(self) -> PPEDetector:
        if self._ppe_detector is None:
            self._ppe_detector = resolve_ppe_detector()
        return self._ppe_detector

    # ── queries ──────────────────────────────────────────────────────────
    def get(self, camera_id: str) -> CameraPipeline | None:
        with self._lock:
            return self._pipelines.get(camera_id)

    def all(self) -> dict[str, CameraPipeline]:
        with self._lock:
            return dict(self._pipelines)

    def runtimes(self) -> dict[str, dict[str, Any]]:
        """Live telemetry for every running camera."""
        return {cid: p.runtime() for cid, p in self.all().items()}

    @property
    def online_count(self) -> int:
        return sum(1 for p in self.all().values() if p.status == "online")

    # ── lifecycle ────────────────────────────────────────────────────────
    def start_camera(self, camera_id: str) -> tuple[bool, str]:
        """Start (or restart) one camera. Returns ``(started, message)``."""
        with self._lock:
            existing = self._pipelines.get(camera_id)
            if existing and existing.is_running:
                return True, "already running"

            with session_scope() as session:
                camera = session.get(CameraModel, camera_id)
                if camera is None:
                    return False, "camera not found"
                if not camera.enabled:
                    return False, "camera is disabled"
                record = {
                    "name": camera.name,
                    "location": camera.location,
                    "source_type": camera.source_type,
                    "stream_url": camera.stream_url,
                    "analytics_enabled": camera.analytics_enabled,
                    "profile": (camera.settings_json or {}).get("profile", "thorough"),
                    "loop": (camera.settings_json or {}).get("loop"),
                }

            try:
                zones = load_zones_for_camera(camera_id)
            except Exception as exc:
                logger.warning("Could not load zones for %s: %s", camera_id, exc)
                zones = []

            pipeline = CameraPipeline(
                camera_id=camera_id,
                name=record["name"],
                source_type=record["source_type"],
                stream_url=record["stream_url"],
                location=record["location"],
                detector=self._shared_detector(),
                ppe_detector=self._shared_ppe(),
                zones=zones,
                profile=record["profile"],
                analytics_enabled=record["analytics_enabled"],
                loop=record["loop"],
            )
            self._pipelines[camera_id] = pipeline
            pipeline.start()
            return True, "starting"

    def stop_camera(self, camera_id: str) -> bool:
        """Stop and deregister one camera."""
        with self._lock:
            pipeline = self._pipelines.pop(camera_id, None)
        if pipeline is None:
            return False
        pipeline.stop()
        return True

    def restart_camera(self, camera_id: str) -> tuple[bool, str]:
        self.stop_camera(camera_id)
        return self.start_camera(camera_id)

    def start_all(self) -> dict[str, str]:
        """Start every enabled camera. One failure does not stop the rest."""
        with session_scope() as session:
            camera_ids = [c.camera_id for c in list_cameras(session, enabled_only=True)]

        results: dict[str, str] = {}
        for camera_id in camera_ids:
            try:
                started, message = self.start_camera(camera_id)
                results[camera_id] = message if started else f"failed: {message}"
            except Exception as exc:
                logger.exception("Failed to start camera %s", camera_id)
                results[camera_id] = f"error: {exc}"

        if results:
            logger.info(
                "Camera startup: %d requested, %d accepted",
                len(results),
                sum(1 for v in results.values() if not v.startswith(("failed", "error"))),
            )
        return results

    def stop_all(self) -> None:
        """Stop every pipeline. Called on application shutdown."""
        for camera_id in list(self.all()):
            try:
                self.stop_camera(camera_id)
            except Exception as exc:
                logger.warning("Error stopping camera %s: %s", camera_id, exc)

    # ── zone propagation ─────────────────────────────────────────────────
    def notify_zones_changed(self, camera_id: str) -> None:
        """Tell a running pipeline its zones were edited."""
        pipeline = self.get(camera_id)
        if pipeline is not None:
            pipeline.mark_zones_dirty()

    # ── diagnostics ──────────────────────────────────────────────────────
    def info(self) -> dict[str, Any]:
        pipelines = self.all()
        return {
            "cameras_registered": len(pipelines),
            "cameras_online": self.online_count,
            "target_fps": settings.target_fps,
            "detect_every_n_frames": settings.detect_every_n_frames,
            "max_stream_width": settings.max_stream_width,
            "clip_buffer_seconds": settings.clip_buffer_seconds,
            "ppe_method": self._ppe_detector.method if self._ppe_detector else None,
            "buffer_memory_mb": round(
                sum(p.buffer.memory_bytes for p in pipelines.values()) / 1e6, 2
            ),
            "pipelines": {cid: p.runtime() for cid, p in pipelines.items()},
        }


_manager: CameraManager | None = None


def get_manager() -> CameraManager:
    """Process-wide camera manager."""
    global _manager
    if _manager is None:
        _manager = CameraManager()
    return _manager
