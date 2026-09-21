"""Backend resolution.

`INFERENCE_BACKEND=auto` walks :data:`AUTO_ORDER` and activates the first
backend that both probes clean *and* successfully loads its model. An explicit
name is honoured strictly: if it cannot load, that is an error the operator
should see, not something to paper over — except that the platform still comes
up on the mock backend so the dashboard and API stay usable while it is fixed.

Whatever happens, :func:`describe_backends` reports the true state of every
backend, including the exact reason each unavailable one failed.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.inference.backends.max_backend import MaxDetector
from backend.inference.backends.mock_backend import MockDetector
from backend.inference.backends.onnx_backend import OnnxDetector
from backend.inference.backends.ultralytics_backend import UltralyticsDetector
from backend.inference.base import BackendUnavailable, Detector
from backend.inference.mojo.bridge import get_bridge
from backend.logging_conf import get_logger
from backend.schemas import BackendInfo

logger = get_logger(__name__)

REGISTRY: dict[str, type[Detector]] = {
    "max": MaxDetector,
    "onnx": OnnxDetector,
    "ultralytics": UltralyticsDetector,
    "mock": MockDetector,
}

#: Resolution order for `auto`. MAX first because it is the fastest path where
#: it is supported; ONNX Runtime is the dependable default; mock is last and
#: only ever reached when nothing real can load.
AUTO_ORDER: tuple[str, ...] = ("max", "onnx", "ultralytics", "mock")

#: Resolution order for a *secondary* model such as PPE. Identical, minus the
#: mock: synthetic detections are a way to keep the dashboard usable when no
#: engine loads, never something that should reach a safety verdict. A PPE
#: model that cannot load produces no PPE readings at all, and diagnostics say
#: exactly why.
SECONDARY_ORDER: tuple[str, ...] = ("max", "onnx", "ultralytics")


class _State:
    detector: Detector | None = None
    requested: str = "auto"
    resolved: str = ""
    attempts: list[tuple[str, str]] = []


_state = _State()


def resolve_detector(force: str | None = None) -> Detector:
    """Load and return the active detector, creating it on first call."""
    if _state.detector is not None and force is None:
        return _state.detector

    requested = (force or settings.inference_backend).lower()
    _state.requested = requested
    _state.attempts = []

    candidates: tuple[str, ...]
    if requested == "auto":
        candidates = AUTO_ORDER
    elif requested in REGISTRY:
        # Explicit choice, with mock as a last-resort so the server still boots.
        candidates = (requested,) if requested == "mock" else (requested, "mock")
    else:
        logger.error(
            "Unknown INFERENCE_BACKEND=%r; valid values are %s",
            requested, ["auto", *REGISTRY],
        )
        candidates = AUTO_ORDER

    for name in candidates:
        detector = REGISTRY[name]()
        try:
            detector.load()
        except BackendUnavailable as exc:
            _state.attempts.append((name, str(exc)))
            logger.info("Backend '%s' unavailable: %s", name, exc)
            continue
        except Exception as exc:  # a backend bug must not take the server down
            _state.attempts.append((name, f"unexpected error: {exc}"))
            logger.exception("Backend '%s' raised while loading", name)
            continue

        _state.detector = detector
        _state.resolved = name
        if name == "mock":
            logger.warning(
                "No real inference backend could load — running on SYNTHETIC "
                "detections. Detections and events are not real. See "
                "GET /api/diagnostics.",
                extra={"backend": name},
            )
        else:
            logger.info("AI inference backend: %s", name, extra={"backend": name})
        return detector

    # MockDetector.load() cannot fail, so this is unreachable in practice.
    raise RuntimeError("no inference backend could be initialised")


def build_detector(
    model_path: Path,
    labels_path: Path,
    *,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    purpose: str = "secondary",
) -> tuple[Detector, list[tuple[str, str]]]:
    """Load a second model on the first backend that can take it.

    This is how the PPE model reaches the hardware: it walks the same backend
    chain as the person detector — so MAX stays first, ONNX Runtime with
    CoreML remains the working path on Apple Silicon, and the Mojo-accelerated
    decode/NMS in :mod:`backend.inference.postprocess` is shared — without any
    engine-specific code being written twice.

    Returns the loaded detector plus the ``(backend, reason)`` list of what was
    tried and why it failed, which diagnostics reports verbatim.

    Raises :class:`BackendUnavailable` when nothing can load it. Unlike the
    primary path there is no mock to fall back to, and callers are expected to
    degrade gracefully rather than serve invented results.
    """
    attempts: list[tuple[str, str]] = []
    requested = settings.inference_backend.lower()
    candidates = (
        SECONDARY_ORDER
        if requested in ("auto", "mock") or requested not in REGISTRY
        else (requested, *[n for n in SECONDARY_ORDER if n != requested])
    )

    for name in candidates:
        try:
            detector = REGISTRY[name](
                model_path=model_path,
                labels_path=labels_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            detector.load()
        except BackendUnavailable as exc:
            attempts.append((name, str(exc)))
            continue
        except Exception as exc:
            attempts.append((name, f"unexpected error: {exc}"))
            logger.debug("Backend '%s' raised loading %s", name, model_path, exc_info=True)
            continue

        logger.info(
            "%s model %s loaded on the %s backend",
            purpose, model_path.name, name, extra={"backend": name},
        )
        return detector, attempts

    detail = "; ".join(f"{n}: {r}" for n, r in attempts) or "no backends available"
    raise BackendUnavailable(
        f"no inference backend could load the {purpose} model {model_path.name} "
        f"({detail})"
    )


def get_detector() -> Detector | None:
    """Currently active detector, or None if never resolved."""
    return _state.detector


def active_backend_name() -> str:
    """Name of the running backend — the value diagnostics reports."""
    return _state.resolved or "none"


def reset_detector() -> None:
    """Tear down the active detector (tests, config reload)."""
    if _state.detector is not None:
        _state.detector.close()
    _state.detector = None
    _state.resolved = ""
    _state.attempts = []


def describe_backends() -> list[BackendInfo]:
    """Truthful per-backend availability report."""
    failures = dict(_state.attempts)
    infos: list[BackendInfo] = []
    for name, cls in REGISTRY.items():
        try:
            available, reason, detail = cls.probe()
        except Exception as exc:  # probe() is documented not to raise; be safe
            available, reason, detail = False, f"probe failed: {exc}", {}

        is_active = name == _state.resolved
        if not is_active and name in failures:
            # A backend can probe OK yet fail to load (MAX + ONNX is exactly
            # this case). Report the load failure — it is the real reason.
            reason = failures[name]
            available = False

        infos.append(
            BackendInfo(
                name=name,
                available=available or is_active,
                active=is_active,
                reason=reason,
                detail=detail,
            )
        )
    return infos


def inference_info() -> dict[str, Any]:
    """Everything the diagnostics endpoint needs about inference."""
    detector = _state.detector
    return {
        "requested_backend": _state.requested,
        "active_backend": active_backend_name(),
        "model_path": str(settings.model_file),
        "model_present": settings.model_file.exists(),
        "confidence_threshold": settings.confidence_threshold,
        "nms_iou_threshold": settings.nms_iou_threshold,
        "input_size": [settings.inference_width, settings.inference_height],
        "is_synthetic": _state.resolved == "mock",
        "stats": detector.stats() if detector else {},
        "fallback_chain": [
            {"backend": n, "reason": r} for n, r in _state.attempts
        ],
    }


def video_output_info() -> dict[str, Any]:
    """State of the annotated-video pipeline, for diagnostics.

    Lives here beside the other truthful capability reports. The questions it
    answers are the ones that decide whether an analysis will produce something
    watchable: can we write, and is ffmpeg there to make the result seekable.
    """
    import os

    from backend.video.writer import ffmpeg_path, ffprobe_path

    directory = settings.processed_root
    renders = sorted(directory.glob("*_annotated.mp4")) if directory.is_dir() else []
    total = sum(f.stat().st_size for f in renders)
    return {
        "enabled": settings.annotated_video_enabled,
        "processed_dir": str(directory),
        "processed_dir_exists": directory.is_dir(),
        "processed_dir_writable": directory.is_dir()
        and os.access(directory, os.W_OK),
        "render_count": len(renders),
        "render_mb": round(total / 1e6, 1),
        "ffmpeg": ffmpeg_path() or "",
        "ffprobe": ffprobe_path() or "",
        "ffmpeg_finalise": settings.annotated_ffmpeg_finalise,
        "max_width": settings.annotated_max_width,
        "event_overlay_seconds": settings.event_overlay_seconds,
        "crf": settings.annotated_crf,
        # What the render actually draws, so an operator can tell at a glance
        # why there are — or are not — boxes around the people in it.
        "show_ppe_boxes": settings.show_ppe_boxes,
        "show_person_boxes": settings.show_person_boxes,
        "show_track_id": settings.show_track_id,
    }


def platform_info() -> dict[str, Any]:
    """Host description, used by diagnostics."""
    import os

    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }


def mojo_info() -> dict[str, Any]:
    """Mojo kernel status, including the per-kernel benchmark results."""
    return get_bridge().info()
