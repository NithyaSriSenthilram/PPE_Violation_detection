"""Inference abstraction.

The pipeline depends only on :class:`Detector` and :class:`Detection`. Which
engine actually runs — ONNX Runtime, Ultralytics/PyTorch, MAX, or the
deterministic mock — is decided once at startup by
:mod:`backend.inference.registry` and is invisible to callers.

Adding a backend means subclassing :class:`Detector` and registering it; no
other module changes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used in a runtime-evaluated annotation
from typing import Any

import numpy as np


@dataclass(slots=True)
class Detection:
    """One detected object in **source-frame pixel** coordinates.

    Backends are responsible for undoing any letterbox/resize they applied, so
    consumers never need to know the network input size.
    """

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box.

        Zone membership uses this rather than the centroid: a person standing
        just outside a floor-marked zone should not trigger it merely because
        their torso overhangs the boundary.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def as_list(self) -> list[float]:
        return [float(v) for v in self.bbox]


@dataclass(slots=True)
class InferenceResult:
    """Detections for one frame plus the timings the dashboard reports."""

    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    backend: str = ""

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def of_class(self, label: str) -> list[Detection]:
        return [d for d in self.detections if d.label == label]


class BackendUnavailable(RuntimeError):
    """Raised by :meth:`Detector.load` when the backend cannot be used.

    The message is surfaced verbatim in ``GET /api/diagnostics`` so operators
    see the real reason — a missing package, an unsupported model format —
    rather than a silent downgrade.
    """


class Detector(ABC):
    """Common interface for every detection backend."""

    #: Short identifier, matches the `INFERENCE_BACKEND` value.
    name: str = "base"
    #: Human-readable description for diagnostics.
    description: str = ""

    def __init__(
        self,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> None:
        self._loaded = False
        self._load_error: str = ""
        self._detail: dict[str, Any] = {}
        self._frames = 0
        self._total_ms = 0.0
        # None means "follow the global setting". A second detector loaded for
        # a different model — the PPE network — needs its own thresholds, since
        # it was trained on a different dataset with a different score profile.
        self._conf_override = conf_threshold
        self._iou_override = iou_threshold

    @property
    def conf_threshold(self) -> float:
        from backend.config import settings

        return (
            settings.confidence_threshold
            if self._conf_override is None
            else self._conf_override
        )

    @property
    def iou_threshold(self) -> float:
        from backend.config import settings

        return (
            settings.nms_iou_threshold
            if self._iou_override is None
            else self._iou_override
        )

    # ── Lifecycle ────────────────────────────────────────────────────────
    @classmethod
    @abstractmethod
    def probe(cls, model_path: Path | None = None) -> tuple[bool, str, dict[str, Any]]:
        """Report whether this backend *could* run, without loading a model.

        Returns ``(available, reason, detail)``. `reason` explains a negative
        result and is shown in diagnostics. Must never raise.

        `model_path` names the weights to check. It defaults to `MODEL_PATH`,
        the person detector; the PPE detector passes its own path so the same
        backends can be probed and loaded for a second model.
        """

    @abstractmethod
    def load(self) -> None:
        """Load the model. Raises :class:`BackendUnavailable` on failure."""

    @abstractmethod
    def _infer(self, frame: np.ndarray) -> InferenceResult:
        """Backend-specific inference on a BGR uint8 frame."""

    def close(self) -> None:
        """Release resources. Default is a no-op."""
        self._loaded = False

    # ── Public API ───────────────────────────────────────────────────────
    def infer(self, frame: np.ndarray) -> InferenceResult:
        """Run detection, tracking rolling latency statistics."""
        if not self._loaded:
            raise BackendUnavailable(
                self._load_error or f"backend '{self.name}' is not loaded"
            )
        started = time.perf_counter()
        result = self._infer(frame)
        result.backend = self.name
        elapsed = (time.perf_counter() - started) * 1000.0
        self._frames += 1
        self._total_ms += elapsed
        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def average_ms(self) -> float:
        return self._total_ms / self._frames if self._frames else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "loaded": self._loaded,
            "frames": self._frames,
            "average_ms": round(self.average_ms, 2),
            **self._detail,
        }
