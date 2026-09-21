"""Ultralytics/PyTorch detection backend (optional).

Useful as a cross-check against the ONNX path and on hosts where a `.pt`
checkpoint is all that is available. Prefers Apple MPS, then CUDA, then CPU.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from backend.config import settings
from backend.inference.base import (
    BackendUnavailable,
    Detection,
    Detector,
    InferenceResult,
)
from backend.logging_conf import get_logger

logger = get_logger(__name__)


def _pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UltralyticsDetector(Detector):
    """Detector executed by Ultralytics YOLO."""

    name = "ultralytics"
    description = "Ultralytics YOLO (PyTorch)"

    def __init__(
        self, model_path=None, labels_path=None,
        conf_threshold=None, iou_threshold=None,
    ) -> None:
        super().__init__(conf_threshold, iou_threshold)
        self.labels_path = labels_path or settings.model_labels_file
        # Prefer a .pt sibling of the configured ONNX model; ultralytics can
        # also run the .onnx directly if that is all we have.
        configured = model_path or settings.model_file
        candidate = configured.with_suffix(".pt")
        self.model_path = candidate if candidate.exists() else configured
        self.model: Any = None
        self.device = "cpu"

    @classmethod
    def probe(cls, model_path=None) -> tuple[bool, str, dict[str, Any]]:
        try:
            import ultralytics
        except ImportError:
            return False, "ultralytics is not installed", {}

        detail: dict[str, Any] = {
            "ultralytics_version": ultralytics.__version__,
            "device": _pick_device(),
        }
        try:
            import torch

            detail["torch_version"] = torch.__version__
        except ImportError:
            return False, "torch is not installed", detail

        configured = model_path or settings.model_file
        pt = configured.with_suffix(".pt")
        if not pt.exists() and not configured.exists():
            return False, f"no weights found at {pt} or {configured}", detail
        detail["model"] = (pt if pt.exists() else configured).name
        return True, "", detail

    def load(self) -> None:
        available, reason, detail = self.probe(self.model_path)
        if not available:
            self._load_error = reason
            raise BackendUnavailable(reason)

        from ultralytics import YOLO

        try:
            self.model = YOLO(str(self.model_path))
            self.device = _pick_device()
        except Exception as exc:
            self._load_error = f"failed to load {self.model_path.name}: {exc}"
            raise BackendUnavailable(self._load_error) from exc

        self._detail = {**detail, "device": self.device}
        self._loaded = True
        logger.info(
            "Ultralytics detector ready: %s on %s",
            self.model_path.name, self.device, extra={"backend": self.name},
        )

    def close(self) -> None:
        self.model = None
        super().close()

    def _infer(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=settings.inference_width,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        detections: list[Detection] = []
        names = getattr(self.model, "names", {}) or {}
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, conf, cid in zip(xyxy, confs, classes, strict=False):
                detections.append(
                    Detection(
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=float(conf),
                        class_id=int(cid),
                        label=str(names.get(int(cid), cid)),
                    )
                )
        # Ultralytics fuses pre/post-processing into predict(); report it as
        # inference time rather than inventing a split.
        return InferenceResult(detections=detections, inference_ms=elapsed)
