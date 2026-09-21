"""ONNX Runtime detection backend — the default production path.

Execution providers are tried in descending order of expected speed and the
one actually granted by the runtime is reported in diagnostics. On Apple
Silicon CoreML delegates to the GPU/Neural Engine; on an NVIDIA host CUDA or
TensorRT is picked up automatically if those provider builds are installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.config import settings
from backend.inference.base import BackendUnavailable, Detector, InferenceResult
from backend.inference.postprocess import decode_and_filter, load_labels
from backend.inference.preprocess import letterbox, to_nchw
from backend.logging_conf import get_logger

logger = get_logger(__name__)

#: Preference order. Unavailable providers are skipped silently by ORT.
PROVIDER_PREFERENCE: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
)


class OnnxDetector(Detector):
    """YOLO-family detector executed by ONNX Runtime."""

    name = "onnx"
    description = "ONNX Runtime (hardware execution providers)"

    def __init__(
        self, model_path=None, labels_path=None,
        conf_threshold=None, iou_threshold=None,
    ) -> None:
        super().__init__(conf_threshold, iou_threshold)
        self.model_path = model_path or settings.model_file
        self.labels_path = labels_path or settings.model_labels_file
        self.session: Any = None
        self.labels: list[str] = []
        self._input_name = ""
        self._input_size = (settings.inference_width, settings.inference_height)

    # ── probe ────────────────────────────────────────────────────────────
    @classmethod
    def probe(cls, model_path=None) -> tuple[bool, str, dict[str, Any]]:
        try:
            import onnxruntime as ort
        except ImportError:
            return False, "onnxruntime is not installed", {}

        detail: dict[str, Any] = {
            "onnxruntime_version": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
        }
        model = model_path or settings.model_file
        if not model.exists():
            return (
                False,
                f"model file not found: {model} — run scripts/fetch_models.py",
                detail,
            )
        detail["model"] = model.name
        detail["model_size_mb"] = round(model.stat().st_size / 1e6, 1)
        return True, "", detail

    # ── lifecycle ────────────────────────────────────────────────────────
    def load(self) -> None:
        available, reason, detail = self.probe(self.model_path)
        if not available:
            self._load_error = reason
            raise BackendUnavailable(reason)

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.inference_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT logs a CoreML partitioning notice per session at level 2.
        options.log_severity_level = 3

        installed = set(ort.get_available_providers())
        providers = [p for p in PROVIDER_PREFERENCE if p in installed]
        if not providers:
            providers = ["CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(
                str(self.model_path), options, providers=providers
            )
        except Exception as exc:
            # A corrupt or opset-incompatible model lands here.
            self._load_error = f"failed to create ONNX session: {exc}"
            raise BackendUnavailable(self._load_error) from exc

        spec = self.session.get_inputs()[0]
        self._input_name = spec.name
        # Static shapes in the export win over the configured size.
        shape = spec.shape
        if isinstance(shape[3], int) and isinstance(shape[2], int):
            self._input_size = (int(shape[3]), int(shape[2]))

        self.labels = load_labels(self.labels_path, fallback=["person"])
        active = self.session.get_providers()

        self._detail = {
            **detail,
            "active_providers": active,
            "input_name": self._input_name,
            "input_size": list(self._input_size),
            "classes": len(self.labels),
            "accelerated": active[0] != "CPUExecutionProvider",
        }
        self._loaded = True
        logger.info(
            "ONNX detector ready: %s @ %dx%d via %s",
            self.model_path.name, self._input_size[0], self._input_size[1],
            active[0], extra={"backend": self.name},
        )

    def close(self) -> None:
        self.session = None
        super().close()

    # ── inference ────────────────────────────────────────────────────────
    def _infer(self, frame: np.ndarray) -> InferenceResult:
        import time

        t0 = time.perf_counter()
        canvas, transform = letterbox(frame, *self._input_size)
        tensor = to_nchw(canvas)
        t1 = time.perf_counter()

        outputs = self.session.run(None, {self._input_name: tensor})
        t2 = time.perf_counter()

        detections = decode_and_filter(
            outputs[0],
            transform,
            self.labels,
            self.conf_threshold,
            self.iou_threshold,
        )
        t3 = time.perf_counter()

        return InferenceResult(
            detections=detections,
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
            postprocess_ms=(t3 - t2) * 1000,
        )
