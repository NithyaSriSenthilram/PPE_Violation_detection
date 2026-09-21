"""Modular MAX inference backend.

Status on the current machine (verified, not assumed): the MAX runtime installs
and runs on Apple Silicon, but **MAX 26.5 cannot compile a generic ONNX graph** —
`InferenceSession.load()` rejects it with "cannot compile input with format
input has unknown contents". MAX now targets its own graph format plus
safetensors/GGUF checkpoints rather than acting as a drop-in ONNX runtime.

This backend is therefore written to *try for real* and to report the exact
truth: it probes the runtime, attempts the load, and on failure records the
engine's own error message, which surfaces verbatim in `GET /api/diagnostics`.
It activates automatically on any host/MAX version where the load succeeds
(or where `MODEL_PATH` points at a MAX-native artefact). It never claims to be
running when it is not — the registry falls back to ONNX Runtime and says so.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import numpy as np

from backend.config import settings
from backend.inference.base import BackendUnavailable, Detector, InferenceResult
from backend.inference.postprocess import decode_and_filter, load_labels
from backend.inference.preprocess import letterbox, to_nchw
from backend.logging_conf import get_logger

logger = get_logger(__name__)

#: Extensions MAX can consume natively, beyond a compiled MAX graph.
MAX_NATIVE_SUFFIXES = {".maxgraph", ".safetensors", ".gguf"}


class MaxDetector(Detector):
    """Detector executed by the MAX inference engine."""

    name = "max"
    description = "Modular MAX inference engine"

    def __init__(
        self, model_path=None, labels_path=None,
        conf_threshold=None, iou_threshold=None,
    ) -> None:
        super().__init__(conf_threshold, iou_threshold)
        self.model_path = model_path or settings.model_file
        self.labels_path = labels_path or settings.model_labels_file
        self.model: Any = None
        self.session: Any = None
        self.labels: list[str] = []
        self._input_size = (settings.inference_width, settings.inference_height)

    # ── probe ────────────────────────────────────────────────────────────
    @classmethod
    def probe(cls, model_path=None) -> tuple[bool, str, dict[str, Any]]:
        """Report MAX runtime availability and device inventory."""
        try:
            from max import driver, engine  # noqa: F401
        except ImportError:
            return (
                False,
                "MAX runtime is not installed (pip install modular)",
                {},
            )

        detail: dict[str, Any] = {}
        try:
            import importlib.metadata as md

            detail["max_version"] = md.version("max")
        except Exception:
            detail["max_version"] = "unknown"

        # Enumerate devices so diagnostics can show what MAX would run on.
        try:
            from max import driver

            count = int(driver.accelerator_count())
            detail["accelerator_count"] = count
            devices = ["cpu"]
            if count > 0:
                try:
                    devices.append(str(driver.Accelerator()))
                except Exception as exc:  # accelerator present but unusable
                    detail["accelerator_error"] = str(exc)[:200]
            detail["devices"] = devices
        except Exception as exc:
            detail["driver_error"] = str(exc)[:200]

        model = model_path or settings.model_file
        if not model.exists():
            return False, f"model file not found: {model}", detail
        detail["model"] = model.name
        detail["model_format"] = model.suffix

        if model.suffix not in MAX_NATIVE_SUFFIXES and model.suffix != ".onnx":
            return (
                False,
                f"MODEL_PATH format '{model.suffix}' is not a MAX-loadable "
                f"artefact (expected one of {sorted(MAX_NATIVE_SUFFIXES)} or .onnx)",
                detail,
            )

        # The runtime is present; whether it can compile this specific model is
        # only knowable by attempting the load, which load() does.
        return True, "", detail

    # ── lifecycle ────────────────────────────────────────────────────────
    def load(self) -> None:
        available, reason, detail = self.probe(self.model_path)
        self._detail = dict(detail)
        if not available:
            self._load_error = reason
            raise BackendUnavailable(reason)

        from max import driver, engine

        devices = [driver.CPU()]
        # CPU is always a valid device; accelerator selection is best-effort.
        with contextlib.suppress(Exception):
            if int(driver.accelerator_count()) > 0:
                devices = [driver.Accelerator()]

        try:
            self.session = engine.InferenceSession(devices=devices)
            self.model = self.session.load(str(self.model_path))
        except Exception as exc:
            # This is the expected outcome for ONNX on MAX 26.5. Record the
            # engine's own words rather than paraphrasing them.
            self._load_error = (
                f"MAX runtime v{detail.get('max_version', '?')} could not compile "
                f"{self.model_path.name}: {exc}"
            )
            self._detail["load_error"] = str(exc)[:300]
            raise BackendUnavailable(self._load_error) from exc

        self.labels = load_labels(self.labels_path, fallback=["person"])
        with contextlib.suppress(Exception):
            self._detail["input_metadata"] = str(self.model.input_metadata)[:300]
        self._detail["devices_used"] = [str(d) for d in devices]
        self._detail["classes"] = len(self.labels)
        self._loaded = True
        logger.info(
            "MAX detector ready: %s on %s",
            self.model_path.name, self._detail["devices_used"],
            extra={"backend": self.name},
        )

    def close(self) -> None:
        self.model = None
        self.session = None
        super().close()

    # ── inference ────────────────────────────────────────────────────────
    def _infer(self, frame: np.ndarray) -> InferenceResult:
        t0 = time.perf_counter()
        canvas, transform = letterbox(frame, *self._input_size)
        tensor = to_nchw(canvas)
        t1 = time.perf_counter()

        outputs = self.model.execute(tensor)
        t2 = time.perf_counter()

        raw = outputs[0]
        # MAX returns its own tensor type; normalise to numpy.
        array = raw if isinstance(raw, np.ndarray) else np.asarray(
            raw.to_numpy() if hasattr(raw, "to_numpy") else raw
        )
        detections = decode_and_filter(
            array,
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
