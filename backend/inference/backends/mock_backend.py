"""Deterministic synthetic detector.

Two jobs:

* **Tests** — exercise tracking, zones, behaviour analysis, the event engine
  and the API without a model file or any inference dependency.
* **Degraded operation** — if no real backend can load, the platform still
  starts, serves the UI and reports `backend: mock` in diagnostics instead of
  crashing. It is always labelled as synthetic so it can never be mistaken for
  real output.

Motion is a smooth deterministic function of the frame counter, so a test can
assert on stable track identities and on speeds it computed itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.inference.base import Detection, Detector, InferenceResult


class MockDetector(Detector):
    """Generates repeatable synthetic person detections."""

    name = "mock"
    description = "Synthetic detections (no model — testing / degraded mode)"

    def __init__(self, people: int = 3, seed: int = 0) -> None:
        super().__init__()
        self.people = people
        self.seed = seed
        self._frame = 0

    @classmethod
    def probe(cls, model_path=None) -> tuple[bool, str, dict[str, Any]]:
        return True, "", {"synthetic": True}

    def load(self) -> None:
        self._loaded = True
        self._detail = {"synthetic": True, "people": self.people}

    def reset(self) -> None:
        """Restart the motion sequence (tests)."""
        self._frame = 0

    def _infer(self, frame: np.ndarray) -> InferenceResult:
        h, w = frame.shape[:2]
        t = self._frame
        self._frame += 1

        detections: list[Detection] = []
        for i in range(self.people):
            phase = self.seed + i * 1.7
            # Horizontal sweep with a per-person period; vertical bob.
            fx = 0.5 + 0.35 * math.sin(t * 0.03 + phase)
            fy = 0.55 + 0.08 * math.cos(t * 0.05 + phase)
            box_h = h * 0.34
            box_w = box_h * 0.42
            cx, cy = fx * w, fy * h
            detections.append(
                Detection(
                    bbox=(
                        max(0.0, cx - box_w / 2),
                        max(0.0, cy - box_h / 2),
                        min(float(w), cx + box_w / 2),
                        min(float(h), cy + box_h / 2),
                    ),
                    confidence=0.80 + 0.08 * math.sin(t * 0.1 + i),
                    class_id=0,
                    label="person",
                )
            )
        return InferenceResult(detections=detections, inference_ms=0.5)
