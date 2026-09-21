"""Kalman filter for bounding-box tracking.

State is the 8-vector ``[cx, cy, a, h, vcx, vcy, va, vh]`` — centre, aspect
ratio, height and their velocities — the SORT/ByteTrack parameterisation.
Tracking aspect+height rather than width+height keeps the motion model stable
when a detector's box jitters horizontally.

Uncertainty is scaled by object height, so a distant (small) person is
expected to move fewer pixels than a near one.
"""

from __future__ import annotations

import numpy as np


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over a single box."""

    # Process/measurement noise relative to object height.
    STD_WEIGHT_POSITION = 1.0 / 20
    STD_WEIGHT_VELOCITY = 1.0 / 160

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        # x' = F x  (position gains velocity * dt)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # z = H x  (we observe position only)
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

    # ── lifecycle ────────────────────────────────────────────────────────
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a state from a first observation ``[cx, cy, a, h]``."""
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement
        h = float(measurement[3])
        std = np.array(
            [
                2 * self.STD_WEIGHT_POSITION * h,
                2 * self.STD_WEIGHT_POSITION * h,
                1e-2,
                2 * self.STD_WEIGHT_POSITION * h,
                10 * self.STD_WEIGHT_VELOCITY * h,
                10 * self.STD_WEIGHT_VELOCITY * h,
                1e-5,
                10 * self.STD_WEIGHT_VELOCITY * h,
            ],
            dtype=np.float32,
        )
        return mean, np.diag(np.square(std))

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one time step."""
        h = float(mean[3])
        std = np.array(
            [
                self.STD_WEIGHT_POSITION * h,
                self.STD_WEIGHT_POSITION * h,
                1e-2,
                self.STD_WEIGHT_POSITION * h,
                self.STD_WEIGHT_VELOCITY * h,
                self.STD_WEIGHT_VELOCITY * h,
                1e-5,
                self.STD_WEIGHT_VELOCITY * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with a new observation."""
        h = float(mean[3])
        std = np.array(
            [
                self.STD_WEIGHT_POSITION * h,
                self.STD_WEIGHT_POSITION * h,
                1e-1,
                self.STD_WEIGHT_POSITION * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )

        # Solve rather than invert — cheaper and numerically better behaved.
        try:
            kalman_gain = np.linalg.solve(
                projected_cov.T, (covariance @ self._update_mat.T).T
            ).T
        except np.linalg.LinAlgError:
            # Degenerate covariance: skip the correction rather than crash the
            # whole pipeline; the next frame re-seeds it.
            return mean, covariance

        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean.astype(np.float32), new_cov.astype(np.float32)


# ── conversions ───────────────────────────────────────────────────────────
def xyxy_to_xyah(box: tuple[float, float, float, float]) -> np.ndarray:
    """``[x1, y1, x2, y2]`` → ``[cx, cy, aspect, height]``."""
    x1, y1, x2, y2 = box
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float32)


def xyah_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    """``[cx, cy, aspect, height]`` → ``[x1, y1, x2, y2]``."""
    cx, cy, a, h = state[:4]
    h = max(1e-3, float(h))
    w = max(1e-3, float(a) * h)
    return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))
