"""Frame preparation for the detector.

Letterboxing (resize preserving aspect ratio, pad the remainder) is what YOLO
expects; getting it wrong shifts every box. :func:`letterbox` returns the
transform alongside the image so :func:`scale_boxes` can map predictions back
to source-frame pixels exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.inference.mojo.bridge import get_bridge


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """The geometry applied to a frame, needed to invert it afterwards."""

    scale: float
    pad_x: float
    pad_y: float
    source_width: int
    source_height: int


def letterbox(
    frame: np.ndarray,
    width: int,
    height: int,
    fill: int = 114,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize `frame` into a `width` × `height` canvas, preserving aspect."""
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))

    # INTER_AREA is the right filter when shrinking; INTER_LINEAR when growing.
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    pad_x = (width - new_w) / 2.0
    pad_y = (height - new_h) / 2.0
    top, left = int(round(pad_y)), int(round(pad_x))
    canvas[top : top + new_h, left : left + new_w] = resized

    return canvas, LetterboxTransform(scale, left, top, src_w, src_h)


def to_nchw(frame: np.ndarray, scale: float = 1.0 / 255.0) -> np.ndarray:
    """BGR HWC uint8 → RGB NCHW float32. Mojo-accelerated where it wins."""
    return get_bridge().bgr_to_chw(frame, scale)


def scale_boxes(boxes: np.ndarray, tf: LetterboxTransform) -> np.ndarray:
    """Undo letterboxing and clip to the source frame."""
    if boxes.size == 0:
        return boxes
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] -= tf.pad_x
    out[:, [1, 3]] -= tf.pad_y
    out /= tf.scale
    # Clip via explicit assignment. `np.clip(out[:, [0, 2]], ..., out=...)`
    # looks equivalent but silently does nothing: fancy indexing returns a
    # copy, so the clipped values are written into a temporary and thrown
    # away, leaving edge detections with out-of-frame coordinates.
    out[:, 0] = out[:, 0].clip(0.0, tf.source_width)
    out[:, 2] = out[:, 2].clip(0.0, tf.source_width)
    out[:, 1] = out[:, 1].clip(0.0, tf.source_height)
    out[:, 3] = out[:, 3].clip(0.0, tf.source_height)
    return out


def downscale_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Cap decode-side resolution. Returns the input untouched if small."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(
        frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA
    )
