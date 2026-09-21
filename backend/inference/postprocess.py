"""Detection post-processing: decode, NMS, class filtering."""

from __future__ import annotations

import numpy as np

from backend.inference.base import Detection
from backend.inference.mojo.bridge import get_bridge
from backend.inference.preprocess import LetterboxTransform, scale_boxes


def decode_and_filter(
    raw: np.ndarray,
    transform: LetterboxTransform,
    labels: list[str],
    conf_threshold: float,
    iou_threshold: float,
    keep_classes: set[int] | None = None,
) -> list[Detection]:
    """Turn a YOLOv8 head tensor into source-space :class:`Detection` objects.

    `raw` may arrive as ``(1, attrs, anchors)`` or ``(attrs, anchors)``.
    """
    bridge = get_bridge()

    pred = raw[0] if raw.ndim == 3 else raw
    pred = _orient_head(pred, len(labels))

    boxes, scores, class_ids = bridge.decode_yolov8(
        np.ascontiguousarray(pred, dtype=np.float32), conf_threshold
    )
    if len(boxes) == 0:
        return []

    if keep_classes is not None:
        mask = np.isin(class_ids, list(keep_classes))
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        if len(boxes) == 0:
            return []

    # NMS per class — a person overlapping a car must not suppress it.
    kept: list[int] = []
    for cid in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == cid)
        local = bridge.nms(boxes[idx], scores[idx], iou_threshold)
        kept.extend(idx[local].tolist())

    if not kept:
        return []

    order = np.array(kept, dtype=np.int32)
    final_boxes = scale_boxes(boxes[order], transform)

    detections: list[Detection] = []
    for box, score, cid in zip(final_boxes, scores[order], class_ids[order], strict=False):
        # Drop degenerate boxes that survive clipping at a frame edge.
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        index = int(cid)
        detections.append(
            Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(score),
                class_id=index,
                label=labels[index] if 0 <= index < len(labels) else str(index),
            )
        )
    return detections


def _orient_head(pred: np.ndarray, label_count: int) -> np.ndarray:
    """Return the head as ``(attrs, anchors)``.

    Exports differ: some emit ``(attrs, anchors)``, others the transpose. The
    attribute count is knowable — 4 box values plus one score per class — so
    match against that first and only fall back to the "anchors outnumber
    attributes" shape heuristic when the label count does not identify either
    axis. Relying on the shape heuristic alone silently mis-decodes any head
    with fewer anchors than attributes.
    """
    expected = 4 + label_count
    if label_count > 0:
        if pred.shape[0] == expected:
            return pred
        if pred.shape[1] == expected:
            return pred.T
    return pred.T if pred.shape[0] > pred.shape[1] else pred


def load_labels(path, fallback: list[str] | None = None) -> list[str]:
    """Read a newline-delimited label file, tolerating a missing file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return list(fallback or [])
    return [line.strip() for line in text.splitlines() if line.strip()]
