"""Inference abstraction, Mojo kernels, and pre/post-processing."""

from __future__ import annotations

import numpy as np
import pytest

from backend.inference.base import BackendUnavailable, Detection, Detector
from backend.inference.mojo.bridge import (
    KERNEL_NAMES,
    get_bridge,
    np_bgr_to_chw,
    np_decode_yolov8,
    np_iou_matrix,
    np_nms,
    np_points_in_polygon,
)
from backend.inference.postprocess import decode_and_filter, load_labels
from backend.inference.preprocess import (
    LetterboxTransform,
    downscale_to_width,
    letterbox,
    scale_boxes,
    to_nchw,
)
from backend.inference.registry import REGISTRY, describe_backends


# ══════════════════════════════════════════════════════════════════════════
#  Detection value object
# ══════════════════════════════════════════════════════════════════════════
class TestDetection:
    def test_geometry(self):
        det = Detection(bbox=(10, 20, 110, 220), confidence=0.9, class_id=0, label="person")
        assert det.width == 100 and det.height == 200
        assert det.area == 20_000
        assert det.centroid == (60, 120)
        # The foot point is what zone membership uses.
        assert det.foot_point == (60, 220)

    def test_degenerate_box_is_not_negative(self):
        det = Detection(bbox=(50, 50, 10, 10), confidence=0.5, class_id=0, label="person")
        assert det.width == 0 and det.height == 0


# ══════════════════════════════════════════════════════════════════════════
#  Backends
# ══════════════════════════════════════════════════════════════════════════
class TestBackends:
    def test_every_backend_probes_without_raising(self):
        """probe() is documented never to raise — diagnostics depends on it."""
        for name, cls in REGISTRY.items():
            available, reason, detail = cls.probe()
            assert isinstance(available, bool), name
            assert isinstance(reason, str), name
            assert isinstance(detail, dict), name
            if not available:
                assert reason, f"{name} was unavailable without a reason"

    def test_describe_reports_at_most_one_active(self):
        active = [b for b in describe_backends() if b.active]
        assert len(active) <= 1

    def test_max_backend_reports_truthfully(self):
        """Spec §24: never claim MAX is running when it is not."""
        from backend.inference.backends.max_backend import MaxDetector

        available, reason, _ = MaxDetector.probe()
        if not available:
            assert reason, "MAX must explain why it is unavailable"

    def test_unloaded_detector_refuses_to_infer(self, blank_frame):
        from backend.inference.backends.onnx_backend import OnnxDetector

        with pytest.raises(BackendUnavailable):
            OnnxDetector().infer(blank_frame)

    def test_mock_backend_is_deterministic(self, blank_frame):
        from backend.inference.backends.mock_backend import MockDetector

        first, second = MockDetector(people=3), MockDetector(people=3)
        first.load()
        second.load()
        for _ in range(5):
            a = first.infer(blank_frame).detections
            b = second.infer(blank_frame).detections
            assert [d.bbox for d in a] == [d.bbox for d in b]

    def test_mock_boxes_stay_inside_frame(self, blank_frame):
        from backend.inference.backends.mock_backend import MockDetector

        detector = MockDetector(people=5)
        detector.load()
        height, width = blank_frame.shape[:2]
        for _ in range(60):
            for det in detector.infer(blank_frame).detections:
                x1, y1, x2, y2 = det.bbox
                assert 0 <= x1 < x2 <= width
                assert 0 <= y1 < y2 <= height

    def test_detector_tracks_latency(self, mock_detector, blank_frame):
        for _ in range(4):
            mock_detector.infer(blank_frame)
        stats = mock_detector.stats()
        assert stats["frames"] == 4
        assert stats["loaded"] is True

    def test_interface_is_complete(self):
        for name, cls in REGISTRY.items():
            assert issubclass(cls, Detector), name
            for method in ("probe", "load", "_infer", "infer", "close"):
                assert hasattr(cls, method), f"{name} missing {method}"


# ══════════════════════════════════════════════════════════════════════════
#  Pre/post-processing
# ══════════════════════════════════════════════════════════════════════════
class TestPreprocess:
    @pytest.mark.parametrize(
        "shape", [(480, 854), (720, 1280), (1080, 1920), (540, 540), (200, 1000)]
    )
    def test_letterbox_preserves_aspect(self, shape):
        frame = np.random.randint(0, 256, (*shape, 3), dtype=np.uint8)
        canvas, tf = letterbox(frame, 640, 640)
        assert canvas.shape == (640, 640, 3)
        assert tf.scale == pytest.approx(min(640 / shape[1], 640 / shape[0]))

    @pytest.mark.parametrize("shape", [(480, 854), (1080, 1920), (200, 1000)])
    def test_boxes_round_trip_exactly(self, shape):
        """Letterboxing that does not invert exactly shifts every detection."""
        frame = np.zeros((*shape, 3), np.uint8)
        _, tf = letterbox(frame, 640, 640)
        full = np.array(
            [[tf.pad_x, tf.pad_y,
              tf.pad_x + shape[1] * tf.scale, tf.pad_y + shape[0] * tf.scale]],
            dtype=np.float32,
        )
        recovered = scale_boxes(full, tf)[0]
        assert recovered == pytest.approx([0, 0, shape[1], shape[0]], abs=1.0)

    def test_boxes_clipped_to_frame(self):
        tf = LetterboxTransform(scale=1.0, pad_x=0, pad_y=0, source_width=100, source_height=100)
        clipped = scale_boxes(np.array([[-50, -50, 500, 500]], np.float32), tf)[0]
        assert clipped.tolist() == [0, 0, 100, 100]

    def test_nchw_shape_and_range(self):
        frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        tensor = to_nchw(frame)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    def test_channel_order_is_rgb(self):
        """YOLO expects RGB; feeding BGR silently degrades accuracy."""
        frame = np.zeros((4, 4, 3), np.uint8)
        frame[:, :, 2] = 255  # pure red in BGR
        tensor = np_bgr_to_chw(frame)
        assert tensor[0, 0].mean() == pytest.approx(1.0)  # R plane
        assert tensor[0, 2].mean() == pytest.approx(0.0)  # B plane

    def test_downscale_only_shrinks(self):
        small = np.zeros((100, 200, 3), np.uint8)
        assert downscale_to_width(small, 400).shape == small.shape
        assert downscale_to_width(np.zeros((540, 1920, 3), np.uint8), 960).shape[1] == 960


class TestPostprocess:
    def test_decode_and_nms(self):
        rng = np.random.default_rng(5)
        pred = (rng.random((84, 500), dtype=np.float32) * 0.2)
        # Two confident, well-separated people.
        for anchor, cx in ((10, 100.0), (300, 500.0)):
            pred[0, anchor] = cx
            pred[1, anchor] = 300.0
            pred[2, anchor] = 60.0
            pred[3, anchor] = 160.0
            pred[4, anchor] = 0.95
        tf = LetterboxTransform(1.0, 0, 0, 640, 640)
        detections = decode_and_filter(pred, tf, ["person"], 0.5, 0.45)
        assert len(detections) == 2
        assert all(d.label == "person" for d in detections)

    def test_empty_prediction_is_empty(self):
        pred = np.zeros((84, 100), np.float32)
        tf = LetterboxTransform(1.0, 0, 0, 640, 640)
        assert decode_and_filter(pred, tf, ["person"], 0.5, 0.45) == []

    def test_class_filter(self):
        labels = ["person", "bicycle", "car"]
        pred = np.zeros((4 + len(labels), 300), np.float32)
        pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0] = 100, 100, 50, 100
        pred[6, 0] = 0.9  # class index 2 -> "car"
        tf = LetterboxTransform(1.0, 0, 0, 640, 640)
        assert decode_and_filter(pred, tf, labels, 0.5, 0.45)[0].label == "car"
        assert decode_and_filter(pred, tf, labels, 0.5, 0.45, keep_classes={0}) == []

    def test_head_orientation_resolved_from_label_count(self):
        """Both export layouts must decode identically.

        Regression: the previous rule was "transpose when rows > columns",
        which mis-decodes any head with fewer anchors than attributes.
        """
        labels = ["person"]
        attrs = 4 + len(labels)
        canonical = np.zeros((attrs, 3), np.float32)   # (attrs, anchors)
        canonical[0, 0], canonical[1, 0] = 100, 100
        canonical[2, 0], canonical[3, 0] = 50, 120
        canonical[4, 0] = 0.9
        tf = LetterboxTransform(1.0, 0, 0, 640, 640)

        from_canonical = decode_and_filter(canonical, tf, labels, 0.5, 0.45)
        from_transposed = decode_and_filter(canonical.T, tf, labels, 0.5, 0.45)
        assert len(from_canonical) == 1
        assert [d.bbox for d in from_canonical] == [d.bbox for d in from_transposed]

    def test_labels_missing_file_uses_fallback(self, tmp_path):
        assert load_labels(tmp_path / "absent.names", fallback=["person"]) == ["person"]

    def test_labels_read(self, tmp_path):
        path = tmp_path / "l.names"
        path.write_text("person\nhelmet\n\nvest\n")
        assert load_labels(path) == ["person", "helmet", "vest"]


# ══════════════════════════════════════════════════════════════════════════
#  Mojo bridge — the accelerated path must agree with numpy exactly
# ══════════════════════════════════════════════════════════════════════════
class TestMojoBridge:
    def test_bridge_reports_coherent_state(self):
        info = get_bridge().info()
        assert set(info["kernels"]) == set(KERNEL_NAMES)
        # Never active without a loaded library.
        assert not (info["active"] and not info["available"])
        if not info["available"]:
            assert info["reason"], "an unavailable bridge must explain itself"

    def test_calibration_verified_correctness(self):
        """Any enabled kernel must have passed its numpy comparison."""
        bridge = get_bridge()
        for name, enabled in bridge.enabled.items():
            if enabled:
                assert bridge.benchmarks[name]["correct"] == 1.0, name

    def test_dispatched_results_match_numpy(self):
        """Whichever path is active, the answers must be identical."""
        bridge = get_bridge()
        rng = np.random.default_rng(11)

        frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        assert np.allclose(bridge.bgr_to_chw(frame), np_bgr_to_chw(frame), atol=1e-6)

        a = rng.random((12, 4), dtype=np.float32) * 100
        b = rng.random((9, 4), dtype=np.float32) * 100
        a[:, 2:] += a[:, :2]
        b[:, 2:] += b[:, :2]
        assert np.allclose(bridge.iou_matrix(a, b), np_iou_matrix(a, b), atol=1e-5)

        points = rng.random((40, 2), dtype=np.float32)
        polygon = np.array([[0.2, 0.2], [0.8, 0.25], [0.85, 0.8], [0.15, 0.9]], np.float32)
        assert np.array_equal(
            bridge.points_in_polygon(points, polygon),
            np_points_in_polygon(points, polygon),
        )

        boxes = rng.random((80, 4), dtype=np.float32) * 400
        boxes[:, 2:] += boxes[:, :2]
        scores = rng.random((80,), dtype=np.float32)
        assert np.array_equal(bridge.nms(boxes, scores, 0.5), np_nms(boxes, scores, 0.5))

        pred = rng.random((84, 600), dtype=np.float32) * 0.3
        pred[4:8, ::53] = 0.9
        gb, gs, _ = bridge.decode_yolov8(pred, 0.5)
        rb, rs, _ = np_decode_yolov8(pred, 0.5)
        assert gb.shape == rb.shape
        assert np.allclose(np.sort(gs), np.sort(rs), atol=1e-5)

    def test_empty_inputs_are_safe(self):
        bridge = get_bridge()
        empty4 = np.zeros((0, 4), np.float32)
        assert bridge.iou_matrix(empty4, empty4).shape == (0, 0)
        assert len(bridge.nms(empty4, np.zeros((0,), np.float32), 0.5)) == 0
        assert len(bridge.points_in_polygon(np.zeros((0, 2), np.float32),
                                           np.zeros((4, 2), np.float32))) == 0

    def test_degenerate_polygon_returns_all_false(self):
        points = np.array([[0.5, 0.5]], np.float32)
        assert not get_bridge().points_in_polygon(points, np.zeros((2, 2), np.float32)).any()

    def test_nms_suppresses_overlaps_keeps_distinct(self):
        boxes = np.array(
            [[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], np.float32
        )
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        assert len(np_nms(boxes, scores, 0.5)) == 2

    def test_iou_known_values(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        assert np_iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
        b = np.array([[20, 20, 30, 30]], np.float32)
        assert np_iou_matrix(a, b)[0, 0] == pytest.approx(0.0)
        half = np.array([[5, 0, 15, 10]], np.float32)
        assert np_iou_matrix(a, half)[0, 0] == pytest.approx(50 / 150)
