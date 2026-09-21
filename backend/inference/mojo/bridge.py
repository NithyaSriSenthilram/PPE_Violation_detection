"""Python ↔ Mojo boundary.

Every accelerated kernel is exposed here as a plain Python function with a
numpy implementation of identical semantics. At import time the bridge:

1. tries to ``dlopen`` the compiled Mojo library and verify its ABI version,
2. checks each kernel against its numpy twin for numerical agreement,
3. benchmarks both and enables the Mojo path **only where it is faster**.

That last step matters: numpy delegates to BLAS/SIMD C for some of these
shapes and genuinely wins. Shipping a "Mojo accelerated" call that is slower
than the fallback would be a lie told in code, so the bridge measures instead
of assuming. Results are reported by ``GET /api/diagnostics``.

Nothing outside this module knows Mojo exists.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np

from backend.config import settings
from backend.logging_conf import get_logger

logger = get_logger(__name__)

#: Must match `sv_abi_version` in kernels.mojo. Bump when a signature changes.
REQUIRED_ABI: Final[int] = 1

_F32P = ctypes.POINTER(ctypes.c_float)
_U8P = ctypes.POINTER(ctypes.c_uint8)
_I32P = ctypes.POINTER(ctypes.c_int32)

KERNEL_NAMES: Final[tuple[str, ...]] = (
    "bgr_to_chw",
    "decode_yolov8",
    "iou_matrix",
    "points_in_polygon",
    "nms",
)


def _f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr_f32(a: np.ndarray) -> Any:
    return a.ctypes.data_as(_F32P)


def _ptr_u8(a: np.ndarray) -> Any:
    return a.ctypes.data_as(_U8P)


def _ptr_i32(a: np.ndarray) -> Any:
    return a.ctypes.data_as(_I32P)


# ══════════════════════════════════════════════════════════════════════════
#  numpy reference implementations — always correct, always available
# ══════════════════════════════════════════════════════════════════════════
def np_bgr_to_chw(frame: np.ndarray, scale: float = 1.0 / 255.0) -> np.ndarray:
    """BGR HWC uint8 → RGB CHW float32, scaled, with a batch axis."""
    rgb = frame[:, :, ::-1]
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    chw *= np.float32(scale)
    return chw[None]


def np_decode_yolov8(
    pred: np.ndarray, conf_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a [num_attrs, num_anchors] YOLOv8 head into xyxy/score/class."""
    class_scores = pred[4:]
    best = class_scores.max(axis=0)
    keep = best >= conf_thr
    if not keep.any():
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int32),
        )
    ids = class_scores[:, keep].argmax(axis=0).astype(np.int32)
    cx, cy, w, h = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
    boxes = np.stack(
        [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], axis=1
    ).astype(np.float32)
    return boxes, best[keep].astype(np.float32), ids


def np_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes → (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def np_points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for many points against one polygon."""
    if len(points) == 0 or len(polygon) < 3:
        return np.zeros((len(points),), dtype=bool)
    px = points[:, 0].astype(np.float32)
    py = points[:, 1].astype(np.float32)
    inside = np.zeros(len(points), dtype=bool)
    xs = polygon[:, 0].astype(np.float32)
    ys = polygon[:, 1].astype(np.float32)
    n = len(polygon)
    j = n - 1
    for i in range(n):
        yi, yj = ys[i], ys[j]
        straddles = (yi > py) != (yj > py)
        if straddles.any():
            denom = yj - yi
            if denom != 0:
                x_cross = (xs[j] - xs[i]) * (py - yi) / denom + xs[i]
                inside ^= straddles & (px < x_cross)
        j = i
    return inside


def np_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return np.zeros((0,), np.int32)
    order = np.argsort(-scores).astype(np.int32)
    keep: list[int] = []
    boxes = boxes.astype(np.float32, copy=False)
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np_iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thr]
    return np.asarray(keep, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════
#  Mojo-backed implementations
# ══════════════════════════════════════════════════════════════════════════
class _MojoLibrary:
    """Thin ctypes wrapper around the compiled kernels."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self._bind()
        version = int(self.lib.sv_abi_version())
        if version != REQUIRED_ABI:
            raise OSError(
                f"Mojo kernel ABI mismatch: library reports v{version}, "
                f"this build requires v{REQUIRED_ABI} — rebuild with "
                "./backend/inference/mojo/build.sh"
            )
        self.abi_version = version

    def _bind(self) -> None:
        lib = self.lib
        lib.sv_abi_version.argtypes = []
        lib.sv_abi_version.restype = ctypes.c_int32

        lib.sv_bgr_hwc_to_rgb_chw.argtypes = [
            _U8P, _F32P, ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
        ]
        lib.sv_bgr_hwc_to_rgb_chw.restype = ctypes.c_int32

        lib.sv_decode_yolov8.argtypes = [
            _F32P, ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
            _F32P, _F32P, _I32P, ctypes.c_int32,
        ]
        lib.sv_decode_yolov8.restype = ctypes.c_int32

        lib.sv_iou_matrix.argtypes = [
            _F32P, ctypes.c_int32, _F32P, ctypes.c_int32, _F32P,
        ]
        lib.sv_iou_matrix.restype = ctypes.c_int32

        lib.sv_points_in_polygon.argtypes = [
            _F32P, ctypes.c_int32, _F32P, ctypes.c_int32, _I32P,
        ]
        lib.sv_points_in_polygon.restype = ctypes.c_int32

        lib.sv_nms.argtypes = [
            _F32P, _I32P, ctypes.c_int32, ctypes.c_float, _I32P,
        ]
        lib.sv_nms.restype = ctypes.c_int32

    # ── kernels ──────────────────────────────────────────────────────────
    def bgr_to_chw(self, frame: np.ndarray, scale: float = 1.0 / 255.0) -> np.ndarray:
        h, w = frame.shape[:2]
        src = np.ascontiguousarray(frame, dtype=np.uint8)
        dst = np.empty((1, 3, h, w), dtype=np.float32)
        self.lib.sv_bgr_hwc_to_rgb_chw(_ptr_u8(src), _ptr_f32(dst), h, w, scale)
        return dst

    def decode_yolov8(
        self, pred: np.ndarray, conf_thr: float, max_out: int = 4096
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = _f32(pred)
        attrs, anchors = p.shape
        boxes = np.empty((max_out, 4), np.float32)
        scores = np.empty((max_out,), np.float32)
        classes = np.empty((max_out,), np.int32)
        n = self.lib.sv_decode_yolov8(
            _ptr_f32(p), attrs, anchors, conf_thr,
            _ptr_f32(boxes), _ptr_f32(scores), _ptr_i32(classes), max_out,
        )
        if n < 0:
            # Output budget exhausted — fall back rather than truncate, so a
            # pathological frame cannot silently drop detections.
            return np_decode_yolov8(pred, conf_thr)
        return boxes[:n].copy(), scores[:n].copy(), classes[:n].copy()

    def iou_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), np.float32)
        aa, bb = _f32(a), _f32(b)
        dst = np.empty((len(a), len(b)), np.float32)
        self.lib.sv_iou_matrix(_ptr_f32(aa), len(a), _ptr_f32(bb), len(b), _ptr_f32(dst))
        return dst

    def points_in_polygon(self, points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        if len(points) == 0 or len(polygon) < 3:
            return np.zeros((len(points),), dtype=bool)
        pts, poly = _f32(points), _f32(polygon)
        dst = np.empty((len(points),), np.int32)
        self.lib.sv_points_in_polygon(
            _ptr_f32(pts), len(points), _ptr_f32(poly), len(polygon), _ptr_i32(dst)
        )
        return dst.astype(bool)

    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0,), np.int32)
        bx = _f32(boxes)
        order = np.ascontiguousarray(np.argsort(-scores), dtype=np.int32)
        keep = np.empty((len(boxes),), np.int32)
        n = self.lib.sv_nms(
            _ptr_f32(bx), _ptr_i32(order), len(boxes), iou_thr, _ptr_i32(keep)
        )
        return keep[:n].copy()


# ══════════════════════════════════════════════════════════════════════════
#  Dispatcher
# ══════════════════════════════════════════════════════════════════════════
class KernelBridge:
    """Chooses, per kernel, between Mojo and numpy based on measurement."""

    def __init__(self) -> None:
        self.library: _MojoLibrary | None = None
        self.load_error: str = ""
        self.enabled: dict[str, bool] = {k: False for k in KERNEL_NAMES}
        self.benchmarks: dict[str, dict[str, float]] = {}
        self.mode: str = settings.mojo_enabled
        self._initialise()

    # ── setup ────────────────────────────────────────────────────────────
    def _initialise(self) -> None:
        if self.mode == "off":
            self.load_error = "disabled by configuration (MOJO_ENABLED=off)"
            return

        path = settings.mojo_lib_file
        if not path.exists():
            self.load_error = (
                f"kernel library not built at {path.name} — run "
                "./backend/inference/mojo/build.sh (numpy fallbacks in use)"
            )
            logger.info("Mojo kernels not built; using numpy fallbacks")
            return

        try:
            self.library = _MojoLibrary(path)
        except OSError as exc:
            # Typically a missing Mojo runtime dylib on the baked rpath.
            self.load_error = f"could not load {path.name}: {exc}"
            logger.warning("Mojo kernel library failed to load: %s", exc)
            return

        self._calibrate()

    def _calibrate(self) -> None:
        """Verify correctness, then keep only the kernels that are faster."""
        assert self.library is not None
        force = self.mode == "on"
        for name, check in _CALIBRATIONS.items():
            try:
                correct, mojo_ms, numpy_ms = check(self.library)
            except Exception as exc:  # a broken kernel must never break startup
                logger.warning("Mojo kernel %s failed calibration: %s", name, exc)
                self.benchmarks[name] = {"error": 1.0}
                continue

            speedup = (numpy_ms / mojo_ms) if mojo_ms > 0 else 0.0
            self.benchmarks[name] = {
                "mojo_ms": round(mojo_ms, 4),
                "numpy_ms": round(numpy_ms, 4),
                "speedup": round(speedup, 2),
                "correct": float(correct),
            }
            if not correct:
                logger.warning(
                    "Mojo kernel %s disagreed with the numpy reference; "
                    "using numpy", name,
                )
                continue
            # 1.05 keeps us from flapping on measurement noise.
            self.enabled[name] = force or speedup > 1.05

        active = [k for k, v in self.enabled.items() if v]
        if active:
            logger.info(
                "Mojo kernels active: %s (ABI v%d)",
                ", ".join(active), self.library.abi_version,
                extra={"backend": "mojo"},
            )
        else:
            logger.info(
                "Mojo library loaded but numpy was faster for every kernel; "
                "using numpy", extra={"backend": "mojo"},
            )

    # ── status ───────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self.library is not None

    @property
    def active(self) -> bool:
        return any(self.enabled.values())

    def info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "available": self.available,
            "active": self.active,
            "library": str(settings.mojo_lib_file) if self.available else None,
            "abi_version": self.library.abi_version if self.library else None,
            "reason": self.load_error,
            "kernels": {
                name: {
                    "active": self.enabled[name],
                    "implementation": "mojo" if self.enabled[name] else "numpy",
                    **self.benchmarks.get(name, {}),
                }
                for name in KERNEL_NAMES
            },
        }

    # ── dispatched kernels ───────────────────────────────────────────────
    def bgr_to_chw(self, frame: np.ndarray, scale: float = 1.0 / 255.0) -> np.ndarray:
        if self.enabled["bgr_to_chw"] and self.library:
            return self.library.bgr_to_chw(frame, scale)
        return np_bgr_to_chw(frame, scale)

    def decode_yolov8(
        self, pred: np.ndarray, conf_thr: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.enabled["decode_yolov8"] and self.library:
            return self.library.decode_yolov8(pred, conf_thr)
        return np_decode_yolov8(pred, conf_thr)

    def iou_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.enabled["iou_matrix"] and self.library:
            return self.library.iou_matrix(a, b)
        return np_iou_matrix(a, b)

    def points_in_polygon(self, points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        if self.enabled["points_in_polygon"] and self.library:
            return self.library.points_in_polygon(points, polygon)
        return np_points_in_polygon(points, polygon)

    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
        if self.enabled["nms"] and self.library:
            return self.library.nms(boxes, scores, iou_thr)
        return np_nms(boxes, scores, iou_thr)


# ══════════════════════════════════════════════════════════════════════════
#  Calibration: correctness + timing per kernel on representative shapes
# ══════════════════════════════════════════════════════════════════════════
def _time(fn: Callable[[], Any], repeats: int) -> float:
    fn()  # warm up
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1000.0


def _cal_bgr(lib: _MojoLibrary) -> tuple[bool, float, float]:
    frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    got, ref = lib.bgr_to_chw(frame), np_bgr_to_chw(frame)
    ok = bool(np.allclose(got, ref, atol=1e-6))
    return ok, _time(lambda: lib.bgr_to_chw(frame), 8), _time(lambda: np_bgr_to_chw(frame), 8)


def _cal_decode(lib: _MojoLibrary) -> tuple[bool, float, float]:
    rng = np.random.default_rng(7)
    pred = rng.random((84, 8400), dtype=np.float32) * 0.4
    pred[4:10, ::97] = 0.9  # plant some confident detections
    thr = 0.5
    gb, gs, gc = lib.decode_yolov8(pred, thr)
    rb, rs, rc = np_decode_yolov8(pred, thr)
    ok = (
        gb.shape == rb.shape
        and bool(np.allclose(np.sort(gs), np.sort(rs), atol=1e-5))
        and bool(np.allclose(gb, rb, atol=1e-4))
    )
    return (
        ok,
        _time(lambda: lib.decode_yolov8(pred, thr), 5),
        _time(lambda: np_decode_yolov8(pred, thr), 5),
    )


def _cal_iou(lib: _MojoLibrary) -> tuple[bool, float, float]:
    rng = np.random.default_rng(11)
    a = rng.random((40, 4), dtype=np.float32) * 100
    b = rng.random((40, 4), dtype=np.float32) * 100
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    ok = bool(np.allclose(lib.iou_matrix(a, b), np_iou_matrix(a, b), atol=1e-5))
    return ok, _time(lambda: lib.iou_matrix(a, b), 50), _time(lambda: np_iou_matrix(a, b), 50)


def _cal_pip(lib: _MojoLibrary) -> tuple[bool, float, float]:
    rng = np.random.default_rng(13)
    pts = rng.random((64, 2), dtype=np.float32)
    poly = np.array(
        [[0.15, 0.15], [0.85, 0.2], [0.9, 0.8], [0.2, 0.9]], dtype=np.float32
    )
    ok = bool(
        np.array_equal(lib.points_in_polygon(pts, poly), np_points_in_polygon(pts, poly))
    )
    return (
        ok,
        _time(lambda: lib.points_in_polygon(pts, poly), 100),
        _time(lambda: np_points_in_polygon(pts, poly), 100),
    )


def _cal_nms(lib: _MojoLibrary) -> tuple[bool, float, float]:
    rng = np.random.default_rng(17)
    boxes = rng.random((300, 4), dtype=np.float32) * 500
    boxes[:, 2:] += boxes[:, :2]
    scores = rng.random((300,), dtype=np.float32)
    ok = bool(np.array_equal(lib.nms(boxes, scores, 0.45), np_nms(boxes, scores, 0.45)))
    return (
        ok,
        _time(lambda: lib.nms(boxes, scores, 0.45), 20),
        _time(lambda: np_nms(boxes, scores, 0.45), 20),
    )


_CALIBRATIONS: Final[dict[str, Callable[[_MojoLibrary], tuple[bool, float, float]]]] = {
    "bgr_to_chw": _cal_bgr,
    "decode_yolov8": _cal_decode,
    "iou_matrix": _cal_iou,
    "points_in_polygon": _cal_pip,
    "nms": _cal_nms,
}


_bridge: KernelBridge | None = None


def get_bridge() -> KernelBridge:
    """Process-wide kernel bridge (calibrated once, on first use)."""
    global _bridge
    if _bridge is None:
        _bridge = KernelBridge()
    return _bridge
