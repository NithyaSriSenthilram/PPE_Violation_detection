# ===----------------------------------------------------------------------=== #
#  SentinelVision AI — Mojo acceleration kernels
#
#  Scope (deliberately narrow, per the project's architecture rules): only the
#  numeric inner loops that run on every frame live here. Model execution
#  stays in ONNX Runtime, orchestration stays in Python.
#
#  Each kernel is exported with the C ABI and called from Python through
#  ctypes (see bridge.py). Every one has a numpy fallback with identical
#  semantics, and the bridge benchmarks both at startup so a kernel is only
#  used when it actually wins on the host machine.
#
#  Build:  ./build.sh          (or scripts/build_mojo.sh from the repo root)
#  Tested: Mojo 1.0.0, macOS arm64
#
#  Note on deprecation warnings: Mojo 1.0 accepts `@export(ABI="C")` while
#  advising an `abi("C")` function effect. The effect form is not yet usable
#  with `@export` in 1.0.0, so the supported keyword form is used here.
# ===----------------------------------------------------------------------=== #

from std.memory import Pointer

comptime F32 = Pointer[Float32, MutAnyOrigin]
comptime U8 = Pointer[UInt8, MutAnyOrigin]
comptime I32 = Pointer[Int32, MutAnyOrigin]


# ─────────────────────────────────────────────────────────────────────────
#  1. Frame conversion: BGR HWC uint8  ->  RGB CHW float32, scaled
#
#  This is the per-frame preprocessing step for YOLO. numpy needs three
#  passes over ~1.2 M elements (channel reverse, astype, divide) plus a
#  transpose; this does the whole thing in one pass with no temporaries.
# ─────────────────────────────────────────────────────────────────────────
@export("sv_bgr_hwc_to_rgb_chw", ABI="C")
def sv_bgr_hwc_to_rgb_chw(
    src: U8, dst: F32, h: Int32, w: Int32, scale: Float32
) -> Int32:
    var height = Int(h)
    var width = Int(w)
    var plane = height * width
    for y in range(height):
        var row = y * width * 3
        var out_row = y * width
        for x in range(width):
            var i = row + x * 3
            var o = out_row + x
            # source is BGR, destination planes are R, G, B
            dst[unsafe_offset=o] = Float32(Int(src[unsafe_offset=i + 2])) * scale
            dst[unsafe_offset=plane + o] = Float32(Int(src[unsafe_offset=i + 1])) * scale
            dst[unsafe_offset=2 * plane + o] = Float32(Int(src[unsafe_offset=i])) * scale
    return 0


# ─────────────────────────────────────────────────────────────────────────
#  2. YOLOv8 head decode + confidence gate
#
#  `pred` is the raw [num_attrs, num_anchors] output (e.g. 84 x 8400), laid
#  out row-major: all cx, then all cy, then w, h, then one row per class.
#  For every anchor we take the best class score and keep the box only if it
#  clears `conf_thr`. Writing xyxy directly saves another Python-side pass.
#
#  Returns the number of surviving boxes, or -1 if `max_out` was too small.
# ─────────────────────────────────────────────────────────────────────────
@export("sv_decode_yolov8", ABI="C")
def sv_decode_yolov8(
    pred: F32,
    num_attrs: Int32,
    num_anchors: Int32,
    conf_thr: Float32,
    boxes: F32,
    scores: F32,
    classes: I32,
    max_out: Int32,
) -> Int32:
    var anchors = Int(num_anchors)
    var attrs = Int(num_attrs)
    var num_classes = attrs - 4
    var cap = Int(max_out)
    var count = 0

    for a in range(anchors):
        # Best class for this anchor.
        var best = Float32(-1.0)
        var best_id = 0
        for c in range(num_classes):
            var s = pred[unsafe_offset=(4 + c) * anchors + a]
            if s > best:
                best = s
                best_id = c
        if best < conf_thr:
            continue
        if count >= cap:
            return -1

        var cx = pred[unsafe_offset=a]
        var cy = pred[unsafe_offset=anchors + a]
        var bw = pred[unsafe_offset=2 * anchors + a]
        var bh = pred[unsafe_offset=3 * anchors + a]
        var half_w = bw * 0.5
        var half_h = bh * 0.5

        var o = count * 4
        boxes[unsafe_offset=o] = cx - half_w
        boxes[unsafe_offset=o + 1] = cy - half_h
        boxes[unsafe_offset=o + 2] = cx + half_w
        boxes[unsafe_offset=o + 3] = cy + half_h
        scores[unsafe_offset=count] = best
        classes[unsafe_offset=count] = Int32(best_id)
        count += 1

    return Int32(count)


# ─────────────────────────────────────────────────────────────────────────
#  3. IoU matrix — the tracker's association cost
#
#  `a` is na x 4 and `b` is nb x 4, both xyxy. `dst` is na x nb row-major.
#  ByteTrack calls this twice per frame; it is pure arithmetic with no
#  allocation, which is exactly where a compiled kernel pays off.
# ─────────────────────────────────────────────────────────────────────────
@export("sv_iou_matrix", ABI="C")
def sv_iou_matrix(a: F32, na: Int32, b: F32, nb: Int32, dst: F32) -> Int32:
    var n = Int(na)
    var m = Int(nb)
    for i in range(n):
        var ai = i * 4
        var ax1 = a[unsafe_offset=ai]
        var ay1 = a[unsafe_offset=ai + 1]
        var ax2 = a[unsafe_offset=ai + 2]
        var ay2 = a[unsafe_offset=ai + 3]
        var area_a = (ax2 - ax1) * (ay2 - ay1)
        for j in range(m):
            var bj = j * 4
            var bx1 = b[unsafe_offset=bj]
            var by1 = b[unsafe_offset=bj + 1]
            var bx2 = b[unsafe_offset=bj + 2]
            var by2 = b[unsafe_offset=bj + 3]

            var ix1 = ax1 if ax1 > bx1 else bx1
            var iy1 = ay1 if ay1 > by1 else by1
            var ix2 = ax2 if ax2 < bx2 else bx2
            var iy2 = ay2 if ay2 < by2 else by2

            var iw = ix2 - ix1
            var ih = iy2 - iy1
            var iou = Float32(0.0)
            if iw > 0.0 and ih > 0.0:
                var inter = iw * ih
                var area_b = (bx2 - bx1) * (by2 - by1)
                var union = area_a + area_b - inter
                if union > 0.0:
                    iou = inter / union
            dst[unsafe_offset=i * m + j] = iou
    return 0


# ─────────────────────────────────────────────────────────────────────────
#  4. Batched point-in-polygon (even-odd ray casting)
#
#  Zone membership for every tracked person against one polygon. `pts` is
#  npts x 2, `poly` is nverts x 2, `dst` receives 0/1 per point.
# ─────────────────────────────────────────────────────────────────────────
@export("sv_points_in_polygon", ABI="C")
def sv_points_in_polygon(
    pts: F32, npts: Int32, poly: F32, nverts: Int32, dst: I32
) -> Int32:
    var n = Int(npts)
    var v = Int(nverts)
    for k in range(n):
        var px = pts[unsafe_offset=k * 2]
        var py = pts[unsafe_offset=k * 2 + 1]
        var inside = 0
        var j = v - 1
        for i in range(v):
            var xi = poly[unsafe_offset=i * 2]
            var yi = poly[unsafe_offset=i * 2 + 1]
            var xj = poly[unsafe_offset=j * 2]
            var yj = poly[unsafe_offset=j * 2 + 1]
            if (yi > py) != (yj > py):
                var denom = yj - yi
                if denom != 0.0:
                    if px < (xj - xi) * (py - yi) / denom + xi:
                        inside = 1 - inside
            j = i
        dst[unsafe_offset=k] = Int32(inside)
    return 0


# ─────────────────────────────────────────────────────────────────────────
#  5. Greedy NMS over pre-sorted boxes
#
#  `order` lists box indices by descending score. `keep` receives the
#  surviving indices; the return value is how many were kept.
# ─────────────────────────────────────────────────────────────────────────
@export("sv_nms", ABI="C")
def sv_nms(
    boxes: F32, order: I32, n: Int32, iou_thr: Float32, keep: I32
) -> Int32:
    var count = Int(n)
    var kept = 0
    for a in range(count):
        var i = Int(order[unsafe_offset=a])
        var bi = i * 4
        var ix1 = boxes[unsafe_offset=bi]
        var iy1 = boxes[unsafe_offset=bi + 1]
        var ix2 = boxes[unsafe_offset=bi + 2]
        var iy2 = boxes[unsafe_offset=bi + 3]
        var area_i = (ix2 - ix1) * (iy2 - iy1)

        var suppressed = False
        for b in range(kept):
            var j = Int(keep[unsafe_offset=b])
            var bj = j * 4
            var jx1 = boxes[unsafe_offset=bj]
            var jy1 = boxes[unsafe_offset=bj + 1]
            var jx2 = boxes[unsafe_offset=bj + 2]
            var jy2 = boxes[unsafe_offset=bj + 3]

            var ox1 = ix1 if ix1 > jx1 else jx1
            var oy1 = iy1 if iy1 > jy1 else jy1
            var ox2 = ix2 if ix2 < jx2 else jx2
            var oy2 = iy2 if iy2 < jy2 else jy2
            var ow = ox2 - ox1
            var oh = oy2 - oy1
            if ow > 0.0 and oh > 0.0:
                var inter = ow * oh
                var area_j = (jx2 - jx1) * (jy2 - jy1)
                var union = area_i + area_j - inter
                if union > 0.0 and (inter / union) > iou_thr:
                    suppressed = True
                    break
        if not suppressed:
            keep[unsafe_offset=kept] = Int32(i)
            kept += 1
    return Int32(kept)


# ─────────────────────────────────────────────────────────────────────────
#  6. Build/ABI probe — lets the bridge confirm it loaded a matching library
# ─────────────────────────────────────────────────────────────────────────
@export("sv_abi_version", ABI="C")
def sv_abi_version() -> Int32:
    return 1
