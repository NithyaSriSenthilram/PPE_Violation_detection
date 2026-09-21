#!/usr/bin/env python3
"""Measure the real cost of every stage of the detection pipeline.

Every number printed here is measured on this machine, in this process, on the
frames given. Nothing is estimated, scaled from a reference machine, or carried
over from a previous run. Where a stage cannot run — no PPE model installed,
say — the report says so instead of substituting a plausible figure.

Stages measured
---------------
1. **Person detection** — the configured detector on the resolved backend.
2. **PPE detection**    — the PPE model, including association to tracks.
3. **Tracking**         — ByteTrack update over the detections.
4. **Event processing** — analysers plus the event engine, evidence off.
5. **Full pipeline**    — all of the above per frame, as the camera runs it,
                          which is what the achievable FPS actually depends on.

Method: a warm-up pass (the first ONNX/CoreML call includes graph compilation
and would otherwise dominate) followed by the measured run. Latency is reported
as mean and p95 — the tail is what drops frames, not the average.

Usage
-----
    python scripts/benchmark.py                       # sample video, 120 frames
    python scripts/benchmark.py --source path/to.mp4 --frames 300
    python scripts/benchmark.py --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE = PROJECT_ROOT / "tests" / "fixtures" / "surveillance_sample.mp4"
WARMUP_FRAMES = 5


class Timer:
    """Collects per-call latencies for one stage."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: list[float] = []
        self._started = 0.0

    def __enter__(self) -> Timer:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.samples.append((time.perf_counter() - self._started) * 1000.0)

    def summary(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        mean = statistics.fmean(self.samples)
        return {
            "stage": self.name,
            "calls": len(self.samples),
            "mean_ms": round(mean, 2),
            "median_ms": round(statistics.median(self.samples), 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(ordered[0], 2),
            "max_ms": round(ordered[-1], 2),
            "throughput_fps": round(1000.0 / mean, 1) if mean > 0 else None,
        }


def read_frames(source: Path, limit: int, max_width: int) -> list:
    """Decode up to `limit` frames once, so decode cost never skews a stage."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"could not open {source}")
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(
                frame, (max_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
    capture.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {source}")
    return frames


def memory_mb() -> float:
    """Resident set size of this process, in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux kilobytes.
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def run(source: Path, frame_limit: int, max_width: int) -> dict[str, Any]:
    from backend.analysis import build_analysers
    from backend.analysis.base import FrameContext
    from backend.config import settings
    from backend.events.engine import CooldownRegistry, EventEngine
    from backend.inference.ppe import ppe_diagnostics, resolve_ppe_detector
    from backend.inference.registry import (
        active_backend_name,
        inference_info,
        mojo_info,
        platform_info,
        resolve_detector,
    )
    from backend.tracking import ByteTracker

    # Models are loaded before the frames are decoded, so the model cost is
    # measurable on its own. Decoding first buries a 40 MB session inside a
    # 200 MB frame buffer and the subtraction comes out meaningless (or
    # negative, once the allocator returns arenas).
    baseline_memory = memory_mb()
    detector = resolve_detector()
    ppe_detector = resolve_ppe_detector()
    ppe_report = ppe_diagnostics()
    after_load = memory_mb()

    frames = read_frames(source, frame_limit + WARMUP_FRAMES, max_width)
    after_decode = memory_mb()

    person_timer = Timer("person detection")
    ppe_timer = Timer("PPE detection")
    track_timer = Timer("tracking")
    event_timer = Timer("event processing")
    pipeline_timer = Timer("full pipeline")

    tracker = ByteTracker()
    analysers = build_analysers("thorough")
    # Evidence and persistence are deliberately off: this measures the
    # detection pipeline, and disk I/O for snapshots is a separate cost that
    # would otherwise be attributed to event processing.
    engine = EventEngine(
        cooldown=CooldownRegistry(default_seconds=0.0, max_per_minute=0),
        persist=False, broadcast=False, capture_evidence=False,
    )

    ppe_usable = ppe_detector.method != "disabled"
    events_raised = 0
    people_seen = 0

    for index, frame in enumerate(frames):
        warming = index < WARMUP_FRAMES
        height, width = frame.shape[:2]
        started = time.perf_counter()

        if warming:
            result = detector.infer(frame)
            people = [d for d in result.detections if d.label == "person"]
            tracks = tracker.update(people, timestamp=index * 0.1)
            if ppe_usable and tracks:
                ppe_detector.assess(frame, tracks, people)
            continue

        with person_timer:
            result = detector.infer(frame)
        people = [d for d in result.detections if d.label == "person"]
        people_seen += len(people)

        with track_timer:
            tracks = tracker.update(people, timestamp=index * 0.1)

        assessments: dict = {}
        if ppe_usable and tracks:
            with ppe_timer:
                assessments = ppe_detector.assess(frame, tracks, people)

        with event_timer:
            context = FrameContext(
                camera_id="benchmark",
                frame_index=index,
                timestamp=index * 0.1,
                wall_time=time.time(),
                frame_width=width,
                frame_height=height,
                tracks=tracks,
                zones=[],
                ppe=assessments,
            )
            candidates = []
            for analyser in analysers:
                candidates.extend(analyser.analyse(context))
            events_raised += len(
                engine.submit(candidates, camera_id="benchmark",
                              monotonic_time=index * 0.1)
            )

        pipeline_timer.samples.append((time.perf_counter() - started) * 1000.0)

    peak_memory = memory_mb()
    stages = [
        t.summary()
        for t in (person_timer, ppe_timer, track_timer, event_timer, pipeline_timer)
    ]

    return {
        "source": str(source),
        "frames_measured": len(frames) - WARMUP_FRAMES,
        "warmup_frames": WARMUP_FRAMES,
        "frame_size": [frames[0].shape[1], frames[0].shape[0]],
        "people_detected_total": people_seen,
        "events_raised": events_raised,
        "stages": [s for s in stages if s],
        "skipped": (
            [] if ppe_usable else
            [f"PPE detection: {ppe_report['method_label']} — {ppe_report['warning']}"]
        ),
        "backends": {
            "person_detection": active_backend_name(),
            "person_model": inference_info().get("model_path"),
            "person_providers": inference_info().get("stats", {}).get(
                "active_providers", []
            ),
            "ppe_method": ppe_report["method"],
            "ppe_backend": ppe_report["backend"],
            "ppe_model": ppe_report.get("model_path"),
            "ppe_providers": (ppe_report.get("backend_detail") or {}).get(
                "active_providers", []
            ),
            "mojo_active": mojo_info().get("active"),
            "mojo_kernels": {
                name: kernel.get("implementation")
                for name, kernel in (mojo_info().get("kernels") or {}).items()
            },
        },
        "memory_mb": {
            "process_start": round(baseline_memory, 1),
            "after_loading_models": round(after_load, 1),
            "after_decoding_frames": round(after_decode, 1),
            "peak": round(peak_memory, 1),
            "models_cost": round(after_load - baseline_memory, 1),
            "frame_buffer_cost": round(after_decode - after_load, 1),
        },
        "platform": platform_info(),
        "config": {
            "confidence_threshold": settings.confidence_threshold,
            "ppe_confidence_threshold": settings.ppe_confidence_threshold,
            "inference_size": [settings.inference_width, settings.inference_height],
            "detect_every_n_frames": settings.detect_every_n_frames,
        },
        "note": (
            "Every stage runs on every measured frame. A live camera applies "
            f"DETECT_EVERY_N_FRAMES={settings.detect_every_n_frames}, so its "
            "sustained frame rate is higher than the full-pipeline figure here."
        ),
    }


def render(report: dict[str, Any]) -> None:
    bold, dim, reset = "\033[1m", "\033[2m", "\033[0m"
    if not sys.stdout.isatty():
        bold = dim = reset = ""

    print(f"\n{bold}SentinelVision AI — pipeline benchmark{reset}")
    print(f"{dim}{report['source']} · {report['frames_measured']} frames @ "
          f"{report['frame_size'][0]}x{report['frame_size'][1]} "
          f"({report['warmup_frames']} warm-up frames discarded){reset}\n")

    backends = report["backends"]
    print(f"{bold}Backends{reset}")
    print(f"  person detection   {backends['person_detection']} "
          f"({', '.join(backends['person_providers']) or 'n/a'})")
    print(f"  PPE detection      {backends['ppe_method']} on "
          f"{backends['ppe_backend']} "
          f"({', '.join(backends['ppe_providers']) or 'n/a'})")
    active = [n for n, impl in backends["mojo_kernels"].items() if impl == "mojo"]
    print(f"  Mojo kernels       {', '.join(active) if active else 'none active'}")

    print(f"\n{bold}Stages{reset}")
    print(f"  {'stage':<20} {'calls':>6} {'mean':>9} {'p95':>9} {'max':>9} {'fps':>8}")
    for stage in report["stages"]:
        print(f"  {stage['stage']:<20} {stage['calls']:>6} "
              f"{stage['mean_ms']:>7.2f}ms {stage['p95_ms']:>7.2f}ms "
              f"{stage['max_ms']:>7.2f}ms {stage['throughput_fps'] or 0:>8.1f}")

    memory = report["memory_mb"]
    print(f"\n{bold}Memory (RSS){reset}")
    print(f"  interpreter        {memory['process_start']:.1f} MB")
    print(f"  models loaded      {memory['after_loading_models']:.1f} MB "
          f"{dim}(+{memory['models_cost']:.1f} MB for both models){reset}")
    print(f"  frames buffered    {memory['after_decoding_frames']:.1f} MB "
          f"{dim}(+{memory['frame_buffer_cost']:.1f} MB — a benchmark artefact; "
          f"the pipeline streams frames){reset}")
    print(f"  peak               {memory['peak']:.1f} MB")

    print(f"\n{bold}Workload{reset}")
    print(f"  people detected    {report['people_detected_total']}")
    print(f"  events raised      {report['events_raised']}")
    for skip in report["skipped"]:
        print(f"  {bold}not measured{reset}       {skip}")
    print(f"\n{dim}{report['note']}{reset}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args.source, args.frames, args.max_width)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        render(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
