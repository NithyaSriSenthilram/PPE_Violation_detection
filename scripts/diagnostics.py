#!/usr/bin/env python3
"""Report what SentinelVision AI is *actually* running.

Works two ways, in this order:

1. If the API is up, it reads ``GET /api/diagnostics`` — the live truth,
   including which backend the running process resolved and the Mojo kernels
   it actually bound.
2. If it is not, it resolves the same information locally, in-process. That
   matters because the most common reason to run diagnostics is that the
   server will not start.

Nothing here fabricates a value. A missing model, an unloadable backend or a
class the mapping does not recognise is reported as such, with the action that
fixes it.

Usage
-----
    python scripts/diagnostics.py             # live if possible, else local
    python scripts/diagnostics.py --local     # never contact the API
    python scripts/diagnostics.py --json      # machine-readable, full detail
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m",
)


def _colour(enabled: bool):
    if enabled:
        return GREEN, YELLOW, RED, DIM, BOLD, RESET
    return "", "", "", "", "", ""


def fetch_live(url: str, timeout: float) -> dict[str, Any] | None:
    """Read the live diagnostics endpoint, or None when the API is not up."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def collect_local() -> dict[str, Any]:
    """Resolve diagnostics in this process, without the API."""
    from backend.config import settings
    from backend.inference.ppe import ppe_diagnostics
    from backend.inference.registry import (
        describe_backends,
        inference_info,
        mojo_info,
        platform_info,
        resolve_detector,
        video_output_info,
    )

    try:
        resolve_detector()
    except Exception as exc:  # reported below rather than raised
        print(f"  detector resolution failed: {exc}", file=sys.stderr)

    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "platform": platform_info(),
        "inference": inference_info(),
        "backends": [b.model_dump() for b in describe_backends()],
        "mojo": mojo_info(),
        "ppe": ppe_diagnostics(),
        "video_output": video_output_info(),
        "config": {
            "confidence_threshold": settings.confidence_threshold,
            "detect_every_n_frames": settings.detect_every_n_frames,
            "target_fps": settings.target_fps,
            "event_cooldown_seconds": settings.event_cooldown_seconds,
            "event_max_per_minute": settings.event_max_per_minute,
        },
        "source": "local",
    }


def render(data: dict[str, Any], colour: bool) -> int:
    """Print the human report. Returns the process exit code."""
    green, yellow, red, dim, bold, reset = _colour(colour)

    def mark(ok: bool | None) -> str:
        if ok is None:
            return f"{yellow}?{reset}"
        return f"{green}OK{reset}" if ok else f"{red}NO{reset}"

    def row(label: str, value: Any, status: bool | None = None) -> None:
        prefix = f"  {mark(status)} " if status is not None else "     "
        print(f"{prefix}{label:<26} {value}")

    warnings: list[str] = []

    print(f"\n{bold}SentinelVision AI — diagnostics{reset}")
    print(f"{dim}source: {data.get('source', 'live API')}{reset}\n")

    platform = data.get("platform", {})
    print(f"{bold}Host{reset}")
    row("platform", f"{platform.get('system')} {platform.get('release')} "
                    f"({platform.get('machine')})")
    row("cpu count", platform.get("cpu_count"))
    row("python", platform.get("python") or data.get("python_version"))

    # ── person detector ──────────────────────────────────────────────────
    inference = data.get("inference", {})
    print(f"\n{bold}Person detection{reset}")
    active = inference.get("active_backend", "none")
    synthetic = bool(inference.get("is_synthetic"))
    row("requested backend", inference.get("requested_backend"))
    row("active backend", active, status=active not in ("none", "mock"))
    row("model", inference.get("model_path"), status=inference.get("model_present"))
    row("confidence threshold", inference.get("confidence_threshold"))
    stats = inference.get("stats", {})
    if stats.get("active_providers"):
        row("execution providers", ", ".join(stats["active_providers"]))
    if stats.get("average_ms"):
        row("average latency", f"{stats['average_ms']} ms over {stats.get('frames', 0)} frames")
    if synthetic:
        warnings.append(
            "Person detection is running on the MOCK backend: detections are "
            "SYNTHETIC and no event from this process is real."
        )
    for attempt in inference.get("fallback_chain", []):
        print(f"     {dim}tried {attempt['backend']}: {attempt['reason']}{reset}")

    # ── PPE ──────────────────────────────────────────────────────────────
    ppe = data.get("ppe", {})
    print(f"\n{bold}PPE detection{reset}")
    method = ppe.get("method", "unknown")
    row("method", f"{ppe.get('method_label', method.upper())}", status=method == "model")
    row("model path", ppe.get("model_path"), status=ppe.get("model_present"))
    row("model format", ppe.get("model_format"))
    row("model loaded", ppe.get("model_loaded"), status=ppe.get("model_loaded"))
    row("inference backend", ppe.get("backend"),
        status=None if method != "model" else ppe.get("backend") not in ("none", "mock"))
    classes = ppe.get("classes") or []
    row("classes", f"{len(classes)}: {', '.join(map(str, classes))}" if classes else "—")
    mapping = ppe.get("mapping") or {}
    for label, role in mapping.items():
        print(f"       {dim}{label:<22} -> {role}{reset}")
    if ppe.get("unmapped_classes"):
        row("unmapped classes", ", ".join(ppe["unmapped_classes"]))
        warnings.append(
            f"PPE classes {', '.join(ppe['unmapped_classes'])} are not "
            f"recognised and are ignored — map them with PPE_CLASS_MAP."
        )
    row("confidence threshold", ppe.get("confidence_threshold"))
    row("association threshold", ppe.get("association_threshold"))
    row("required PPE", ", ".join(ppe.get("required_ppe") or []) or "—")
    row("heuristic fallback",
        "ENABLED" if ppe.get("heuristic_allowed") else "disabled")
    for attempt in ppe.get("fallback_chain", []):
        print(f"     {dim}tried {attempt['backend']}: {attempt['reason']}{reset}")
    if ppe.get("warning"):
        warnings.append(ppe["warning"])

    # ── hard-hat validation ──────────────────────────────────────────────
    # Reported as its own section, not a footnote to PPE: a deployment whose
    # helmet detections are unvalidated is not compliance-ready, however
    # healthy the PPE model looks above.
    hardhat = ppe.get("hardhat", {})
    if hardhat:
        print(f"\n{bold}Hard-hat validation{reset}")
        method = hardhat.get("method", "unknown")
        row("validator", hardhat.get("method_label", method.upper()),
            status=method == "model")
        row("validation", "enabled" if hardhat.get("enabled") else "DISABLED",
            status=bool(hardhat.get("enabled")))
        row("model path", hardhat.get("model_path"),
            status=hardhat.get("model_present"))
        row("prompt bank", hardhat.get("prompts_path"),
            status=hardhat.get("prompts_present"))
        if hardhat.get("model_id"):
            row("validator model", hardhat["model_id"])
        providers = hardhat.get("active_providers") or []
        if providers:
            row("backend", providers[0].replace("ExecutionProvider", "").lower())
        row("hard-hat threshold", hardhat.get("threshold"))
        row("revalidate every", f"{hardhat.get('revalidate_every_n_detections')} "
                                f"detections (window "
                                f"{hardhat.get('smoothing_window')})")
        row("heuristic fallback",
            "ENABLED" if hardhat.get("heuristic_allowed") else "disabled")
        if hardhat.get("hardhat_concepts"):
            print(f"     {dim}hard hat: "
                  f"{', '.join(hardhat['hardhat_concepts'])}{reset}")
            print(f"     {dim}not:      "
                  f"{', '.join(hardhat.get('other_concepts', []))}{reset}")
        for attempt in hardhat.get("fallback_chain", []):
            print(f"     {dim}tried {attempt['stage']}: {attempt['reason']}{reset}")
        row("PPE compliance ready", hardhat.get("compliance_ready"),
            status=bool(hardhat.get("compliance_ready")))
        if hardhat.get("warning"):
            warnings.append(hardhat["warning"])

    # ── video output ─────────────────────────────────────────────────────
    video = data.get("video_output", {})
    if video:
        print(f"\n{bold}Annotated video output{reset}")
        row("enabled", video.get("enabled"), status=video.get("enabled"))
        row("output directory", video.get("processed_dir"),
            status=video.get("processed_dir_writable"))
        row("renders on disk", f"{video.get('render_count', 0)} "
                               f"({video.get('render_mb', 0)} MB)")
        row("ffmpeg", video.get("ffmpeg") or "not installed",
            status=bool(video.get("ffmpeg")))
        row("ffprobe", video.get("ffprobe") or "not installed",
            status=bool(video.get("ffprobe")))
        row("max render width", video.get("max_width"))
        row("event overlay", f"{video.get('event_overlay_seconds')}s")
        drawn = [
            name
            for name, on in (
                ("local PPE boxes", video.get("show_ppe_boxes")),
                ("person boxes", video.get("show_person_boxes")),
                ("track IDs", video.get("show_track_id")),
            )
            if on
        ]
        row("overlay draws", ", ".join(drawn) or "nothing")
        if not video.get("ffmpeg"):
            warnings.append(
                "ffmpeg is not installed: annotated renders cannot preserve "
                "audio, cannot be transcoded if the codec is unplayable, and "
                "will not carry a front-loaded index — so seeking in the "
                "browser needs the whole file downloaded first."
            )
        if not video.get("processed_dir_writable"):
            warnings.append(
                f"Cannot write to {video.get('processed_dir')} — annotated "
                f"video generation will fail."
            )

    # ── backends ─────────────────────────────────────────────────────────
    print(f"\n{bold}Inference backends{reset}")
    for backend in data.get("backends", []):
        state = "ACTIVE" if backend.get("active") else (
            "available" if backend.get("available") else "unavailable"
        )
        row(backend.get("name", "?"), state,
            status=backend.get("active") or backend.get("available"))
        if backend.get("reason"):
            print(f"       {dim}{backend['reason']}{reset}")

    # ── Mojo ─────────────────────────────────────────────────────────────
    mojo = data.get("mojo", {})
    print(f"\n{bold}Mojo kernels{reset}")
    row("library", mojo.get("library") or mojo.get("reason") or "—",
        status=mojo.get("available"))
    row("mode", mojo.get("mode"))
    row("active", mojo.get("active"), status=mojo.get("active"))
    for name, kernel in (mojo.get("kernels") or {}).items():
        if not isinstance(kernel, dict):
            continue
        speedup = kernel.get("speedup")
        # A kernel is only used where it actually measured faster than numpy;
        # reporting the implementation in use keeps that honest.
        detail = f"{speedup}x vs numpy" if speedup else kernel.get("reason", "")
        row(f"  {name}", f"{kernel.get('implementation', '?'):<6} {dim}{detail}{reset}")

    # ── verdict ──────────────────────────────────────────────────────────
    if warnings:
        print(f"\n{bold}{yellow}Warnings{reset}")
        for warning in warnings:
            print(f"  {yellow}!{reset} {warning}")
    else:
        print(f"\n{green}No warnings — every component is running on real "
              f"models.{reset}")
    print()
    return 1 if synthetic else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="diagnostics endpoint URL")
    parser.add_argument("--local", action="store_true", help="never contact the API")
    parser.add_argument("--json", action="store_true", help="raw JSON output")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--no-colour", action="store_true")
    args = parser.parse_args()

    data: dict[str, Any] | None = None
    if not args.local:
        from backend.config import settings

        url = args.url or f"http://127.0.0.1:{settings.app_port}/api/diagnostics"
        data = fetch_live(url, args.timeout)
        if data is not None:
            data["source"] = f"live API ({url})"

    if data is None:
        if not args.local and not args.json:
            print(
                "  API not reachable — resolving diagnostics locally instead.",
                file=sys.stderr,
            )
        data = collect_local()

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0
    return render(data, colour=not args.no_colour and sys.stdout.isatty())


if __name__ == "__main__":
    raise SystemExit(main())
