#!/usr/bin/env python3
"""Measure the hard-hat validator against labelled head crops.

The number this exists for is **cap_false_positive_rate**: how often ordinary
headwear is accepted as a valid industrial hard hat. That is the failure that
reports a worker in a baseball cap as PPE-compliant, and it is the one metric
that must approach zero. Recall is measured beside it and deliberately not
traded against it — a hard hat this stage fails to recognise costs an operator
one glance at the footage; a cap it accepts costs them the finding entirely.

The same numbers are reported for the PPE detector alone, so the comparison is
explicit: on the reference crops the detector calls 96% of caps helmets.

Usage
-----
    python scripts/hardhat_metrics.py                       # bundled fixtures
    python scripts/hardhat_metrics.py --crops path/to/dir   # your own set
    python scripts/hardhat_metrics.py --sweep               # threshold curve
    python scripts/hardhat_metrics.py --json                # machine-readable

The crop directory needs a ``manifest.json`` of ``{"file.png": "hardhat" |
"cap" | "bare"}``. Everything that is not ``hardhat`` counts as headwear that
must be rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CROPS = PROJECT_ROOT / "tests" / "fixtures" / "headwear"

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m",
)


def load_manifest(crops: Path) -> dict[str, str]:
    manifest = crops / "manifest.json"
    if not manifest.exists():
        print(
            f"ERROR: {manifest} not found. It maps each crop to its true "
            f'class: {{"cap_001.png": "cap", ...}}',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return json.loads(manifest.read_text())


def score(crops: Path, manifest: dict[str, str]) -> list[dict[str, Any]]:
    """Run the validator over every crop, keeping the raw probability."""
    import cv2

    from backend.inference.hardhat import ClipHardHatValidator

    available, reason, _ = ClipHardHatValidator.probe()
    if not available:
        print(f"ERROR: {reason}", file=sys.stderr)
        raise SystemExit(2)

    validator = ClipHardHatValidator()
    validator.load()

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for name, truth in sorted(manifest.items()):
        image = cv2.imread(str(crops / name))
        if image is None:
            print(f"  {YELLOW}skipping unreadable crop {name}{RESET}", file=sys.stderr)
            continue
        result = validator.classify(image)
        rows.append(
            {
                "file": name,
                "truth": truth,
                "is_hardhat": truth == "hardhat",
                "probability": round(result.hard_hat_probability, 4),
                "verdict": result.verdict,
            }
        )
    elapsed = (time.perf_counter() - started) / max(1, len(rows)) * 1000
    for row in rows:
        row["ms"] = round(elapsed, 1)
    return rows


def metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    """Precision, recall and the cap false-positive rate at one threshold."""
    hats = [r for r in rows if r["is_hardhat"]]
    others = [r for r in rows if not r["is_hardhat"]]
    accepted_hats = sum(r["probability"] >= threshold for r in hats)
    accepted_others = sum(r["probability"] >= threshold for r in others)
    accepted = accepted_hats + accepted_others

    return {
        "threshold": threshold,
        "hardhat_precision": (accepted_hats / accepted) if accepted else 1.0,
        "hardhat_recall": (accepted_hats / len(hats)) if hats else 0.0,
        # The headline. Every non-hard-hat crop that was accepted as PPE.
        "cap_false_positive_rate": (accepted_others / len(others)) if others else 0.0,
        "helmet_precision": (accepted_hats / accepted) if accepted else 1.0,
        "helmet_recall": (accepted_hats / len(hats)) if hats else 0.0,
        "accepted": accepted,
        "hardhats": len(hats),
        "others": len(others),
    }


def detector_baseline(manifest: dict[str, str]) -> dict[str, float] | None:
    """What the PPE detector alone says, for comparison.

    The bundled crops are all boxes the PPE detector labelled ``helmet`` — the
    fixture set exists because it got them wrong — so the baseline is simply
    "everything is accepted". Reported explicitly rather than implied.
    """
    hats = sum(1 for t in manifest.values() if t == "hardhat")
    others = len(manifest) - hats
    if not others:
        return None
    return {
        "hardhat_precision": hats / len(manifest),
        "hardhat_recall": 1.0,
        "cap_false_positive_rate": 1.0,
    }


def render(rows: list[dict[str, Any]], baseline: dict[str, float] | None, sweep: bool) -> int:
    from backend.config import settings

    threshold = settings.hardhat_confidence_threshold

    print(f"\n{BOLD}Hard-hat validator — measured on {len(rows)} labelled crops{RESET}")
    print(f"{DIM}{sum(r['is_hardhat'] for r in rows)} industrial hard hats, "
          f"{sum(not r['is_hardhat'] for r in rows)} caps / bare heads"
          f"   ({rows[0]['ms'] if rows else 0:.0f} ms per crop){RESET}\n")

    print(f"  {'crop':22} {'truth':9} {'p(hard hat)':>11}  verdict")
    for row in rows:
        ok = (row["probability"] >= threshold) == row["is_hardhat"]
        colour = "" if ok else RED
        mark = "" if ok else "  <-- wrong"
        print(f"  {colour}{row['file']:22} {row['truth']:9} "
              f"{row['probability']:11.3f}  {row['verdict']}{mark}{RESET}")

    current = metrics(rows, threshold)
    print(f"\n{BOLD}At the configured threshold ({threshold:.2f}){RESET}")
    fpr = current["cap_false_positive_rate"]
    colour = GREEN if fpr == 0 else (YELLOW if fpr < 0.05 else RED)
    print(f"  {colour}cap_false_positive_rate   {fpr:7.1%}{RESET}   "
          f"{DIM}ordinary headwear accepted as PPE — the number that matters{RESET}")
    print(f"  hardhat_recall            {current['hardhat_recall']:7.1%}")
    print(f"  hardhat_precision         {current['hardhat_precision']:7.1%}")
    print(f"  helmet_recall             {current['helmet_recall']:7.1%}")
    print(f"  helmet_precision          {current['helmet_precision']:7.1%}")

    if baseline:
        print(f"\n{BOLD}PPE detector alone, on the same crops{RESET}")
        print(f"  {RED}cap_false_positive_rate   "
              f"{baseline['cap_false_positive_rate']:7.1%}{RESET}   "
              f"{DIM}every cap here was labelled 'helmet'{RESET}")
        print(f"  hardhat_precision         {baseline['hardhat_precision']:7.1%}")

    if sweep:
        print(f"\n{BOLD}Threshold sweep{RESET}")
        print(f"  {'thr':>5}  {'recall':>7}  {'cap FPR':>8}  {'precision':>9}")
        for step in range(1, 20):
            point = metrics(rows, step / 20)
            print(f"  {point['threshold']:5.2f}  {point['hardhat_recall']:7.1%}  "
                  f"{point['cap_false_positive_rate']:8.1%}  "
                  f"{point['hardhat_precision']:9.1%}")

    if fpr > 0:
        print(f"\n{RED}{BOLD}FAIL{RESET} — {fpr:.1%} of ordinary headwear was accepted "
              f"as a valid hard hat.")
        return 1
    print(f"\n{GREEN}No ordinary headwear was accepted as a hard hat.{RESET}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crops", type=Path, default=DEFAULT_CROPS,
        help="directory of labelled crops with a manifest.json",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="print the threshold curve"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    manifest = load_manifest(args.crops)
    rows = score(args.crops, manifest)
    if not rows:
        print("ERROR: no readable crops", file=sys.stderr)
        return 2

    from backend.config import settings

    if args.json:
        payload = {
            "crops": rows,
            "at_configured_threshold": metrics(
                rows, settings.hardhat_confidence_threshold
            ),
            "sweep": [metrics(rows, step / 20) for step in range(1, 20)],
            "detector_baseline": detector_baseline(manifest),
        }
        print(json.dumps(payload, indent=2))
        return 0 if payload["at_configured_threshold"]["cap_false_positive_rate"] == 0 else 1

    return render(rows, detector_baseline(manifest), args.sweep)


if __name__ == "__main__":
    raise SystemExit(main())
