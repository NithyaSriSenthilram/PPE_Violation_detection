#!/usr/bin/env python3
"""Download and prepare detection models for SentinelVision AI.

The runtime needs an ONNX detector at ``MODEL_PATH``. This script produces one
by downloading YOLO weights and exporting them with Ultralytics, which is only
required *here* — the serving path uses ONNX Runtime and never imports torch.

Usage
-----
    python scripts/fetch_models.py                 # yolov8n -> models/yolov8n.onnx
    python scripts/fetch_models.py --model yolo11n # newer, slightly better
    python scripts/fetch_models.py --imgsz 512     # smaller input, faster
    python scripts/fetch_models.py --labels-only   # just (re)write coco.names
    python scripts/fetch_models.py --ppe           # trained PPE detector
    python scripts/fetch_models.py --ppe-only      # PPE model, skip the person one
    python scripts/fetch_models.py --hardhat       # hard-hat validator + prompts
    python scripts/fetch_models.py --hardhat-variant q4f16   # 53 MB instead of 351

If Ultralytics is unavailable, the script explains how to install it and exits
non-zero rather than writing a broken placeholder file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# COCO class names, in the exact index order every YOLO COCO model emits.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Reference PPE class ordering. Written as documentation for anyone dropping in
# a PPE model — the runtime reads whatever names file accompanies the model and
# maps by name, so this exact order is not mandatory.
PPE_REFERENCE_NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "head"]

# Default PPE weights: a YOLO11s fine-tuned on a construction-safety dataset,
# with explicit *absence* classes (`no-helmet`, `no-vest`) as well as presence
# ones. That matters — a model that only labels helmets can report a missing
# helmet only by inference, which is a weaker signal. Any PPE detector whose
# classes the taxonomy recognises (or that PPE_CLASS_MAP maps) works here;
# override with --ppe-repo / --ppe-file.
PPE_REPO = "leeyunjai/yolo11-ppe"
PPE_WEIGHTS_FILE = "ppe-11s.pt"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{name}"

# ── Hard-hat validator ───────────────────────────────────────────────────
# A PPE detector finds head *coverings*: it calls a baseball cap a helmet, and
# it is confident when it does. Only an industrial hard hat is valid PPE, so
# every helmet detection is checked by a second model that can actually tell
# the two apart.
#
# CLIP ViT-B/32, used zero-shot. Nothing public is fine-tuned for hard-hat vs
# cap — every "hardhat detection" model on the Hub is trained on the same
# helmet/head datasets and inherits the same failure. A general vision-language
# model genuinely holds the distinction, and it measures out: on 45 head crops
# from real site footage (16 hard hats, 29 caps and bare heads) it gives 0%
# cap-false-positive at 75% hard-hat recall, against the PPE detector's 96%
# cap-false-positive rate on the same crops.
HARDHAT_REPO = "Xenova/clip-vit-base-patch32"
#: fp32 is the reference calibration and the fastest here (55 ms/crop against
#: 107 for q4f16); q4f16 matches its accuracy in 53 MB instead of 351.
HARDHAT_VARIANTS = {
    "fp32": "onnx/vision_model.onnx",
    "fp16": "onnx/vision_model_fp16.onnx",
    "q4f16": "onnx/vision_model_q4f16.onnx",
}
HARDHAT_TEXT_MODEL = "onnx/text_model.onnx"
HARDHAT_TOKENIZER = "tokenizer.json"

#: Prompt templates, averaged per concept. Plain and varied in framing rather
#: than clever: CLIP zero-shot is well known to gain from template ensembling,
#: and low-resolution wording matters here because surveillance heads are small.
HARDHAT_TEMPLATES = [
    "a photo of {}",
    "a close-up photo of {}",
    "a low resolution photo of {}",
    "a cropped photo of {}",
    "a blurry photo of {}",
]

#: The two classes, as concrete concepts rather than one averaged abstraction.
#: Each concept keeps its own embedding and the probabilities are grouped at
#: the end — averaging "a hard hat" with "a bare head" into a single anchor
#: produced a meaningless direction and scored at chance when measured.
# The colours are enumerated to *remove* a bias, not to create one. Every entry
# is still a hard hat: naming only white and yellow left the class leaning
# toward the two colours in the reference footage, which would under-serve a
# site issuing blue or red hats. Colour is never the rule — a blue cap and a
# blue hard hat differ in construction, and that is what the concepts oppose.
# Measured: with the colours, 0% cap-false-positive holds from 0.40 to 0.60
# instead of collapsing at 0.60, at identical recall.
HARDHAT_CONCEPTS = [
    "a hard hat",
    "an industrial safety helmet",
    "a construction hard hat",
    "a worker wearing a hard hat",
    "a white hard hat",
    "a yellow hard hat",
    "an orange hard hat",
    "a blue hard hat",
    "a red hard hat",
    "a green hard hat",
    "a black hard hat",
]
NOT_HARDHAT_CONCEPTS = [
    "a baseball cap",
    "a sports cap",
    "a cloth cap",
    "a knitted beanie",
    "a person wearing a baseball cap",
    "a bare head with hair",
    "a hood",
    "a sun hat",
    "a person with no hat",
]

#: CLIP's trained temperature. Baked into the bank so the runtime does not
#: have to know which model produced it.
CLIP_LOGIT_SCALE = 100.0


def write_labels(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({len(names)} classes)")


def export_onnx(model_name: str, imgsz: int, opset: int) -> Path:
    """Export YOLO weights to ONNX. Returns the final model path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "ERROR: Ultralytics is required to export ONNX weights.\n"
            "  Install it (one-off, export only):\n"
            "      pip install ultralytics\n"
            "  Or drop a pre-built ONNX detector at models/ and point\n"
            "  MODEL_PATH at it in .env.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / f"{model_name}.onnx"
    if target.exists():
        print(f"  {target.relative_to(PROJECT_ROOT)} already exists — skipping export")
        return target

    print(f"  downloading + exporting {model_name} (imgsz={imgsz}, opset={opset}) ...")
    # Ultralytics downloads the .pt into the cwd; keep it inside models/.
    weights = MODELS_DIR / f"{model_name}.pt"
    model = YOLO(str(weights) if weights.exists() else f"{model_name}.pt")

    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=False)
    exported_path = Path(exported)
    if exported_path.resolve() != target.resolve():
        shutil.move(str(exported_path), target)

    # Tidy the intermediate .pt into models/ so the repo root stays clean.
    stray = PROJECT_ROOT / f"{model_name}.pt"
    if stray.exists() and not weights.exists():
        shutil.move(str(stray), weights)

    print(f"  exported -> {target.relative_to(PROJECT_ROOT)} "
          f"({target.stat().st_size / 1e6:.1f} MB)")
    return target


def download(url: str, target: Path) -> Path:
    """Stream a file to disk, reporting progress on a single line."""
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310
        total = int(response.headers.get("content-length") or 0)
        written = 0
        with partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r    {written / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    print()
    partial.replace(target)
    return target


def export_ppe(repo: str, weights_name: str, imgsz: int, opset: int) -> Path:
    """Fetch trained PPE weights and export them to ONNX with their labels.

    The label file is written from the checkpoint's own class names, in index
    order — the runtime maps classes by name, and getting that order from
    anywhere else is how a helmet ends up being reported as a vest.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "ERROR: Ultralytics is required to export the PPE model.\n"
            "  pip install ultralytics",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "ppe.onnx"
    labels = MODELS_DIR / "ppe.names"
    if target.exists() and labels.exists():
        print(f"  {target.relative_to(PROJECT_ROOT)} already exists — skipping")
        return target

    weights = MODELS_DIR / "ppe.pt"
    if not weights.exists():
        download(HF_RESOLVE.format(repo=repo, name=weights_name), weights)

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    print(f"  PPE classes: {', '.join(names)}")

    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=False))
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    write_labels(labels, names)
    print(f"  exported -> {target.relative_to(PROJECT_ROOT)} "
          f"({target.stat().st_size / 1e6:.1f} MB)")
    return target


def export_hardhat(repo: str, variant: str) -> Path:
    """Fetch the hard-hat validator and build the prompt bank beside it.

    Two files land in models/: the image encoder, which is all the runtime
    executes, and an .npz holding the class prompt embeddings plus the exact
    preprocessing that produced them. Computing the text side *here* is what
    keeps the serving path free of a tokeniser, a text encoder and a set of
    hard-coded normalisation constants — the same split as the YOLO models,
    where the export needs torch and the runtime does not.
    """
    if variant not in HARDHAT_VARIANTS:
        print(
            f"ERROR: unknown --hardhat-variant {variant!r}; choose from "
            f"{', '.join(HARDHAT_VARIANTS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        print(
            "ERROR: building the hard-hat prompt bank needs the CLIP tokenizer.\n"
            "  Install it (one-off, this script only):\n"
            "      pip install tokenizers\n"
            "  The runtime does not need it — the prompts are precomputed here.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "hardhat_validator.onnx"
    prompts = MODELS_DIR / "hardhat_prompts.npz"
    if target.exists() and prompts.exists():
        print(f"  {target.relative_to(PROJECT_ROOT)} already exists — skipping")
        return target

    if not target.exists():
        download(HF_RESOLVE.format(repo=repo, name=HARDHAT_VARIANTS[variant]), target)

    # The text encoder and tokeniser are build-time only. They are fetched to a
    # scratch directory and deleted, so nothing that large ships in models/.
    scratch = MODELS_DIR / ".hardhat-build"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        text_model = scratch / "text_model.onnx"
        tokenizer_file = scratch / "tokenizer.json"
        if not text_model.exists():
            download(HF_RESOLVE.format(repo=repo, name=HARDHAT_TEXT_MODEL), text_model)
        if not tokenizer_file.exists():
            download(HF_RESOLVE.format(repo=repo, name=HARDHAT_TOKENIZER), tokenizer_file)

        print("  embedding prompts ...")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        session = ort.InferenceSession(
            str(text_model), providers=["CPUExecutionProvider"]
        )
        inputs = {spec.name for spec in session.get_inputs()}

        concepts = HARDHAT_CONCEPTS + NOT_HARDHAT_CONCEPTS
        vectors = []
        for concept in concepts:
            ids = []
            for template in HARDHAT_TEMPLATES:
                encoded = tokenizer.encode(template.format(concept)).ids[:77]
                ids.append(encoded + [0] * (77 - len(encoded)))
            batch = np.array(ids, dtype=np.int64)
            feed = {"input_ids": batch}
            if "attention_mask" in inputs:
                feed["attention_mask"] = (batch != 0).astype(np.int64)
            embedded = session.run(None, feed)[0].astype(np.float32)
            embedded /= np.linalg.norm(embedded, axis=1, keepdims=True)
            # Templates of ONE concept average to that concept; concepts are
            # never averaged with each other.
            mean = embedded.mean(axis=0)
            vectors.append(mean / np.linalg.norm(mean))

        np.savez(
            prompts,
            text_embeddings=np.stack(vectors).astype(np.float32),
            is_hardhat=np.array(
                [True] * len(HARDHAT_CONCEPTS) + [False] * len(NOT_HARDHAT_CONCEPTS)
            ),
            concepts=np.array(concepts),
            image_size=np.int32(224),
            image_mean=np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32),
            image_std=np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32),
            normalise=np.bool_(True),
            logit_scale=np.float32(CLIP_LOGIT_SCALE),
            model_id=np.str_(f"{repo}:{variant}"),
        )
        print(f"  wrote {prompts.relative_to(PROJECT_ROOT)} "
              f"({len(concepts)} concepts, {len(HARDHAT_CONCEPTS)} hard-hat)")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"  validator -> {target.relative_to(PROJECT_ROOT)} "
          f"({target.stat().st_size / 1e6:.1f} MB)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="yolov8n",
        help="YOLO checkpoint to export (yolov8n, yolov8s, yolo11n, ...)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="export input size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--labels-only", action="store_true", help="only write the label files"
    )
    parser.add_argument(
        "--ppe", action="store_true",
        help="also fetch and export the trained PPE detector",
    )
    parser.add_argument(
        "--ppe-only", action="store_true", help="fetch only the PPE detector"
    )
    parser.add_argument("--ppe-repo", default=PPE_REPO, help="Hugging Face repo id")
    parser.add_argument(
        "--ppe-file", default=PPE_WEIGHTS_FILE, help="weights file within the repo"
    )
    parser.add_argument(
        "--hardhat", action="store_true",
        help="also fetch the hard-hat validator that rejects ordinary caps",
    )
    parser.add_argument(
        "--hardhat-only", action="store_true", help="fetch only the hard-hat validator"
    )
    parser.add_argument("--hardhat-repo", default=HARDHAT_REPO, help="Hugging Face repo id")
    parser.add_argument(
        "--hardhat-variant", default="fp32", choices=sorted(HARDHAT_VARIANTS),
        help="fp32 (351 MB, reference) | fp16 | q4f16 (53 MB, same measured accuracy)",
    )
    args = parser.parse_args()

    print("SentinelVision AI — model setup")
    write_labels(MODELS_DIR / "coco.names", COCO_NAMES)
    write_labels(MODELS_DIR / "ppe.names.reference", PPE_REFERENCE_NAMES)

    if args.labels_only:
        print("Done (labels only).")
        return 0

    if not args.ppe_only and not args.hardhat_only:
        path = export_onnx(args.model, args.imgsz, args.opset)
        print(
            "\nPerson detector ready. Point .env at it if you exported a "
            "non-default model:\n"
            f"    MODEL_PATH=./models/{path.name}"
        )

    if args.ppe or args.ppe_only:
        ppe_path = export_ppe(args.ppe_repo, args.ppe_file, args.imgsz, args.opset)
        print(
            "\nPPE detector ready:\n"
            f"    PPE_MODEL_PATH=./models/{ppe_path.name}\n"
            "    PPE_MODEL_LABELS_PATH=./models/ppe.names\n"
            "Verify with: make diagnostics"
        )
    elif not args.hardhat_only:
        print(
            "\nNo PPE model fetched. PPE violations need one — run this with "
            "--ppe to get it."
        )

    if args.hardhat or args.hardhat_only:
        validator = export_hardhat(args.hardhat_repo, args.hardhat_variant)
        print(
            "\nHard-hat validator ready — ordinary caps will no longer count "
            "as helmets:\n"
            f"    HARDHAT_MODEL_PATH=./models/{validator.name}\n"
            "    HARDHAT_PROMPTS_PATH=./models/hardhat_prompts.npz\n"
            "Verify with: make diagnostics"
        )
    elif args.ppe or args.ppe_only:
        print(
            "\nNo hard-hat validator fetched. Without it every helmet "
            "detection is reported as a violation, because the platform "
            "cannot confirm the headwear is an industrial hard hat — run this "
            "with --hardhat."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
