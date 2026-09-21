"""Hard-hat validation: is that head covering actually industrial PPE?

A PPE detector is trained to find *head coverings*. Ask one what a construction
worker in a baseball cap is wearing and it says ``helmet`` — measured on the
reference footage at 27 of 28 caps, with confidences up to 0.80. Confidence is
no help here, because the model is confident and wrong: the failure is
semantic, not statistical. Raising ``PPE_CONFIDENCE_THRESHOLD`` only trades
away the real hard hats.

So every helmet the detector proposes goes through a second stage that asks a
different question — *is this an industrial hard hat, or is it a cap?* — and
the answer, not the detection, decides compliance:

    PPE detector: helmet 0.76  →  validator: hard hat 0.12  →  NO HELMET
    PPE detector: helmet 0.88  →  validator: hard hat 0.94  →  HELMET

Two implementations, and the difference between them is never hidden.

**ClipHardHatValidator** — a CLIP image/text model run as a zero-shot
classifier over a bank of concrete headwear concepts (hard hat, safety helmet,
baseball cap, beanie, bare head, …), grouped into *hard hat* and *not a hard
hat*. It is not fine-tuned for this task; it is a general vision-language model
that genuinely holds the distinction, which is more than can be said for the
public "hardhat detection" weights — every one of those is trained on the same
helmet/head datasets and inherits the same cap problem. Measured on 45 head
crops from the reference site footage (16 hard hats, 29 caps and bare heads):
**0% cap false-positive rate at 75% hard-hat recall**.

**HeuristicHardHatValidator** — structural, and off unless
``HARDHAT_ALLOW_HEURISTIC=true``. It combines several weak shape signals (dome
roundness, brim, edge density, surface smoothness); it does not classify by
colour, because hard hats and caps come in the same colours. It is materially
weaker than the model and says so.

When neither is available the answer is ``unknown``, and **unknown is not
compliance**: the helmet is rejected, diagnostics report the validator as
unavailable, and the platform never claims a cap is PPE because it could not
check. That direction is deliberate — a missed hard hat costs an operator one
glance at the footage, while a cap accepted as a helmet is a false report of
compliance, which is the whole reason this stage exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from backend.config import settings
from backend.inference.base import BackendUnavailable
from backend.logging_conf import get_logger

logger = get_logger(__name__)

Verdict = Literal["hard_hat", "not_hard_hat", "unknown"]
ValidatorMethod = Literal["model", "heuristic", "unavailable", "disabled"]

#: How each method is named to an operator, matching the PPE method labels.
METHOD_LABELS: dict[str, str] = {
    "model": "AI MODEL",
    "heuristic": "HEURISTIC FALLBACK",
    "unavailable": "UNAVAILABLE",
    "disabled": "DISABLED",
}

#: Ceiling on heuristic confidence. Shape statistics must never present
#: themselves as being as sure as the model.
HEURISTIC_CONFIDENCE_CEILING = 0.75

#: Padding around the detector's helmet box before the crop is classified, as a
#: fraction of the box. Deliberately small: measured on the reference crops,
#: widening the crop to include shoulders and vest drove the cap
#: false-positive rate from 0% to 45%, because a hi-vis worker *looks like*
#: someone in a hard hat to a model reading the whole scene. The tight crop
#: forces the question to be about the headwear itself.
CROP_PADDING = 0.05

#: Below this many pixels on a side, the crop carries too little detail to
#: classify and the answer is `unknown` rather than a guess.
MIN_CROP_SIDE = 16


@dataclass(slots=True)
class HardHatResult:
    """One validator answer about one head."""

    verdict: Verdict
    #: Confidence in *this verdict* — p(hard hat) when the verdict is
    #: ``hard_hat``, p(not a hard hat) when it is ``not_hard_hat``. Never the
    #: PPE detector's confidence, which is kept separately: the detector was
    #: sure it saw a helmet, and it was wrong.
    confidence: float
    #: The raw grouped probability that the crop shows an industrial hard hat.
    hard_hat_probability: float
    method: ValidatorMethod
    reason: str = ""

    @property
    def is_hard_hat(self) -> bool:
        return self.verdict == "hard_hat"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "hard_hat_probability": round(self.hard_hat_probability, 3),
            "method": self.method,
            **({"reason": self.reason} if self.reason else {}),
        }


def unknown(method: ValidatorMethod, reason: str) -> HardHatResult:
    """The answer when the question could not be asked. Never compliance."""
    return HardHatResult("unknown", 0.0, 0.0, method, reason)


def crop_head(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    padding: float = CROP_PADDING,
) -> np.ndarray | None:
    """The head crop a validator classifies, or None if there is not one.

    Built from the detector's own helmet box rather than a fraction of the
    person box: the detector has already localised the headwear, and the
    tighter the crop the less the surrounding construction scene can vote.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right - left < MIN_CROP_SIDE or bottom - top < MIN_CROP_SIDE:
        return None
    crop = frame[top:bottom, left:right]
    return crop if crop.size else None


# ══════════════════════════════════════════════════════════════════════════
#  Interface
# ══════════════════════════════════════════════════════════════════════════
class HardHatValidator(ABC):
    """Decides whether a detected helmet is an industrial hard hat."""

    method: ValidatorMethod = "unavailable"
    description: str = ""

    @abstractmethod
    def classify(self, crop: np.ndarray) -> HardHatResult:
        """Classify one head crop."""

    def validate(
        self, frame: np.ndarray, box: tuple[float, float, float, float]
    ) -> HardHatResult:
        """Validate the headwear in `box`, cropping it from `frame`."""
        crop = crop_head(frame, box)
        if crop is None:
            return unknown(self.method, "head crop too small to classify")
        try:
            return self.classify(crop)
        except Exception as exc:  # a broken session must not fail the frame
            logger.debug("Hard-hat validation failed: %s", exc)
            return unknown(self.method, f"validation failed: {exc}")

    def close(self) -> None:
        """Release resources. Default is a no-op — nothing to release."""
        return None

    def info(self) -> dict[str, Any]:
        return {"method": self.method, "description": self.description}


class UnavailableHardHatValidator(HardHatValidator):
    """No validator. Every answer is ``unknown`` — and therefore not a helmet.

    This is not a degraded version of the model: it is the platform declining
    to certify PPE it cannot check. Diagnostics report it, and the reason is
    carried on every result so it appears in the event metadata too.
    """

    method: ValidatorMethod = "unavailable"

    def __init__(self, reason: str = "", method: ValidatorMethod = "unavailable") -> None:
        self.reason = reason or "no hard-hat validator is installed"
        self.method = method
        self.description = f"Hard-hat validation unavailable — {self.reason}"

    def classify(self, crop: np.ndarray) -> HardHatResult:
        return unknown(self.method, self.reason)

    def validate(
        self, frame: np.ndarray, box: tuple[float, float, float, float]
    ) -> HardHatResult:
        return unknown(self.method, self.reason)

    def info(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "description": self.description,
            "reason": self.reason,
        }


# ══════════════════════════════════════════════════════════════════════════
#  CLIP zero-shot validator
# ══════════════════════════════════════════════════════════════════════════
#: Execution providers for the validator, in preference order.
#:
#: CoreML is deliberately absent. It is first choice for the YOLO detectors and
#: measurably right there, but this is a ViT with a dynamic batch dimension and
#: CoreML partitions it badly: 110 ms per crop against 55 ms on CPU, measured on
#: the M1 this runs on. GPU providers stay in the list for hosts that have them.
PROVIDER_PREFERENCE: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
)


@dataclass(slots=True)
class PromptBank:
    """Text-side of the zero-shot classifier, and how to feed the image side.

    Written by ``scripts/fetch_models.py --hardhat`` next to the model, so the
    runtime needs no tokeniser, no text encoder and no hard-coded preprocessing
    constants — the numbers that produced these embeddings travel with them.
    """

    embeddings: np.ndarray          # (concepts, dim), L2-normalised
    is_hardhat: np.ndarray          # (concepts,) bool
    concepts: list[str]
    image_size: int
    image_mean: np.ndarray
    image_std: np.ndarray
    normalise: bool
    logit_scale: float
    model_id: str = ""

    @classmethod
    def load(cls, path: Path) -> PromptBank:
        if not path.exists():
            raise BackendUnavailable(
                f"no hard-hat prompt bank at {path} — run "
                f"'python scripts/fetch_models.py --hardhat'"
            )
        try:
            data = np.load(path, allow_pickle=False)
            embeddings = np.asarray(data["text_embeddings"], dtype=np.float32)
            is_hardhat = np.asarray(data["is_hardhat"], dtype=bool)
            bank = cls(
                embeddings=embeddings,
                is_hardhat=is_hardhat,
                concepts=[str(c) for c in data["concepts"]],
                image_size=int(data["image_size"]),
                image_mean=np.asarray(data["image_mean"], dtype=np.float32),
                image_std=np.asarray(data["image_std"], dtype=np.float32),
                normalise=bool(data["normalise"]),
                logit_scale=float(data["logit_scale"]),
                model_id=str(data["model_id"]) if "model_id" in data else "",
            )
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable(
                f"hard-hat prompt bank {path} could not be read: {exc}"
            ) from exc

        if bank.embeddings.ndim != 2 or len(bank.is_hardhat) != len(bank.embeddings):
            raise BackendUnavailable(
                f"hard-hat prompt bank {path} is malformed: "
                f"{bank.embeddings.shape} embeddings against "
                f"{len(bank.is_hardhat)} labels"
            )
        if not bank.is_hardhat.any() or bank.is_hardhat.all():
            raise BackendUnavailable(
                f"hard-hat prompt bank {path} has only one class; it cannot "
                f"distinguish a hard hat from anything"
            )
        return bank


class ClipHardHatValidator(HardHatValidator):
    """Zero-shot hard-hat / not-a-hard-hat classification with a CLIP model.

    The image is embedded once and compared with a bank of concept embeddings.
    Probabilities over the concepts are grouped into the two classes, so the
    decision reads as "this looks more like a baseball cap than a hard hat"
    rather than as an opaque score.
    """

    method: ValidatorMethod = "model"

    def __init__(
        self, model_path: Path | None = None, prompts_path: Path | None = None
    ) -> None:
        self.model_path = model_path or settings.hardhat_model_file
        self.prompts_path = prompts_path or settings.hardhat_prompts_file
        self.session: Any = None
        self.bank: PromptBank | None = None
        self.providers: list[str] = []
        self._input_name = "pixel_values"
        self._input_dtype: np.dtype = np.dtype(np.float32)
        self.description = f"Hard-hat validator ({self.model_path.name})"

    # ── lifecycle ────────────────────────────────────────────────────────
    @classmethod
    def probe(
        cls, model_path: Path | None = None, prompts_path: Path | None = None
    ) -> tuple[bool, str, dict[str, Any]]:
        """Report whether the validator is installed, without loading it."""
        model = model_path or settings.hardhat_model_file
        prompts = prompts_path or settings.hardhat_prompts_file
        detail: dict[str, Any] = {
            "model_path": str(model),
            "model_present": model.exists(),
            "prompts_path": str(prompts),
            "prompts_present": prompts.exists(),
        }
        fetch = "run 'python scripts/fetch_models.py --hardhat'"
        if not model.exists():
            return False, f"no hard-hat validator at {model} — {fetch}", detail
        detail["model_size_mb"] = round(model.stat().st_size / 1e6, 1)
        if not prompts.exists():
            return (
                False,
                f"hard-hat validator {model.name} has no prompt bank at "
                f"{prompts}; its output cannot be interpreted — {fetch}",
                detail,
            )
        return True, "", detail

    def load(self) -> None:
        available, reason, _ = self.probe(self.model_path, self.prompts_path)
        if not available:
            raise BackendUnavailable(reason)

        self.bank = PromptBank.load(self.prompts_path)

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.inference_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        installed = set(ort.get_available_providers())
        providers = [p for p in PROVIDER_PREFERENCE if p in installed] or [
            "CPUExecutionProvider"
        ]
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), options, providers=providers
            )
        except Exception as exc:
            raise BackendUnavailable(
                f"failed to create the hard-hat validator session: {exc}"
            ) from exc

        spec = self.session.get_inputs()[0]
        self._input_name = spec.name
        self._input_dtype = np.dtype(
            np.float16 if spec.type == "tensor(float16)" else np.float32
        )
        self.providers = list(self.session.get_providers())

        dim = self.session.get_outputs()[0].shape[-1]
        if isinstance(dim, int) and dim != self.bank.embeddings.shape[1]:
            raise BackendUnavailable(
                f"hard-hat validator emits {dim}-d embeddings but its prompt "
                f"bank is {self.bank.embeddings.shape[1]}-d — they are from "
                f"different models; re-run scripts/fetch_models.py --hardhat"
            )

        self.description = (
            f"Hard-hat validator ({self.model_path.name}"
            f"{f', {self.bank.model_id}' if self.bank.model_id else ''}) "
            f"on {self.providers[0].replace('ExecutionProvider', '').lower()}"
        )
        logger.info(
            "Hard-hat validator ready: %s (%d concepts, %d hard-hat) via %s "
            "@ p(hard hat)>=%.2f",
            self.model_path.name, len(self.bank.concepts),
            int(self.bank.is_hardhat.sum()), self.providers[0],
            settings.hardhat_confidence_threshold,
        )

    def close(self) -> None:
        self.session = None

    # ── classification ───────────────────────────────────────────────────
    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR crop → the model's input tensor.

        Resize on the short edge and centre-crop, which is what the encoder was
        trained on and what measured best; a letterboxed or stretched crop cost
        recall on the reference set.
        """
        assert self.bank is not None
        size = self.bank.image_size
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = size / min(height, width)
        rgb = cv2.resize(
            rgb,
            (max(size, int(round(width * scale))), max(size, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        height, width = rgb.shape[:2]
        top, left = (height - size) // 2, (width - size) // 2
        rgb = rgb[top : top + size, left : left + size]

        tensor = rgb.astype(np.float32) / 255.0
        if self.bank.normalise:
            tensor = (tensor - self.bank.image_mean) / self.bank.image_std
        return np.transpose(tensor, (2, 0, 1))[None].astype(self._input_dtype)

    def classify(self, crop: np.ndarray) -> HardHatResult:
        if self.session is None or self.bank is None:
            return unknown(self.method, "hard-hat validator is not loaded")

        embedding = self.session.run(
            None, {self._input_name: self.preprocess(crop)}
        )[0][0].astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            return unknown(self.method, "validator produced an empty embedding")

        logits = (self.bank.embeddings @ (embedding / norm)) * self.bank.logit_scale
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        hard_hat = float(probabilities[self.bank.is_hardhat].sum())

        return _decide(hard_hat, self.method)

    def info(self) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "method": self.method,
            "description": self.description,
            "model_path": str(self.model_path),
            "prompts_path": str(self.prompts_path),
            "active_providers": self.providers,
            "accelerated": bool(
                self.providers and self.providers[0] != "CPUExecutionProvider"
            ),
        }
        if self.bank is not None:
            detail.update(
                model_id=self.bank.model_id,
                concepts=len(self.bank.concepts),
                hardhat_concepts=[
                    c for c, flag in zip(self.bank.concepts, self.bank.is_hardhat, strict=True) if flag
                ],
                other_concepts=[
                    c for c, flag in zip(self.bank.concepts, self.bank.is_hardhat, strict=True) if not flag
                ],
                image_size=self.bank.image_size,
            )
        return detail


def _decide(hard_hat_probability: float, method: ValidatorMethod) -> HardHatResult:
    """Turn a hard-hat probability into a verdict at the configured threshold."""
    threshold = settings.hardhat_confidence_threshold
    if hard_hat_probability >= threshold:
        return HardHatResult(
            "hard_hat", hard_hat_probability, hard_hat_probability, method
        )
    return HardHatResult(
        "not_hard_hat", 1.0 - hard_hat_probability, hard_hat_probability, method
    )


# ══════════════════════════════════════════════════════════════════════════
#  Heuristic validator
# ══════════════════════════════════════════════════════════════════════════
class HeuristicHardHatValidator(HardHatValidator):
    """Structural fallback, used only when explicitly permitted.

    A hard hat and a baseball cap differ in *construction*, and that shows up in
    the pixels: a hard hat is a rigid dome — a smooth, near-symmetric shell with
    a brim all the way round and very little internal detail — while a cap has a
    soft crown, a single forward visor, fabric texture, panel seams and a
    squarer silhouette.

    Four weak signals are combined; none of them is colour, because hard hats
    and caps come in the same colours and a colour rule would fail on the first
    white cap or black hard hat:

    * **shell smoothness** — internal edge density over the crown, low for a
      moulded shell and high for seamed fabric;
    * **dome curvature** — how well the top contour fits a circular arc;
    * **silhouette symmetry** — left/right balance, broken by a forward visor;
    * **aspect** — hard hats are close to as tall as they are wide, caps sit
      lower and wider.

    The result is capped at :data:`HEURISTIC_CONFIDENCE_CEILING` and reported as
    ``heuristic`` everywhere, so it is never mistaken for the model. It is
    genuinely weaker than the model and is off by default for that reason.
    """

    method: ValidatorMethod = "heuristic"
    description = (
        "Structural hard-hat estimate (shell smoothness, dome curvature, "
        "symmetry, aspect) — no trained validator installed"
    )

    def classify(self, crop: np.ndarray) -> HardHatResult:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape[:2]
        if height < MIN_CROP_SIDE or width < MIN_CROP_SIDE:
            return unknown(self.method, "head crop too small to measure")

        # The crown: the upper band, where a shell is smooth and a cap is not.
        crown = grey[: max(1, int(height * 0.55))]
        crown = cv2.resize(crown, (64, max(8, int(64 * crown.shape[0] / width))))
        edges = cv2.Canny(cv2.GaussianBlur(crown, (3, 3), 0), 60, 160)
        edge_density = float(edges.mean()) / 255.0
        smoothness = float(np.clip(1.0 - edge_density / 0.18, 0.0, 1.0))

        # Dome curvature: the height of the filled silhouette along each column
        # traces the top contour. A shell is a smooth arc; a cap is flatter and
        # steps down at the visor.
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        _, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if mask.mean() > 127:
            mask = 255 - mask
        columns = np.argmax(mask > 0, axis=0).astype(np.float32)
        covered = (mask > 0).any(axis=0)
        curvature = 0.0
        symmetry = 0.0
        if covered.sum() >= 8:
            xs = np.nonzero(covered)[0].astype(np.float32)
            ys = columns[covered]
            fit = np.polyfit(xs, ys, 2)
            residual = float(np.mean((np.polyval(fit, xs) - ys) ** 2)) / max(
                1.0, float(height) ** 2
            )
            # A dome curves downward at the edges: a positive quadratic term.
            arc = float(np.clip(fit[0] * width * width / max(1.0, height) / 4.0, 0.0, 1.0))
            curvature = float(np.clip(arc * (1.0 - np.clip(residual * 40.0, 0.0, 1.0)), 0.0, 1.0))
            mirrored = ys[::-1]
            spread = float(ys.max() - ys.min()) or 1.0
            symmetry = float(
                np.clip(1.0 - np.mean(np.abs(ys - mirrored)) / spread, 0.0, 1.0)
            )

        aspect = height / max(1.0, float(width))
        # Hard hats sit near square; caps are wider than tall.
        aspect_score = float(np.clip((aspect - 0.55) / 0.35, 0.0, 1.0))

        score = (
            0.35 * smoothness + 0.25 * curvature + 0.20 * symmetry + 0.20 * aspect_score
        )
        # Pull the score toward uncertainty and cap it: this is shape
        # statistics, not recognition, and it must never sound certain.
        probability = float(0.5 + (score - 0.5) * HEURISTIC_CONFIDENCE_CEILING)
        result = _decide(probability, self.method)
        result.reason = (
            f"smoothness={smoothness:.2f} curvature={curvature:.2f} "
            f"symmetry={symmetry:.2f} aspect={aspect_score:.2f}"
        )
        return result


# ══════════════════════════════════════════════════════════════════════════
#  Temporal smoothing
# ══════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class _TrackVerdict:
    """What the validator has recently said about one tracked person."""

    probabilities: deque[float]
    method: ValidatorMethod
    last_checked: int
    reason: str = ""


class TrackedHardHatValidator:
    """Per-person validation: revalidated periodically, averaged, and cheap.

    Two problems solved by one small piece of memory. The check costs about
    55 ms per head, which every frame cannot afford; and a single unlucky look —
    a head turned away, a frame of motion blur — should not flip a worker
    between compliant and not. So each person is re-checked every
    ``HARDHAT_VALIDATE_EVERY_N_DETECTIONS`` detection frames, and the verdict
    read out is the mean over the last ``HARDHAT_SMOOTHING_WINDOW`` checks.

    This sits *above* the event debounce rather than replacing it: PPE
    violations still have to hold for ``PPE_MIN_CONSECUTIVE_FRAMES`` before an
    event fires, and the event engine's cooldown still sits behind that.

    Keys are ``(scope, track_id)``. The scope matters: track IDs are per
    tracker, so a camera and an upload job both counting from 1 would otherwise
    share each other's verdicts.
    """

    def __init__(self, validator: HardHatValidator) -> None:
        self.validator = validator
        self._state: dict[tuple[str, int], _TrackVerdict] = {}
        self._frame = 0

    @property
    def method(self) -> ValidatorMethod:
        return self.validator.method

    def begin_frame(self) -> None:
        """Advance the detection-frame counter that paces revalidation."""
        self._frame += 1

    def validate_track(
        self,
        frame: np.ndarray,
        box: tuple[float, float, float, float],
        track_id: int,
        scope: str = "",
    ) -> HardHatResult:
        """Verdict for one tracked person, revalidating only when due."""
        key = (scope, track_id)
        state = self._state.get(key)
        interval = settings.hardhat_validate_every_n_detections

        if state is None or self._frame - state.last_checked >= interval:
            fresh = self.validator.validate(frame, box)
            if fresh.verdict == "unknown":
                # Nothing learned. Keep whatever is known rather than
                # discarding it, but never invent a verdict.
                if state is None:
                    return fresh
                state.last_checked = self._frame
            else:
                window = max(1, settings.hardhat_smoothing_window)
                if state is None:
                    state = _TrackVerdict(
                        deque(maxlen=window), fresh.method, self._frame, fresh.reason
                    )
                    self._state[key] = state
                if state.probabilities.maxlen != window:
                    state.probabilities = deque(state.probabilities, maxlen=window)
                state.probabilities.append(fresh.hard_hat_probability)
                state.method = fresh.method
                state.reason = fresh.reason
                state.last_checked = self._frame

        state = self._state.get(key)
        if state is None or not state.probabilities:
            return unknown(self.method, "no validation yet for this person")
        smoothed = float(np.mean(state.probabilities))
        result = _decide(smoothed, state.method)
        result.reason = state.reason
        return result

    def forget(self, scope: str = "", keep: set[int] | None = None) -> None:
        """Drop state for people no longer tracked in `scope`."""
        for key in list(self._state):
            if key[0] == scope and (keep is None or key[1] not in keep):
                del self._state[key]

    def info(self) -> dict[str, Any]:
        return {
            **self.validator.info(),
            "tracked_people": len(self._state),
            "revalidate_every_n_detections": (
                settings.hardhat_validate_every_n_detections
            ),
            "smoothing_window": settings.hardhat_smoothing_window,
        }


# ══════════════════════════════════════════════════════════════════════════
#  Resolution
# ══════════════════════════════════════════════════════════════════════════
_validator: TrackedHardHatValidator | None = None
#: Why each stage was skipped, for diagnostics — the same honesty the
#: inference registry applies to its own backend chain.
_fallback_chain: list[tuple[str, str]] = []


def resolve_hardhat_validator() -> TrackedHardHatValidator:
    """The process-wide validator, chosen once and reported honestly.

    Order: the trained model, then the heuristic *if explicitly permitted*,
    then nothing — and "nothing" means every helmet is rejected, not accepted.
    """
    global _validator
    if _validator is not None:
        return _validator

    _fallback_chain.clear()

    if not settings.hardhat_validation_enabled:
        logger.warning(
            "HARDHAT_VALIDATION_ENABLED=false: helmet detections are trusted "
            "as-is, so ordinary caps will be reported as valid PPE."
        )
        _validator = TrackedHardHatValidator(
            UnavailableHardHatValidator(
                "hard-hat validation is switched off "
                "(HARDHAT_VALIDATION_ENABLED=false)",
                method="disabled",
            )
        )
        return _validator

    candidate = ClipHardHatValidator()
    try:
        candidate.load()
        _validator = TrackedHardHatValidator(candidate)
        return _validator
    except BackendUnavailable as exc:
        _fallback_chain.append(("model", str(exc)))
        logger.warning("Hard-hat validator unavailable: %s", exc)

    if settings.hardhat_allow_heuristic:
        logger.warning(
            "Falling back to the HEURISTIC hard-hat validator. It is shape "
            "statistics, not recognition, and is reported as HEURISTIC "
            "everywhere it is used."
        )
        _validator = TrackedHardHatValidator(HeuristicHardHatValidator())
        return _validator

    _fallback_chain.append(
        ("heuristic", "HARDHAT_ALLOW_HEURISTIC=false, so no fallback was tried")
    )
    logger.warning(
        "No hard-hat validator: every detected helmet will be reported as NOT "
        "a hard hat, because the platform cannot verify that it is one. Run "
        "'python scripts/fetch_models.py --hardhat' to install the validator."
    )
    _validator = TrackedHardHatValidator(
        UnavailableHardHatValidator(_fallback_chain[0][1])
    )
    return _validator


def reset_hardhat_validator() -> None:
    """Drop the cached validator. Used by tests and by config reloads."""
    global _validator
    if _validator is not None:
        _validator.validator.close()
    _validator = None
    _fallback_chain.clear()


def hardhat_diagnostics() -> dict[str, Any]:
    """What the hard-hat stage actually is, for /api/diagnostics."""
    validator = resolve_hardhat_validator()
    available, reason, detail = ClipHardHatValidator.probe()
    method = validator.method
    return {
        **detail,
        **validator.info(),
        "enabled": settings.hardhat_validation_enabled,
        "method_label": METHOD_LABELS.get(method, method.upper()),
        "threshold": settings.hardhat_confidence_threshold,
        "heuristic_allowed": settings.hardhat_allow_heuristic,
        "model_available": available,
        "unavailable_reason": "" if available else reason,
        "fallback_chain": [
            {"stage": stage, "reason": text} for stage, text in _fallback_chain
        ],
        # The one line that must never be optimistic: can this deployment
        # actually certify that a helmet is a hard hat?
        "compliance_ready": method in {"model", "heuristic"},
        "warning": _compliance_warning(method),
    }


def _compliance_warning(method: str) -> str:
    if method == "model":
        return ""
    if method == "heuristic":
        return (
            "Hard-hat validation is running on the HEURISTIC fallback. It "
            "measures shape, not recognition, and is materially less reliable "
            "than the validator model."
        )
    if method == "disabled":
        return (
            "Hard-hat validation is DISABLED. Any detected head covering — a "
            "baseball cap included — is reported as a valid helmet."
        )
    return (
        "No hard-hat validator is installed, so no helmet can be confirmed as "
        "an industrial hard hat. Every helmet detection is reported as a "
        "violation. This deployment is not PPE-compliance ready."
    )
