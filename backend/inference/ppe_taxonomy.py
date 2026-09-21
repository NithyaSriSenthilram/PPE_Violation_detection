"""PPE class vocabulary and its mapping onto a model's own label names.

A PPE model is only useful if the platform can say what each of its classes
*means*. Third-party PPE weights disagree wildly on naming — ``Hardhat``,
``hard_hat``, ``helmet``, ``with_helmet`` all denote the same thing, and
``head``, ``no-helmet`` and ``NO-Hardhat`` all denote its absence — so the
mapping is data, not code:

* a built-in synonym table covers the naming conventions actually in use, and
* ``PPE_CLASS_MAP`` overrides or extends it for any model that names things
  differently, without touching Python.

Every item also declares *where on a person it belongs* (:attr:`ItemSpec.band`,
a vertical fraction of the person box). That is what makes association
meaningful: a helmet lying on a bench is not a helmet being worn, and a vest
box that lines up with someone's knees belongs to whoever is standing behind
them. Adding gloves, boots, goggles, a mask or a harness to the deployment is a
matter of listing the item in ``REQUIRED_PPE`` and having a model that emits a
class for it — the specs below are already present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Literal, NamedTuple

#: Every PPE item the platform understands. Extending this list is the only
#: code change a new category needs — rules, events, association and the UI all
#: read from here.
KNOWN_ITEMS: Final[tuple[str, ...]] = (
    "helmet", "vest", "gloves", "boots", "goggles", "mask", "harness",
)


@dataclass(frozen=True, slots=True)
class ItemSpec:
    """Where an item sits on a person, and how it is named to an operator."""

    item: str
    label: str
    #: Vertical band of the person box the item is expected to occupy, as
    #: (top, bottom) fractions measured from the top of the box.
    band: tuple[float, float]
    #: Event type raised when this item is the only one missing. Items without
    #: a dedicated type fall back to the aggregate PPE_VIOLATION.
    event_type: str | None = None


ITEM_SPECS: Final[dict[str, ItemSpec]] = {
    "helmet":  ItemSpec("helmet",  "Helmet",      (0.00, 0.35), "MISSING_HELMET"),
    "vest":    ItemSpec("vest",    "Safety Vest", (0.12, 0.75), "MISSING_VEST"),
    "gloves":  ItemSpec("gloves",  "Gloves",      (0.28, 0.85)),
    "boots":   ItemSpec("boots",   "Boots",       (0.68, 1.00)),
    "goggles": ItemSpec("goggles", "Goggles",     (0.00, 0.30)),
    "mask":    ItemSpec("mask",    "Face Mask",   (0.00, 0.34)),
    "harness": ItemSpec("harness", "Harness",     (0.12, 0.80)),
}


def item_label(item: str) -> str:
    spec = ITEM_SPECS.get(item)
    return spec.label if spec else item.replace("_", " ").title()


#: Body regions, as ``(x0, y0, x1, y1)`` fractions of the person box.
#:
#: These are what a PPE box falls back to when the detector gives no usable box
#: of its own — an absence the model could only report by omission, a heuristic
#: estimate, or a "no_vest" box so large it is really just the person again.
#: They are deliberately tighter than :attr:`ItemSpec.band`: a band is the
#: generous window association will *accept* an item in, a region is where the
#: item is actually drawn.
BODY_REGIONS: Final[dict[str, tuple[float, float, float, float]]] = {
    "head":  (0.18, 0.00, 0.82, 0.30),
    "face":  (0.24, 0.02, 0.76, 0.26),
    "torso": (0.10, 0.20, 0.90, 0.70),
    "hands": (0.00, 0.42, 1.00, 0.78),
    "feet":  (0.06, 0.80, 0.94, 1.00),
}

#: Which region each item is drawn in. Keyed by canonical item, so it covers
#: both the presence and the absence class of that item — ``no-helmet`` and
#: ``helmet`` both normalise to ``helmet`` and both belong on the head.
#: Extending it is data, not code: a new item needs an entry here, an entry in
#: ITEM_SPECS, and a model that emits the class.
PPE_BODY_REGIONS: Final[dict[str, str]] = {
    "helmet":  "head",
    "goggles": "face",
    "mask":    "face",
    "vest":    "torso",
    "harness": "torso",
    "gloves":  "hands",
    "boots":   "feet",
}

#: Region for an item nobody has mapped yet. The torso is the least wrong
#: guess: it is where most protective equipment is worn, and it never places a
#: box outside the person.
DEFAULT_REGION = "torso"


def region_for(item: str) -> str:
    """Name of the body region an item is drawn in."""
    return PPE_BODY_REGIONS.get(item, DEFAULT_REGION)


def body_region_box(
    person_bbox: tuple[float, float, float, float], item: str
) -> tuple[float, float, float, float]:
    """The region of ``person_bbox`` where ``item`` belongs, in pixels."""
    px1, py1, px2, py2 = (float(v) for v in person_bbox)
    width, height = px2 - px1, py2 - py1
    fx0, fy0, fx1, fy1 = BODY_REGIONS[region_for(item)]
    return (
        px1 + width * fx0,
        py1 + height * fy0,
        px1 + width * fx1,
        py1 + height * fy1,
    )


class ClassRole(NamedTuple):
    """What one of the model's classes means."""

    kind: Literal["person", "ppe"]
    item: str | None = None
    #: For ``kind == "ppe"``: True marks the item present, False marks it
    #: explicitly absent (a ``no_helmet`` / bare-head class).
    present: bool = True


_SEPARATORS = re.compile(r"[-_./\\]+")
_SPACES = re.compile(r"\s+")


def normalise(name: str) -> str:
    """Canonical form of a class name: ``NO-Hardhat`` -> ``no hardhat``."""
    lowered = _SEPARATORS.sub(" ", str(name).strip().lower())
    return _SPACES.sub(" ", lowered).strip()


PERSON_SYNONYMS: Final[frozenset[str]] = frozenset(
    normalise(n) for n in (
        "person", "people", "human", "worker", "pedestrian", "man", "employee",
    )
)

#: Built-in synonyms, keyed by normalised class name.
DEFAULT_SYNONYMS: Final[dict[str, ClassRole]] = {}


def _register(item: str, present: bool, *names: str) -> None:
    for name in names:
        DEFAULT_SYNONYMS[normalise(name)] = ClassRole("ppe", item, present)


_register("helmet", True,
          "helmet", "hardhat", "hard hat", "safety helmet", "with helmet",
          "helmet on", "wearing helmet", "helm")
_register("helmet", False,
          "no helmet", "nohelmet", "no hardhat", "without helmet", "helmet off",
          "head", "bare head", "no safety helmet", "not wearing helmet")
_register("vest", True,
          "vest", "safety vest", "reflective vest", "hi vis", "high vis vest",
          "with vest", "vest on", "wearing vest", "safety jacket")
_register("vest", False,
          "no vest", "novest", "without vest", "no safety vest", "vest off",
          "not wearing vest")
_register("gloves", True, "glove", "gloves", "safety gloves", "with gloves")
_register("gloves", False, "no glove", "no gloves", "without gloves", "bare hands")
_register("boots", True, "boot", "boots", "safety boots", "safety shoes", "with boots")
_register("boots", False, "no boot", "no boots", "without boots", "no safety shoes")
_register("goggles", True,
          "goggles", "glasses", "safety glasses", "safety goggles", "eye protection",
          "glass", "with goggles")
_register("goggles", False,
          "no goggles", "no glasses", "no glass", "without goggles",
          "no eye protection")
_register("mask", True, "mask", "face mask", "respirator", "with mask", "face shield")
_register("mask", False, "no mask", "without mask", "no face mask")
_register("harness", True, "harness", "safety harness", "fall harness", "with harness")
_register("harness", False, "no harness", "without harness", "no safety harness")


class ClassMapError(ValueError):
    """Raised for a malformed ``PPE_CLASS_MAP`` entry."""


def parse_overrides(spec: str) -> dict[str, ClassRole]:
    """Parse ``PPE_CLASS_MAP``.

    Format is comma-separated ``label=role``, where role is ``person`` or
    ``item:present`` / ``item:absent``::

        PPE_CLASS_MAP=worker=person,hat=helmet:present,bare_head=helmet:absent

    A malformed entry raises rather than being skipped: a typo that silently
    drops a class would show up much later as a mysteriously absent violation.
    """
    overrides: dict[str, ClassRole] = {}
    for chunk in (spec or "").split(","):
        entry = chunk.strip()
        if not entry:
            continue
        label, _, role = entry.partition("=")
        label, role = label.strip(), role.strip().lower()
        if not label or not role:
            raise ClassMapError(f"PPE_CLASS_MAP entry {entry!r} is not 'label=role'")
        if role == "person":
            overrides[normalise(label)] = ClassRole("person")
            continue
        if role in {"ignore", "none", "-"}:
            continue
        item, _, state = role.partition(":")
        item = item.strip()
        state = (state or "present").strip()
        if item not in ITEM_SPECS:
            raise ClassMapError(
                f"PPE_CLASS_MAP entry {entry!r} names unknown item {item!r}; "
                f"known items are {', '.join(ITEM_SPECS)}"
            )
        if state not in {"present", "absent"}:
            raise ClassMapError(
                f"PPE_CLASS_MAP entry {entry!r}: state must be 'present' or "
                f"'absent', got {state!r}"
            )
        overrides[normalise(label)] = ClassRole("ppe", item, state == "present")
    return overrides


@dataclass(slots=True)
class PPEClassMap:
    """A model's class list, resolved to meanings."""

    labels: list[str]
    roles: dict[str, ClassRole]
    #: Class names that matched nothing — reported so an operator can add them
    #: to PPE_CLASS_MAP rather than wondering why an item never fires.
    unmapped: list[str]

    @classmethod
    def build(cls, labels: list[str], overrides: str | dict[str, ClassRole] = "") -> PPEClassMap:
        table = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
        roles: dict[str, ClassRole] = {}
        unmapped: list[str] = []
        for label in labels:
            key = normalise(label)
            role = table.get(key)
            if role is None:
                if key in PERSON_SYNONYMS:
                    role = ClassRole("person")
                else:
                    role = DEFAULT_SYNONYMS.get(key)
            if role is None:
                unmapped.append(label)
                continue
            roles[key] = role
        return cls(labels=list(labels), roles=roles, unmapped=unmapped)

    # ── lookups ──────────────────────────────────────────────────────────
    def role_of(self, label: str) -> ClassRole | None:
        return self.roles.get(normalise(label))

    def is_person(self, label: str) -> bool:
        role = self.role_of(label)
        return role is not None and role.kind == "person"

    def covered_items(self) -> set[str]:
        """Items this model can say anything at all about."""
        return {r.item for r in self.roles.values() if r.kind == "ppe" and r.item}

    def has_presence_class(self, item: str) -> bool:
        return any(
            r.kind == "ppe" and r.item == item and r.present for r in self.roles.values()
        )

    def has_absence_class(self, item: str) -> bool:
        """True when the model names the *absence* of an item explicitly.

        Without such a class, absence can only be inferred from the item never
        being found on a person the detector clearly saw — a weaker signal that
        is reported with lower confidence.
        """
        return any(
            r.kind == "ppe" and r.item == item and not r.present
            for r in self.roles.values()
        )

    def describe(self) -> dict[str, Any]:
        """Diagnostics view of the mapping."""
        return {
            "classes": list(self.labels),
            "class_count": len(self.labels),
            "mapping": {
                label: (
                    "person" if (r := self.roles[normalise(label)]).kind == "person"
                    else f"{r.item}:{'present' if r.present else 'absent'}"
                )
                for label in self.labels
                if normalise(label) in self.roles
            },
            "covered_items": sorted(self.covered_items()),
            "items_with_absence_class": sorted(
                i for i in self.covered_items() if self.has_absence_class(i)
            ),
            "unmapped_classes": list(self.unmapped),
        }
