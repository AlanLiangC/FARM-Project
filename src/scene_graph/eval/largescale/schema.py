"""Annotation schema for large-scale outdoor grounding (versioned).

One ``annotation.json`` per scene. Plain-dataclass round-trip; no pydantic
dep so this package imports clean on the host outside docker.

Layout on disk::

    <eval_root>/<dataset>/<scene_id>/
        annotation.json              # this schema
        frames.json                  # FrameStream index (see datasets/interfaces.py)
        cloud.npz                    # aggregated scene cloud (downsampled)
        tracks/obj_<NNNN>.npz        # per-object tracked masks (cache)
        expressions/obj_<NNNN>.json  # per-object VLM candidate cache
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

# Allowed dataset discriminator strings. The runner / scoring code is fully
# agnostic to these (the existing ReferIt3D ``Utterance.dataset`` field is a
# free-form string), but we lock the set so the export step rejects typos.
ALLOWED_DATASETS = ("grandtour", "tartanground", "odin1", "spot")

# Canonical strata for referring-expression categorization. Stage C generates
# candidates per-stratum; aggregation reports per-stratum recall.
# - ``composite`` is the ReferIt3D/IRef-VLA-style one-sentence expression that
#   bundles class name + attributes + a spatial relation in a single utterance.
ALLOWED_STRATA = ("class_only", "attribute", "spatial", "sequential", "functional", "composite")


@dataclass
class AnnotatorClick:
    """One pos/neg point prompt the annotator gave to the tracker."""

    frame_id: str
    camera: str
    x: int
    y: int
    label: int  # 1 = positive, 0 = negative


@dataclass
class AnchorFrame:
    """A frame where the tracked object is visible.

    ``view_quality`` is a heuristic in [0,1] (e.g. mask-area / image-area
    bounded, or distance-to-camera weighted). The annotator can also flag
    one anchor as ``best_view`` for downstream rendering.
    """

    frame_id: str
    camera: str
    bbox_2d: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in image coords
    view_quality: float = 0.0
    best_view: bool = False


@dataclass
class ObjectRecord:
    """One annotated 3D instance.

    ``object_id`` is per-scene-namespace, assigned monotonically at click time
    (starts at 1). It is the integer that flows into ``Utterance.target_id``
    and ``Utterance.distractor_ids`` at export.
    """

    object_id: int
    instance_type: str
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    n_voxels: int = 0
    n_observed_frames: int = 0
    anchor_frames: List[AnchorFrame] = field(default_factory=list)
    annotator_clicks: List[AnnotatorClick] = field(default_factory=list)
    vlm_class_topk: List[str] = field(default_factory=list)
    vlm_caption: str = ""
    # Structured attribute tokens for the target (e.g. ["white", "metal",
    # "branded"]). Populated by the Gemini recaption pipeline; empty for
    # objects last touched by the older single-caption Qwen path.
    vlm_attributes: List[str] = field(default_factory=list)
    # Provenance tag for ``vlm_caption`` / ``vlm_attributes`` — e.g. the
    # Gemini model id or ``"qwen3-vl-8b"``. Lets a re-run skip objects that
    # already used the current model and lets analysis stratify by source.
    vlm_caption_source: str = ""
    track_artifact_path: Optional[str] = None  # relative path under the scene dir
    verified: bool = False
    notes: str = ""


@dataclass
class Expression:
    """One referring expression candidate (verified or otherwise).

    ``distractor_object_ids`` is the same-class-near-target set used for the
    VLM-as-judge filter and for ReferIt3D-style ``Utterance.distractor_ids``
    on export. We persist it on the expression so a re-export reproduces the
    exact comparison set even if more objects are later annotated nearby.
    """

    expression_id: str
    object_id: int  # target
    text: str
    stratum: str
    anchor_object_ids: List[int] = field(default_factory=list)
    distractor_object_ids: List[int] = field(default_factory=list)
    generator: str = "vlm"  # "vlm" | "human" | "vlm_edited"
    discriminator_score: Optional[float] = None  # None = not run yet
    verified: bool = False
    verifier_notes: str = ""


@dataclass
class AnnotationV1:
    """One scene's complete annotation state."""

    schema_version: int
    dataset: str
    scene_id: str
    annotator: str
    created_at: str  # ISO-8601
    frames_index_path: str  # relative path to frames.json
    objects: List[ObjectRecord] = field(default_factory=list)
    expressions: List[Expression] = field(default_factory=list)
    notes: str = ""

    # ---- helpers --------------------------------------------------------

    def next_object_id(self) -> int:
        return (max((o.object_id for o in self.objects), default=0)) + 1

    def next_expression_id(self) -> str:
        n = len(self.expressions)
        return f"u{n:06d}"

    def get_object(self, object_id: int) -> Optional[ObjectRecord]:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        return None

    def remove_object(self, object_id: int) -> None:
        self.objects = [o for o in self.objects if o.object_id != object_id]
        # Cascade: drop expressions tied to this object.
        self.expressions = [e for e in self.expressions if e.object_id != object_id]

    # ---- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AnnotationV1":
        v = int(raw.get("schema_version", 0))
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"annotation schema_version mismatch: got {v}, this code understands {SCHEMA_VERSION}"
            )
        return cls(
            schema_version=v,
            dataset=str(raw["dataset"]),
            scene_id=str(raw["scene_id"]),
            annotator=str(raw.get("annotator", "")),
            created_at=str(raw.get("created_at", "")),
            frames_index_path=str(raw.get("frames_index_path", "frames.json")),
            objects=[_object_from_dict(o) for o in raw.get("objects", [])],
            expressions=[_expression_from_dict(e) for e in raw.get("expressions", [])],
            notes=str(raw.get("notes", "")),
        )

    def validate(self) -> List[str]:
        """Return a list of validation problems, empty if OK."""
        problems: List[str] = []
        if self.dataset not in ALLOWED_DATASETS:
            problems.append(f"dataset {self.dataset!r} not in {ALLOWED_DATASETS}")
        seen: set[int] = set()
        for o in self.objects:
            if o.object_id in seen:
                problems.append(f"duplicate object_id {o.object_id}")
            seen.add(o.object_id)
            if o.object_id < 1:
                problems.append(f"object_id must be ≥1, got {o.object_id}")
            if any(b <= a for a, b in zip(o.bbox_min, o.bbox_max)):
                problems.append(f"object {o.object_id}: degenerate bbox {o.bbox_min}→{o.bbox_max}")
        for e in self.expressions:
            if e.stratum not in ALLOWED_STRATA:
                problems.append(f"expression {e.expression_id}: stratum {e.stratum!r} not in {ALLOWED_STRATA}")
            if self.get_object(e.object_id) is None:
                problems.append(f"expression {e.expression_id}: target object_id {e.object_id} not found")
            for d in e.distractor_object_ids:
                if self.get_object(d) is None:
                    problems.append(f"expression {e.expression_id}: distractor object_id {d} not found")
        return problems


# ---- field-by-field decoders (tuple → tuple, robust to JSON list-coercion) ----

def _tuple3(v: Any) -> Tuple[float, float, float]:
    a, b, c = v
    return (float(a), float(b), float(c))


def _bbox2_or_none(v: Any) -> Optional[Tuple[int, int, int, int]]:
    if v is None:
        return None
    a, b, c, d = v
    return (int(a), int(b), int(c), int(d))


def _click_from_dict(raw: Dict[str, Any]) -> AnnotatorClick:
    return AnnotatorClick(
        frame_id=str(raw["frame_id"]),
        camera=str(raw["camera"]),
        x=int(raw["x"]),
        y=int(raw["y"]),
        label=int(raw["label"]),
    )


def _anchor_from_dict(raw: Dict[str, Any]) -> AnchorFrame:
    return AnchorFrame(
        frame_id=str(raw["frame_id"]),
        camera=str(raw["camera"]),
        bbox_2d=_bbox2_or_none(raw.get("bbox_2d")),
        view_quality=float(raw.get("view_quality", 0.0)),
        best_view=bool(raw.get("best_view", False)),
    )


def _object_from_dict(raw: Dict[str, Any]) -> ObjectRecord:
    return ObjectRecord(
        object_id=int(raw["object_id"]),
        instance_type=str(raw["instance_type"]),
        bbox_min=_tuple3(raw["bbox_min"]),
        bbox_max=_tuple3(raw["bbox_max"]),
        n_voxels=int(raw.get("n_voxels", 0)),
        n_observed_frames=int(raw.get("n_observed_frames", 0)),
        anchor_frames=[_anchor_from_dict(a) for a in raw.get("anchor_frames", [])],
        annotator_clicks=[_click_from_dict(c) for c in raw.get("annotator_clicks", [])],
        vlm_class_topk=[str(x) for x in raw.get("vlm_class_topk", [])],
        vlm_caption=str(raw.get("vlm_caption", "")),
        vlm_attributes=[str(x) for x in raw.get("vlm_attributes", [])],
        vlm_caption_source=str(raw.get("vlm_caption_source", "")),
        track_artifact_path=raw.get("track_artifact_path"),
        verified=bool(raw.get("verified", False)),
        notes=str(raw.get("notes", "")),
    )


def _expression_from_dict(raw: Dict[str, Any]) -> Expression:
    score = raw.get("discriminator_score")
    return Expression(
        expression_id=str(raw["expression_id"]),
        object_id=int(raw["object_id"]),
        text=str(raw["text"]),
        stratum=str(raw["stratum"]),
        anchor_object_ids=[int(x) for x in raw.get("anchor_object_ids", [])],
        distractor_object_ids=[int(x) for x in raw.get("distractor_object_ids", [])],
        generator=str(raw.get("generator", "vlm")),
        discriminator_score=None if score is None else float(score),
        verified=bool(raw.get("verified", False)),
        verifier_notes=str(raw.get("verifier_notes", "")),
    )


# ---- IO -------------------------------------------------------------


def load_annotation(path: Path | str) -> AnnotationV1:
    p = Path(path)
    return AnnotationV1.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_annotation(path: Path | str, ann: AnnotationV1) -> None:
    """Atomic write — never leave the file half-written on crash."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(ann.to_dict(), indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(text)
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
