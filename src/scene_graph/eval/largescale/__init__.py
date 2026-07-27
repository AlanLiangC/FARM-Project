"""FARM-Scenes (large-scale) grounding benchmark support.

Ships two pieces used by the FARM-Scenes evaluation
(``scripts/eval_farm_scenes.py``):

- :mod:`.schema` — the ``annotation.json`` dataclasses (per-object 3D AABBs,
  captions, human-verification flags) as released in the FARM-Scenes dataset.
- :mod:`.largescale_gt` — ``load_scene_gt`` reading the released
  ``gt/<dataset>/_gt_instances/<scene>.npz`` GT caches into
  :class:`scene_graph.eval.referit3d.scannet_gt.GTInstance` records.

FARM-Scenes utterances are byte-compatible with
:class:`scene_graph.eval.referit3d.Utterance`; the `dataset` field takes
"grandtour" / "spot" / "odin1". The human-in-the-loop annotation tooling that
produced the dataset lives in the research repo and is not part of this
release.
"""

from .schema import (
    AnchorFrame,
    AnnotatorClick,
    AnnotationV1,
    Expression,
    ObjectRecord,
    SCHEMA_VERSION,
    load_annotation,
    save_annotation,
)

__all__ = [
    "AnchorFrame",
    "AnnotatorClick",
    "AnnotationV1",
    "Expression",
    "ObjectRecord",
    "SCHEMA_VERSION",
    "load_annotation",
    "save_annotation",
]
