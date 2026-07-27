"""Loader for IRef-VLA referential statements (HM3D split).

IRef-VLA stores per-scene statements in
``<scene>/<scene>_referential_statements.json`` with this nested shape::

    {
      "scene_name": "<scene>",
      "regions": {
        "<region_id>": {
          "<statement text>": [
            {
              "target_index": "<int as str>",
              "target_class": "<str>",
              "target_position": [x, y, z],
              "target_size": <bbox-volume>,
              "target_colors": [<3 dominant>],
              "distractor_ids": ["<int as str>", ...],
              "relation": "<str>",
              "relation_type": "binary|...",
              "anchors": {
                "anchor_1": {"index": "<int>", "class": "<str>", ...},
                ...
              },
              "false_statements": {...}    # augmented variants
            },
            ...                            # multiple variants of the same text
          ],
          ...                              # more statement texts
        },
        ...                                # more regions
      }
    }

We flatten the nesting into one :class:`Statement` per ``(region_id,
statement_text, variant_idx)``. Object IDs are canonicalised to ``int``.

Defaults:
- IRef-VLA HM3D extracted to ``$IREF_VLA_ROOT`` (or container default
  ``/data/iref_vla/HM3D``).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

IREF_VLA_ROOT_ENV = "IREF_VLA_ROOT"


def default_iref_vla_root() -> Path:
    """Resolve the IRef-VLA HM3D dataset root.

    Order: ``$IREF_VLA_ROOT`` → container default ``/data/iref_vla/HM3D``.
    """
    env = os.environ.get(IREF_VLA_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return Path("/data/iref_vla/HM3D")


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


@dataclass(frozen=True)
class Statement:
    """One referring statement to evaluate."""

    uid: str                         # '<scene>/<region>/<text-hash>/<variant>'
    scene_id: str                    # '00238-j6fHrce9pHR'
    region_id: int                   # IRef-VLA region id within the scene
    statement: str                   # natural language statement
    target_id: int                   # IRef-VLA object id
    target_class: str                # class label of target
    distractor_ids: List[int]        # same-class distractors (per IRef-VLA)
    anchor_ids: List[int]            # ids of anchor objects in the relation
    anchor_classes: List[str]
    relation: str                    # 'above' | 'closest' | 'between' | ...
    relation_type: str               # 'binary' | 'ternary' | ...
    variant_idx: int                 # which paraphrase variant of this text
    is_false_statement: bool = False # set True for "false_statements" variants

    @property
    def n_distractors(self) -> int:
        return len(self.distractor_ids)

    @property
    def difficulty(self) -> str:
        return "hard" if self.n_distractors >= 3 else "easy"


def _scene_dir(scene_id: str, dataset_root: Optional[Path] = None) -> Path:
    return (Path(dataset_root) if dataset_root else default_iref_vla_root()) / scene_id


def _statements_path(scene_id: str, dataset_root: Optional[Path] = None) -> Path:
    return _scene_dir(scene_id, dataset_root) / f"{scene_id}_referential_statements.json"


def _region_csv_path(scene_id: str, dataset_root: Optional[Path] = None) -> Path:
    return _scene_dir(scene_id, dataset_root) / f"{scene_id}_region_result.csv"


def _object_csv_path(scene_id: str, dataset_root: Optional[Path] = None) -> Path:
    return _scene_dir(scene_id, dataset_root) / f"{scene_id}_object_result.csv"


def list_local_scenes(dataset_root: Optional[Path] = None) -> List[str]:
    """List sub-directories of the IRef-VLA HM3D root that look like scenes."""
    root = Path(dataset_root) if dataset_root else default_iref_vla_root()
    if not root.exists():
        return []
    out: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / f"{p.name}_referential_statements.json").exists():
            out.append(p.name)
    return out


def _flatten_statements(
    payload: Dict,
    scene_id: str,
    *,
    include_false_statements: bool,
) -> List[Statement]:
    """Flatten the nested regions/text/variants tree into Statement objects."""
    out: List[Statement] = []
    regions = payload.get("regions") or {}
    if not isinstance(regions, dict):
        return out
    for region_key, region_dict in regions.items():
        try:
            region_id = int(region_key)
        except (ValueError, TypeError):
            continue
        if not isinstance(region_dict, dict):
            continue
        for statement_text, variants in region_dict.items():
            if not isinstance(variants, list):
                continue
            text_hash = _short_hash(f"{region_id}|{statement_text}")
            for v_idx, variant in enumerate(variants):
                if not isinstance(variant, dict):
                    continue
                target_id = _to_int(variant.get("target_index"))
                if target_id is None:
                    continue
                target_class = str(variant.get("target_class", "") or "")
                distractor_ids: List[int] = []
                for d in variant.get("distractor_ids") or []:
                    di = _to_int(d)
                    if di is not None:
                        distractor_ids.append(di)
                anchors = variant.get("anchors") or {}
                anchor_ids: List[int] = []
                anchor_classes: List[str] = []
                if isinstance(anchors, dict):
                    for anchor_dict in anchors.values():
                        if not isinstance(anchor_dict, dict):
                            continue
                        ai = _to_int(anchor_dict.get("index"))
                        if ai is None:
                            continue
                        anchor_ids.append(ai)
                        anchor_classes.append(str(anchor_dict.get("class", "") or ""))
                relation = str(variant.get("relation", "") or "")
                relation_type = str(variant.get("relation_type", "") or "")
                uid = f"{scene_id}/{region_id}/{text_hash}/{v_idx}"
                out.append(
                    Statement(
                        uid=uid,
                        scene_id=scene_id,
                        region_id=region_id,
                        statement=str(statement_text),
                        target_id=target_id,
                        target_class=target_class,
                        distractor_ids=distractor_ids,
                        anchor_ids=anchor_ids,
                        anchor_classes=anchor_classes,
                        relation=relation,
                        relation_type=relation_type,
                        variant_idx=v_idx,
                        is_false_statement=False,
                    )
                )
                if include_false_statements:
                    fs = variant.get("false_statements") or {}
                    if isinstance(fs, dict):
                        for fs_key, fs_text in fs.items():
                            # Skip nested anchor false-statement dicts.
                            if not isinstance(fs_text, str):
                                continue
                            fs_uid = f"{uid}/false:{fs_key}"
                            out.append(
                                Statement(
                                    uid=fs_uid,
                                    scene_id=scene_id,
                                    region_id=region_id,
                                    statement=fs_text,
                                    target_id=target_id,
                                    target_class=target_class,
                                    distractor_ids=distractor_ids,
                                    anchor_ids=anchor_ids,
                                    anchor_classes=anchor_classes,
                                    relation=relation,
                                    relation_type=relation_type,
                                    variant_idx=v_idx,
                                    is_false_statement=True,
                                )
                            )
    return out


def load_scene_statements(
    scene_id: str,
    *,
    dataset_root: Optional[Path] = None,
    include_false_statements: bool = False,
) -> List[Statement]:
    """Load all referential statements for one scene."""
    path = _statements_path(scene_id, dataset_root)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    return _flatten_statements(payload, scene_id, include_false_statements=include_false_statements)


def load_all_statements(
    *,
    dataset_root: Optional[Path] = None,
    scene_filter: Optional[Sequence[str]] = None,
    include_false_statements: bool = False,
) -> List[Statement]:
    """Load referential statements across many scenes."""
    scenes = list(scene_filter) if scene_filter is not None else list_local_scenes(dataset_root)
    out: List[Statement] = []
    for sid in scenes:
        out.extend(
            load_scene_statements(
                sid,
                dataset_root=dataset_root,
                include_false_statements=include_false_statements,
            )
        )
    return out


def _count_regions_via_csv(scene_id: str, dataset_root: Optional[Path]) -> int:
    """Count rows of region_result.csv (excluding the header)."""
    p = _region_csv_path(scene_id, dataset_root)
    if not p.exists():
        return 0
    with p.open("r", newline="") as fp:
        reader = csv.reader(fp)
        try:
            next(reader)  # header
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def multi_room_scene_ids(
    *,
    dataset_root: Optional[Path] = None,
    min_regions: int = 2,
) -> List[str]:
    """Return scenes with at least ``min_regions`` regions in region_result.csv.

    HM3D scenes from IRef-VLA average ~14 regions/scene, so this filter is mild
    in practice but excludes the rare single-room case.
    """
    out: List[str] = []
    for sid in list_local_scenes(dataset_root):
        if _count_regions_via_csv(sid, dataset_root) >= int(min_regions):
            out.append(sid)
    return out


def filter_statements(
    statements: Iterable[Statement],
    *,
    allowed_scenes: Optional[Iterable[str]] = None,
    relations: Optional[Iterable[str]] = None,
    drop_false_statements: bool = True,
) -> List[Statement]:
    """Restrict statements by scene / relation type / true-vs-false flag."""
    out: List[Statement] = []
    allowed = set(allowed_scenes) if allowed_scenes is not None else None
    rels = set(relations) if relations is not None else None
    for s in statements:
        if drop_false_statements and s.is_false_statement:
            continue
        if allowed is not None and s.scene_id not in allowed:
            continue
        if rels is not None and s.relation not in rels:
            continue
        out.append(s)
    return out


def statements_by_scene(statements: Iterable[Statement]) -> Dict[str, List[Statement]]:
    """Group statements by scene_id."""
    grouped: Dict[str, List[Statement]] = defaultdict(list)
    for s in statements:
        grouped[s.scene_id].append(s)
    return dict(grouped)
