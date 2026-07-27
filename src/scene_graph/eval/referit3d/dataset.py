"""Loaders for the ReferIt3D referring-expression benchmark.

ReferIt3D = NR3D (natural human references) + SR3D+ (programmatic spatial
references), both grounded in ScanNet scenes. The CSVs ship from the upstream
ReferIt3D repo; we read them directly and produce :class:`Utterance` records
suitable for the runner.

Each row identifies a target ScanNet instance (``target_id`` indexes into the
scene's ``aggregation.json`` ``segGroups``) and a small set of same-class
distractors. The utterance disambiguates the target.

Defaults assume the container layout used throughout ``EVALUATION.md``:
- NR3D / SR3D+ CSVs at ``$REFERIT3D_DIR/{nr3d,sr3d+}.csv`` (container
  default ``/data/_eval/referit3d``).
- ScanNet scans at ``$SCANNET_SCANS_DIR`` (container default ``/data/scans``).
- ScanNet v2 val split file at ``$SCANNETV2_VAL_TXT`` (container default
  ``/data/_eval/scannet_v2_val.txt``).
"""

from __future__ import annotations

import ast
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

REFERIT3D_DIR_ENV = "REFERIT3D_DIR"
SCANNET_SCANS_DIR_ENV = "SCANNET_SCANS_DIR"
SCANNETV2_VAL_TXT_ENV = "SCANNETV2_VAL_TXT"


def default_referit3d_dir() -> Path:
    """Resolve the directory holding ``nr3d.csv`` and ``sr3d+.csv``.

    Order: ``$REFERIT3D_DIR`` → container default ``/data/_eval/referit3d``.
    """
    env = os.environ.get(REFERIT3D_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return Path("/data/_eval/referit3d")


def default_scans_dir() -> Path:
    """Resolve the ScanNet ``scans/`` directory.

    Order: ``$SCANNET_SCANS_DIR`` → container default ``/data/scans``.
    """
    env = os.environ.get(SCANNET_SCANS_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return Path("/data/scans")


def default_val_split_path() -> Path:
    """Resolve ``scannetv2_val.txt`` (from the ScanNet benchmark repo).

    Order: ``$SCANNETV2_VAL_TXT`` → container default
    ``/data/_eval/scannet_v2_val.txt``.
    """
    env = os.environ.get(SCANNETV2_VAL_TXT_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return Path("/data/_eval/scannet_v2_val.txt")


def _to_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _parse_int_list(value: str) -> List[int]:
    """Parse a string like ``"[0, 1, 3]"`` into ``[0, 1, 3]``."""
    s = (value or "").strip()
    if not s or s == "[]":
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [int(x) for x in parsed]


def _parse_str_list(value: str) -> List[str]:
    s = (value or "").strip()
    if not s or s == "[]":
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in parsed]


@dataclass(frozen=True)
class Utterance:
    """One referring expression to evaluate."""

    uid: str                         # 'nr3d_<assignmentid>' or 'sr3d_<row_idx>'
    dataset: str                     # 'nr3d' | 'sr3d'
    scan_id: str                     # 'scene0264_01'
    target_id: int                   # ScanNet aggregation instance id
    distractor_ids: List[int]        # same-class distractors in this scene
    instance_type: str               # GT class name, e.g. 'office chair'
    utterance: str
    mentions_target_class: bool
    # SR3D-only:
    reference_type: Optional[str] = None       # 'closest' | 'left' | ...
    coarse_reference_type: Optional[str] = None
    anchor_ids: Optional[List[int]] = None
    anchors_types: Optional[List[str]] = None
    # NR3D-only language flags:
    uses_object_lang: Optional[bool] = None
    uses_spatial_lang: Optional[bool] = None
    uses_color_lang: Optional[bool] = None
    uses_shape_lang: Optional[bool] = None

    @property
    def n_distractors(self) -> int:
        return len(self.distractor_ids)

    @property
    def difficulty(self) -> str:
        """Standard ReferIt3D split: easy = ≤2 distractors, hard = ≥3."""
        return "hard" if self.n_distractors >= 3 else "easy"


def load_nr3d(path: Optional[Path] = None) -> List[Utterance]:
    """Load NR3D utterances."""
    p = Path(path) if path else default_referit3d_dir() / "nr3d.csv"
    out: List[Utterance] = []
    with p.open() as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            uid = f"nr3d_{row['assignmentid']}"
            target_id = int(row["target_id"])
            # NR3D's distractors come from the stimulus_id, e.g.
            # ``scene0525_00-plant-5-9-10-11-12-62`` → all-instances=[9,10,11,12,62]
            # with target=9; distractors are the others.
            distractor_ids = _stimulus_distractors(row.get("stimulus_id", ""), target_id)
            out.append(
                Utterance(
                    uid=uid,
                    dataset="nr3d",
                    scan_id=str(row["scan_id"]),
                    target_id=target_id,
                    distractor_ids=distractor_ids,
                    instance_type=str(row["instance_type"]),
                    utterance=str(row["utterance"]),
                    mentions_target_class=bool(_to_bool(row.get("mentions_target_class"))),
                    uses_object_lang=_to_bool(row.get("uses_object_lang")),
                    uses_spatial_lang=_to_bool(row.get("uses_spatial_lang")),
                    uses_color_lang=_to_bool(row.get("uses_color_lang")),
                    uses_shape_lang=_to_bool(row.get("uses_shape_lang")),
                )
            )
    return out


def load_sr3d_plus(path: Optional[Path] = None) -> List[Utterance]:
    """Load SR3D+ utterances."""
    p = Path(path) if path else default_referit3d_dir() / "sr3d+.csv"
    out: List[Utterance] = []
    with p.open() as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            uid = f"sr3d_{idx}"
            out.append(
                Utterance(
                    uid=uid,
                    dataset="sr3d",
                    scan_id=str(row["scan_id"]),
                    target_id=int(row["target_id"]),
                    distractor_ids=_parse_int_list(row.get("distractor_ids", "")),
                    instance_type=str(row["instance_type"]),
                    utterance=str(row["utterance"]),
                    mentions_target_class=bool(_to_bool(row.get("mentions_target_class"))),
                    reference_type=row.get("reference_type") or None,
                    coarse_reference_type=row.get("coarse_reference_type") or None,
                    anchor_ids=_parse_int_list(row.get("anchor_ids", "")),
                    anchors_types=_parse_str_list(row.get("anchors_types", "")),
                )
            )
    return out


def _stimulus_distractors(stimulus_id: str, target_id: int) -> List[int]:
    """Recover distractor ids from a ReferIt3D stimulus_id.

    Format: ``<scan>-<class>-<n_total>-<id1>-<id2>-...``. The first id is the
    target; the rest are distractors.

    >>> _stimulus_distractors("scene0525_00-plant-5-9-10-11-12-62", 9)
    [10, 11, 12, 62]
    """
    parts = stimulus_id.split("-")
    if len(parts) < 4:
        return []
    try:
        ids = [int(p) for p in parts[3:]]
    except ValueError:
        return []
    return [i for i in ids if i != target_id]


def load_all(
    *,
    nr3d_path: Optional[Path] = None,
    sr3d_path: Optional[Path] = None,
) -> List[Utterance]:
    """Convenience: NR3D + SR3D+ concatenated."""
    return load_nr3d(nr3d_path) + load_sr3d_plus(sr3d_path)


def load_val_scenes(path: Optional[Path] = None) -> List[str]:
    """Read ``scannetv2_val.txt`` (one scene id per line)."""
    p = Path(path) if path else default_val_split_path()
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def list_local_scenes(scans_dir: Optional[Path] = None) -> List[str]:
    """List subdirectories of the ScanNet ``scans/`` directory."""
    d = Path(scans_dir) if scans_dir else default_scans_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("scene"))


def filter_utterances(
    utterances: Iterable[Utterance],
    *,
    allowed_scenes: Optional[Iterable[str]] = None,
) -> List[Utterance]:
    """Restrict utterances to ``allowed_scenes`` (if provided)."""
    if allowed_scenes is None:
        return list(utterances)
    allowed = set(allowed_scenes)
    return [u for u in utterances if u.scan_id in allowed]


def val_local_subset(
    *,
    nr3d_path: Optional[Path] = None,
    sr3d_path: Optional[Path] = None,
    val_split_path: Optional[Path] = None,
    scans_dir: Optional[Path] = None,
) -> List[Utterance]:
    """The headline working set: NR3D ∪ SR3D+ restricted to (val ∩ local) scenes."""
    val = set(load_val_scenes(val_split_path))
    local = set(list_local_scenes(scans_dir))
    allowed = val & local
    return filter_utterances(load_all(nr3d_path=nr3d_path, sr3d_path=sr3d_path), allowed_scenes=allowed)


def partial_scene_ids() -> List[str]:
    """The frozen 75-scene subset used for fast iteration before a full run.

    These are the scenes that were locally available before the missing-215
    ScanNet val download — preserved as a reproducible scope for A/B work.
    Pass ``--partial`` to ``scripts/run_scene_graph_referit3d.py`` /
    ``scripts/eval_referit3d_spatial.py`` to scope to this set; results get
    a ``_partial`` filename suffix so they don't get confused with full-val
    runs.
    """
    p = Path(__file__).with_name("partial_scenes.txt")
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def partial_subset(
    *,
    nr3d_path: Optional[Path] = None,
    sr3d_path: Optional[Path] = None,
) -> List[Utterance]:
    """Utterances scoped to :func:`partial_scene_ids`."""
    return filter_utterances(
        load_all(nr3d_path=nr3d_path, sr3d_path=sr3d_path),
        allowed_scenes=partial_scene_ids(),
    )


def utterances_by_scene(utterances: Iterable[Utterance]) -> Dict[str, List[Utterance]]:
    """Group utterances by ``scan_id``."""
    grouped: Dict[str, List[Utterance]] = defaultdict(list)
    for u in utterances:
        grouped[u.scan_id].append(u)
    return dict(grouped)
