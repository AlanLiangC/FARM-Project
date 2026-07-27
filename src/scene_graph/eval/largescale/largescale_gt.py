"""GT instance loader for largescale eval — analogue of ``scannet_gt``.

Once Stage E exports per-scene GT to ``gt_instances/<scene_id>.npz``, this
module loads it back into the existing
:class:`scene_graph.eval.referit3d.scannet_gt.GTInstance` shape so the
existing matcher/scorer Just Work.

NPZ layout (mirrors ``scannet_gt._save_cache``):
- ``instance_ids``: int64 (N,)
- ``labels``: object array of strings (N,)
- ``bbox_min``: float32 (N, 3)
- ``bbox_max``: float32 (N, 3)
- ``n_vertices``: int64 (N,) — set to ``n_voxels`` from the annotation; not
  used by the matcher, but the field is part of GTInstance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from scene_graph.eval.referit3d.scannet_gt import GTInstance


def gt_dir(eval_root: Path | str, dataset: str) -> Path:
    return Path(eval_root) / dataset / "_gt_instances"


def gt_path(eval_root: Path | str, dataset: str, scene_id: str) -> Path:
    return gt_dir(eval_root, dataset) / f"{scene_id}.npz"


def save_scene_gt(
    path: Path | str,
    gt: Dict[int, GTInstance],
) -> None:
    """Save ``{instance_id: GTInstance}`` to NPZ. Mirrors scannet_gt's writer."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not gt:
        # Still write a (possibly empty) npz so the runner doesn't crash.
        np.savez(
            p,
            instance_ids=np.array([], dtype=np.int64),
            labels=np.array([], dtype=object),
            bbox_min=np.zeros((0, 3), dtype=np.float32),
            bbox_max=np.zeros((0, 3), dtype=np.float32),
            n_vertices=np.array([], dtype=np.int64),
        )
        return
    instance_ids = np.array(sorted(gt.keys()), dtype=np.int64)
    labels = np.array([gt[int(i)].label for i in instance_ids], dtype=object)
    bbox_min = np.stack([gt[int(i)].bbox_min for i in instance_ids]).astype(np.float32)
    bbox_max = np.stack([gt[int(i)].bbox_max for i in instance_ids]).astype(np.float32)
    n_vertices = np.array([gt[int(i)].n_vertices for i in instance_ids], dtype=np.int64)
    np.savez(
        p,
        instance_ids=instance_ids,
        labels=labels,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_vertices=n_vertices,
    )


def load_scene_gt(
    eval_root: Path | str,
    dataset: str,
    scene_id: str,
) -> Dict[int, GTInstance]:
    """Return ``{instance_id: GTInstance}`` for one largescale scene."""
    p = gt_path(eval_root, dataset, scene_id)
    if not p.exists():
        raise FileNotFoundError(f"largescale GT not found: {p}")
    with np.load(p, allow_pickle=True) as data:
        ids = data["instance_ids"]
        labels = data["labels"]
        bbox_min = data["bbox_min"]
        bbox_max = data["bbox_max"]
        n_vertices = data["n_vertices"]
    return {
        int(ids[k]): GTInstance(
            instance_id=int(ids[k]),
            label=str(labels[k]),
            bbox_min=np.asarray(bbox_min[k], dtype=np.float32),
            bbox_max=np.asarray(bbox_max[k], dtype=np.float32),
            n_vertices=int(n_vertices[k]),
        )
        for k in range(len(ids))
    }


def list_scenes(eval_root: Path | str, dataset: str) -> List[str]:
    d = gt_dir(eval_root, dataset)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.npz"))


__all__ = ["gt_dir", "gt_path", "load_scene_gt", "save_scene_gt", "list_scenes"]
