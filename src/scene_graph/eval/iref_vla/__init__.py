"""IRef-VLA referring-expression benchmark integration (HM3D split).

IRef-VLA (Zhang et al., ICRA 2025; https://arxiv.org/abs/2503.17406) bundles
referential statements grounded in scenes from six 3D datasets. We use the
**HM3D split** (140 scenes, 1991 regions — multi-room is the norm) to
benchmark the scene-graph pipeline on multi-room indoor environments where
HM3D meshes are rendered through habitat-sim into RGBD trajectories that
``StreamingMapper`` ingests.

Pipeline (mirrors ``eval/referit3d/`` end-to-end):

1. Render an HM3D scene to NPZ frames via ``scripts/render_hm3d_trajectory.py``
   (runs in the host's apg micromamba env so habitat-sim 0.2.5 is available
   without modifying our docker image).
2. Reconstruct the scene graph by feeding the NPZ frames through the offline
   driver — see ``scripts/run_scene_graph_iref_vla.py``.
3. For each statement, retrieve a ranked list of objects from the saved
   ``scene_state.pt`` via ``SceneGraphRetriever``; convert each object's
   3D Gaussian (or sparse voxel cloud, when available) to an axis-aligned
   bbox via :mod:`scene_graph.eval.referit3d.matching`.
4. Match top-K predictions to the GT target loaded from
   :mod:`.iref_vla_gt`; report Acc@1@IoU={0.25, 0.5}, Recall@K, MRR,
   median rank, with per-relation / per-region / multi-room breakdowns.

Schema invariant: IRef-VLA stores object IDs as **strings** in JSON
(`"target_index": "26"`) but as **ints** in the CSVs. This module canonicalises
to ``int`` everywhere; callers should pass int IDs.
"""

from .dataset import (
    Statement,
    default_iref_vla_root,
    list_local_scenes,
    load_scene_statements,
    load_all_statements,
    multi_room_scene_ids,
    statements_by_scene,
    filter_statements,
)
from .iref_vla_gt import (
    GTInstance,
    RegionInfo,
    load_scene_objects,
    load_scene_regions,
    aabb_from_obb,
)
from .runner import (
    RunnerConfig,
    discover_scene_states,
    run_predictions,
    statement_to_record,
)
from .metrics import (
    ANY_OVERLAP_THRESHOLD,
    PRIMARY_THRESHOLD,
    VISIBLE_MASK_PRIMARY_THRESHOLD,
    StatementScore,
    aggregate,
    format_overall_table,
    score_predictions,
    score_statement,
    score_statement_visible_mask,
    write_metrics_json,
)
from .scoring import metrics_path_for, score_and_persist

__all__ = [
    "Statement",
    "default_iref_vla_root",
    "list_local_scenes",
    "load_scene_statements",
    "load_all_statements",
    "multi_room_scene_ids",
    "statements_by_scene",
    "filter_statements",
    "GTInstance",
    "RegionInfo",
    "load_scene_objects",
    "load_scene_regions",
    "aabb_from_obb",
    "RunnerConfig",
    "discover_scene_states",
    "run_predictions",
    "statement_to_record",
    "ANY_OVERLAP_THRESHOLD",
    "PRIMARY_THRESHOLD",
    "VISIBLE_MASK_PRIMARY_THRESHOLD",
    "StatementScore",
    "aggregate",
    "format_overall_table",
    "score_predictions",
    "score_statement",
    "score_statement_visible_mask",
    "write_metrics_json",
    "metrics_path_for",
    "score_and_persist",
]
