"""ReferIt3D referring-expression benchmark integration.

NR3D + SR3D+ utterances grounded in ScanNet scenes. Pipeline:

1. Reconstruct each ScanNet scene with the offline runner — see
   ``scripts/run_scene_graph_referit3d.py``.
2. For each utterance, retrieve a ranked list of objects from the saved
   ``scene_state.pt`` via ``SceneGraphRetriever``; convert each object's
   3D Gaussian to an axis-aligned bbox via :mod:`.matching`.
3. Match top-K predictions to the GT target instance loaded from
   :mod:`.scannet_gt`; report Acc@1@IoU={0.25,0.5}, Recall@K, MRR, median rank.

Day-1 modules (loaders + GT + matching) have no torch / scene_graph runtime
deps — they can be imported and smoke-tested on the host without docker.
"""

from .dataset import (
    Utterance,
    default_referit3d_dir,
    default_scans_dir,
    default_val_split_path,
    filter_utterances,
    list_local_scenes,
    load_all,
    load_nr3d,
    load_sr3d_plus,
    load_val_scenes,
    partial_scene_ids,
    partial_subset,
    utterances_by_scene,
    val_local_subset,
)
from .matching import (
    MatchResult,
    PredictedObject,
    aabb_volume,
    gaussian_aabb,
    iou_3d_aabb,
    match_predictions_to_target,
)
from .metrics import (
    ANY_OVERLAP_THRESHOLD,
    PRIMARY_THRESHOLD,
    VISIBLE_MASK_PRIMARY_THRESHOLD,
    UtteranceScore,
    aggregate,
    format_overall_table,
    score_predictions,
    score_utterance,
    score_utterance_visible_mask,
    write_metrics_json,
)
from .retrieval_adapter import ScenePredictor, predict_for_utterance, ranked_to_dicts
from .runner import RunnerConfig, discover_scene_states, run_predictions, utterance_to_record
from .scannet_gt import GTInstance, GTInstancePoints, build_caches, load_scene_gt, load_scene_gt_points
from .scoring import metrics_path_for, score_and_persist

__all__ = [
    "Utterance",
    "default_referit3d_dir",
    "default_scans_dir",
    "default_val_split_path",
    "filter_utterances",
    "list_local_scenes",
    "load_all",
    "load_nr3d",
    "load_sr3d_plus",
    "load_val_scenes",
    "partial_scene_ids",
    "partial_subset",
    "utterances_by_scene",
    "val_local_subset",
    "MatchResult",
    "PredictedObject",
    "aabb_volume",
    "gaussian_aabb",
    "iou_3d_aabb",
    "match_predictions_to_target",
    "GTInstance",
    "build_caches",
    "load_scene_gt",
    "ScenePredictor",
    "predict_for_utterance",
    "ranked_to_dicts",
    "RunnerConfig",
    "discover_scene_states",
    "run_predictions",
    "utterance_to_record",
    "ANY_OVERLAP_THRESHOLD",
    "PRIMARY_THRESHOLD",
    "VISIBLE_MASK_PRIMARY_THRESHOLD",
    "UtteranceScore",
    "aggregate",
    "format_overall_table",
    "score_predictions",
    "score_utterance",
    "score_utterance_visible_mask",
    "write_metrics_json",
    "metrics_path_for",
    "score_and_persist",
    "GTInstancePoints",
    "load_scene_gt_points",
]
