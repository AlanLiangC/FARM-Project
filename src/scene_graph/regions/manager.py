"""Orchestrate region clustering + LLM labeling + scene_state writes."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import torch

from .clustering import compute_regions
from .labeling import RegionLabeler


class RegionManager:
    """Periodically cluster scene-graph objects into regions and label them.

    Designed to be called from the mapping loop (or offline scripts).
    Updates ``scene_state`` region fields in-place.
    """

    def __init__(
        self,
        scene_state: Dict[str, Any],
        *,
        labeler: Optional[RegionLabeler] = None,
        distance_threshold_m: float = 2.0,
        min_cluster_size: int = 2,
        min_objects_for_labeling: int = 3,
        covisibility_required: bool = True,
        min_covisibility_weight: float = 0.0,
        max_region_diameter_m: float = 5.0,
    ) -> None:
        self.scene_state = scene_state
        self._labeler = labeler
        self._distance_threshold_m = distance_threshold_m
        self._min_cluster_size = min_cluster_size
        self._min_objects_for_labeling = min_objects_for_labeling
        self._covisibility_required = covisibility_required
        self._min_covisibility_weight = min_covisibility_weight
        self._max_region_diameter_m = max_region_diameter_m
        self._prev_fingerprints: Dict[int, FrozenSet[int]] = {}

    def update_regions(self) -> bool:
        """Re-cluster and re-label changed regions.

        Returns True if region assignments changed.
        """
        clusters = compute_regions(
            self.scene_state,
            distance_threshold_m=self._distance_threshold_m,
            min_cluster_size=self._min_cluster_size,
            covisibility_required=self._covisibility_required,
            min_covisibility_weight=self._min_covisibility_weight,
            max_region_diameter_m=self._max_region_diameter_m,
        )

        if not clusters:
            return self._write_empty()

        means = self.scene_state.get("means")
        if not isinstance(means, torch.Tensor):
            return self._write_empty()
        means_np = means.detach().cpu().numpy().astype(np.float32)

        old_labels = self.scene_state.get("region_labels", [])
        old_lists = self.scene_state.get("region_object_lists", [])
        old_confidences = self.scene_state.get("region_label_confidence", [])

        new_labels: List[str] = []
        new_confidences: List[float] = []
        new_centroids: List[List[float]] = []

        for cluster in clusters:
            centroid = means_np[cluster].mean(axis=0).tolist()
            new_centroids.append(centroid)

            new_fp = frozenset(cluster)
            matched_old = self._match_old_cluster(new_fp, old_lists)

            if matched_old is not None and not self._needs_relabel(new_fp, matched_old, old_lists):
                idx = matched_old
                new_labels.append(old_labels[idx] if idx < len(old_labels) else "room")
                new_confidences.append(old_confidences[idx] if idx < len(old_confidences) else 0.0)
            else:
                label, confidence = self._label_cluster(cluster)
                new_labels.append(label)
                new_confidences.append(confidence)

        n = int(means_np.shape[0])
        region_ids = [-1] * n
        for ri, cluster in enumerate(clusters):
            for obj_idx in cluster:
                if obj_idx < n:
                    region_ids[obj_idx] = ri

        self.scene_state["region_ids"] = region_ids
        self.scene_state["region_labels"] = new_labels
        self.scene_state["region_object_lists"] = [sorted(c) for c in clusters]
        self.scene_state["region_centroids"] = new_centroids
        self.scene_state["region_label_confidence"] = new_confidences
        self.scene_state["region_version"] = int(self.scene_state.get("region_version", 0) or 0) + 1

        self._prev_fingerprints = {
            ri: frozenset(c) for ri, c in enumerate(clusters)
        }

        return True

    def _write_empty(self) -> bool:
        old_labels = self.scene_state.get("region_labels", [])
        n = 0
        means = self.scene_state.get("means")
        if isinstance(means, torch.Tensor):
            n = int(means.shape[0])
        self.scene_state["region_ids"] = [-1] * n
        self.scene_state["region_labels"] = []
        self.scene_state["region_object_lists"] = []
        self.scene_state["region_centroids"] = []
        self.scene_state["region_label_confidence"] = []
        changed = bool(old_labels)
        if changed:
            self.scene_state["region_version"] = int(self.scene_state.get("region_version", 0) or 0) + 1
        return changed

    def _match_old_cluster(
        self,
        new_fp: FrozenSet[int],
        old_lists: List[List[int]],
    ) -> Optional[int]:
        """Find the old cluster index whose membership best matches new_fp."""
        best_idx = None
        best_overlap = 0
        for i, old_list in enumerate(old_lists):
            old_fp = frozenset(old_list)
            overlap = len(new_fp & old_fp)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_idx is not None and best_overlap >= len(new_fp) * 0.5:
            return best_idx
        return None

    def _needs_relabel(
        self,
        new_fp: FrozenSet[int],
        old_idx: int,
        old_lists: List[List[int]],
    ) -> bool:
        """Check if a cluster changed enough to warrant re-labeling."""
        old_fp = frozenset(old_lists[old_idx]) if old_idx < len(old_lists) else frozenset()
        if not old_fp:
            return True
        added = new_fp - old_fp
        return len(added) > 0.3 * len(old_fp)

    def _label_cluster(self, cluster: List[int]) -> Tuple[str, float]:
        """Get a label for the cluster, using the LLM if available."""
        captions = self.scene_state.get("object_caption", [])
        categories = []
        det_conf = self.scene_state.get("object_detection_category_conf", [])

        obj_captions: List[str] = []
        obj_categories: List[str] = []
        for idx in cluster:
            cap = captions[idx] if idx < len(captions) else ""
            obj_captions.append(cap or "unknown object")

            cat = ""
            if idx < len(det_conf) and isinstance(det_conf[idx], dict) and det_conf[idx]:
                cat = max(det_conf[idx], key=det_conf[idx].get)
            obj_categories.append(cat)

        if len(obj_captions) < self._min_objects_for_labeling:
            return "room", 0.1

        if self._labeler is not None:
            return self._labeler.label_region(obj_captions, obj_categories)

        return RegionLabeler._fallback_label(obj_captions, obj_categories), 0.2
