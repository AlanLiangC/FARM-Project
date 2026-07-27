"""Bridge SceneGraphRetriever results to ReferIt3D's PredictedObject contract.

The retriever returns a list of clusters; each cluster has ``candidate_objects``
ranked by their fused retrieval score plus a cluster-level score. ReferIt3D
evaluates a *flat* ranked list of object_ids with axis-aligned bboxes, which we
extract from the underlying SceneGraphProcessing's ``means_xyz + cov6`` via
:func:`scene_graph.eval.referit3d.matching.gaussian_aabb`.

Usage:

    from scene_graph.llm_utils import EmbedInterface
    from scene_graph.retrieval.scene_graph_retriever import SceneGraphRetriever
    from scene_graph.eval.referit3d import (
        Utterance, predict_for_utterance, ScenePredictor,
    )

    retriever = SceneGraphRetriever.from_scene_state(
        "/data/out/scannet/scene0046_02.pt",
        embedder=EmbedInterface(verbose=False),
    )
    predictor = ScenePredictor(retriever)
    ranked = predictor.predict(utterance, k_sigma=2.5)

The predictor caches the per-object cov6 lookup once per scene so repeated
``predict()`` calls within the same scene are O(retrieve) + O(K).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .matching import PredictedObject, gaussian_aabb, voxel_cloud_aabb


@dataclass
class _ObjectGeometry:
    """Per-object geometry pulled from a SceneGraphProcessing instance."""

    object_id: int  # int form of the processor's object_ids[i] (the ScanNet instance id)
    mean: np.ndarray  # (3,) float32
    cov6: np.ndarray  # (6,) float32; may be all-zero if cov6 wasn't available
    voxel_keys: Optional[np.ndarray] = None  # (K,) int64 bit-packed, or None if absent
    voxel_level: int = 0


class ScenePredictor:
    """Wraps a SceneGraphRetriever for one already-loaded scene.

    Builds a {object_id_int: geometry} cache from the retriever's processor,
    then translates every retrieve(query) result into a flat
    List[PredictedObject] with AABBs derived via gaussian_aabb.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        k_sigma: float = 2.5,
        score_field: str = "final_retrieval_score",
        cluster_score_field: str = "cluster_score",
        max_predictions: Optional[int] = None,
        prefer_voxel_aabb: bool = True,
        voxel_sor_k: int = 8,
        voxel_sor_alpha: float = 2.0,
        aabb_mode: str = "voxel",
        hybrid_sparse_n_thresh: int = 30,
        hybrid_volume_ratio_thresh: float = 0.3,
    ) -> None:
        """``aabb_mode`` selects how the AABB is computed for each object:

        - ``voxel``    : voxel_cloud_aabb with SOR (current default — tight,
                          can under-cover when voxel coverage is sparse).
        - ``voxel_no_sor`` : voxel AABB without statistical outlier removal
                          (larger, captures full observed extent).
        - ``gaussian`` : k_sigma Gaussian AABB only — uses Gaussian moments
                          which capture the spread implied by all observations.
        - ``union``    : per-axis max of voxel-AABB and Gaussian-AABB
                          (most expansive — recovers under-coverage but can
                          over-extend).
        - ``hybrid``   : voxel AABB by default; fall back to Gaussian when
                          the voxel cloud has fewer than
                          ``hybrid_sparse_n_thresh`` cells *or* the voxel
                          AABB volume is below
                          ``hybrid_volume_ratio_thresh × Gaussian volume``.
        """
        self.retriever = retriever
        self.k_sigma = float(k_sigma)
        self.score_field = score_field
        self.cluster_score_field = cluster_score_field
        self.max_predictions = max_predictions
        self.prefer_voxel_aabb = bool(prefer_voxel_aabb)
        self.voxel_sor_k = int(voxel_sor_k)
        self.voxel_sor_alpha = float(voxel_sor_alpha)
        if aabb_mode not in ("voxel", "voxel_no_sor", "gaussian", "union", "hybrid"):
            raise ValueError(f"unknown aabb_mode: {aabb_mode}")
        self.aabb_mode = str(aabb_mode)
        self.hybrid_sparse_n_thresh = int(hybrid_sparse_n_thresh)
        self.hybrid_volume_ratio_thresh = float(hybrid_volume_ratio_thresh)
        self._geo_by_id: Dict[int, _ObjectGeometry] = self._build_geometry_cache()

    def _build_geometry_cache(self) -> Dict[int, _ObjectGeometry]:
        proc = getattr(self.retriever, "_processor", None)
        if proc is None:
            return {}
        means = np.asarray(getattr(proc, "means_xyz", np.empty((0, 3))), dtype=np.float32)
        cov6 = np.asarray(getattr(proc, "cov6", np.empty((0, 6))), dtype=np.float32)
        object_ids = getattr(proc, "object_ids", []) or []
        voxel_keys_list: list = list(getattr(proc, "object_voxel_keys", []) or [])
        voxel_levels: list = list(getattr(proc, "object_voxel_levels", []) or [])
        n = len(object_ids)
        out: Dict[int, _ObjectGeometry] = {}
        for i in range(n):
            try:
                oid_int = int(object_ids[i])
            except (TypeError, ValueError):
                continue
            mean = means[i] if i < means.shape[0] else np.zeros(3, dtype=np.float32)
            row_cov6 = cov6[i] if i < cov6.shape[0] else np.zeros(6, dtype=np.float32)
            keys_i = voxel_keys_list[i] if i < len(voxel_keys_list) else None
            level_i = int(voxel_levels[i]) if i < len(voxel_levels) else 0
            if isinstance(keys_i, np.ndarray) and keys_i.size > 0:
                vox_keys: Optional[np.ndarray] = keys_i.astype(np.int64, copy=False)
            else:
                vox_keys = None
            out[oid_int] = _ObjectGeometry(
                object_id=oid_int,
                mean=mean,
                cov6=row_cov6,
                voxel_keys=vox_keys,
                voxel_level=level_i,
            )
        return out

    def _compute_bbox(self, geo: "_ObjectGeometry") -> tuple[np.ndarray, np.ndarray]:
        """Resolve the AABB for one object using ``self.aabb_mode``."""
        gauss_min, gauss_max = gaussian_aabb(geo.mean, geo.cov6, k_sigma=self.k_sigma)
        has_voxels = geo.voxel_keys is not None and geo.voxel_keys.size > 0
        voxel_box: Optional[tuple[np.ndarray, np.ndarray]] = None
        voxel_box_no_sor: Optional[tuple[np.ndarray, np.ndarray]] = None
        if has_voxels:
            voxel_box = voxel_cloud_aabb(
                geo.voxel_keys, geo.voxel_level,
                sor_k=self.voxel_sor_k, sor_alpha=self.voxel_sor_alpha,
            )
            voxel_box_no_sor = voxel_cloud_aabb(
                geo.voxel_keys, geo.voxel_level, sor_k=0,
            )

        mode = self.aabb_mode
        if mode == "gaussian":
            return gauss_min, gauss_max
        if mode == "voxel_no_sor":
            return voxel_box_no_sor if voxel_box_no_sor is not None else (gauss_min, gauss_max)
        if mode == "voxel":
            return voxel_box if voxel_box is not None else (gauss_min, gauss_max)
        if mode == "union":
            if voxel_box is None:
                return gauss_min, gauss_max
            v_min, v_max = voxel_box
            return np.minimum(v_min, gauss_min), np.maximum(v_max, gauss_max)
        if mode == "hybrid":
            # Use Gaussian when (a) no voxels, (b) voxel cloud sparse, or
            # (c) voxel volume far below the moment-implied volume.
            if voxel_box is None or geo.voxel_keys is None:
                return gauss_min, gauss_max
            n_vox = int(geo.voxel_keys.size)
            v_min, v_max = voxel_box
            v_vol = float(np.prod(np.maximum(v_max - v_min, 1e-9)))
            g_vol = float(np.prod(np.maximum(gauss_max - gauss_min, 1e-9)))
            sparse_n = n_vox < self.hybrid_sparse_n_thresh
            sparse_vol = (g_vol > 1e-9) and (v_vol < self.hybrid_volume_ratio_thresh * g_vol)
            if sparse_n or sparse_vol:
                return gauss_min, gauss_max
            return v_min, v_max
        # Fallback (shouldn't reach).
        return gauss_min, gauss_max

    @property
    def n_objects(self) -> int:
        return len(self._geo_by_id)

    def _predicted_object(self, candidate: Dict[str, Any], cluster_idx: int) -> Optional[PredictedObject]:
        raw_oid = candidate.get("object_id")
        try:
            oid_int = int(raw_oid)
        except (TypeError, ValueError):
            return None
        geo = self._geo_by_id.get(oid_int)
        if geo is None:
            # The retriever produced an object id we have no geometry for;
            # skip rather than fabricate a degenerate bbox.
            return None
        bbox_min, bbox_max = self._compute_bbox(geo)
        score = candidate.get(self.score_field)
        if score is None:
            score = candidate.get("rerank_score_normalized") or candidate.get("rerank_score") or 0.0
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        caption = str(candidate.get("caption") or candidate.get("object_caption") or "")
        label = caption.split(",")[0].split(".")[0].strip()[:64] or None
        region_label = candidate.get("region_label")
        return PredictedObject(
            object_id=oid_int,
            score=score_f,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            label=label,
            caption=caption or None,
            region_label=str(region_label) if region_label else None,
        )

    def predict(
        self,
        query: str,
        *,
        max_predictions: Optional[int] = None,
        retrieve_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[PredictedObject]:
        """Run retrieve(query) and flatten clusters → ranked PredictedObject list.

        Order: cluster c's k-th candidate appears before cluster (c+1)'s first
        candidate. Cap with ``max_predictions`` (defaults to the constructor
        value, then to "all").
        """
        kwargs = dict(retrieve_kwargs or {})
        result = self.retriever.retrieve(query, **kwargs) or {}
        clusters = result.get("clusters") or []
        cap = max_predictions if max_predictions is not None else self.max_predictions

        ranked: List[PredictedObject] = []
        seen_oids: set[int] = set()
        for ci, cluster in enumerate(clusters):
            cands = cluster.get("candidate_objects") or []
            for cand in cands:
                pred = self._predicted_object(cand, ci)
                if pred is None:
                    continue
                if pred.object_id in seen_oids:
                    continue
                seen_oids.add(pred.object_id)
                ranked.append(pred)
                if cap is not None and len(ranked) >= cap:
                    return ranked
        return ranked


def predict_for_utterance(
    predictor: ScenePredictor,
    utterance: Any,
    *,
    max_predictions: Optional[int] = None,
) -> List[PredictedObject]:
    """Convenience: run predictor on ``utterance.utterance``."""
    return predictor.predict(getattr(utterance, "utterance"), max_predictions=max_predictions)


def ranked_to_dicts(ranked: Sequence[PredictedObject]) -> List[Dict[str, Any]]:
    """Serialize a ranked PredictedObject list into the JSON shape we persist."""
    out: List[Dict[str, Any]] = []
    for r, pred in enumerate(ranked, start=1):
        out.append({
            "rank": r,
            "object_id": int(pred.object_id),
            "score": float(pred.score),
            "bbox_min": [float(x) for x in pred.bbox_min.tolist()],
            "bbox_max": [float(x) for x in pred.bbox_max.tolist()],
            "label": pred.label,
            "caption": pred.caption,
            "region_label": pred.region_label,
        })
    return out
