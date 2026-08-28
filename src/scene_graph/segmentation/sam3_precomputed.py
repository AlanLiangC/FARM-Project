"""SAM3.1 precomputed-mask backend for FARM's RGB-D mapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dino import DINOFeaturesExtractor
from .yoloe import YOLOESegmenter


class SAM3PrecomputedSegmenter(YOLOESegmenter):
    """Read stable SAM3 video tracks and compute FARM-compatible 3-D fields.

    Inheriting the geometry helpers avoids a second, subtly different RGB-D
    implementation.  ``YOLOESegmenter.__init__`` is intentionally not called,
    so no YOLO model or weights are loaded.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        dino_extractor: DINOFeaturesExtractor,
        device: str | None = None,
        min_depth_points: int = 50,
        mask_erosion_px: int = 3,
        mahalanobis_thresh: float = 2.0,
        depth_mode_filter_enabled: bool = True,
        depth_mode_k_mad: float = 1.5,
        depth_mode_min_mad_m: float = 0.03,
        timing_enabled: bool = False,
        **_: object,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("backend") != "sam3.1_multiplex_video":
            raise ValueError(f"unsupported SAM3 track manifest: {self.manifest_path}")
        self._frame_rows = list(manifest.get("frames") or [])
        self._cursor = 0
        self.names = [str(value) for value in (manifest.get("prompts") or ["object"])]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dino_extractor = dino_extractor
        self._use_dino_features = True
        if self.dino_extractor.hidden_size is None:
            config = getattr(getattr(self.dino_extractor, "model", None), "config", None)
            self.dino_extractor.hidden_size = int(getattr(config, "hidden_size", 0) or 0) or None
        if self.dino_extractor.hidden_size is None:
            raise RuntimeError("DINO hidden size is unavailable")
        self.feature_dim = int(self.dino_extractor.hidden_size)
        self.min_depth_points = max(1, int(min_depth_points))
        self.mask_erosion_px = max(0, int(mask_erosion_px))
        self.mahalanobis_thresh = float(mahalanobis_thresh)
        self.depth_mode_filter_enabled = bool(depth_mode_filter_enabled)
        self.depth_mode_k_mad = float(depth_mode_k_mad)
        self.depth_mode_min_mad_m = float(depth_mode_min_mad_m)
        self._mahalanobis_eps = 1.0e-6
        self._pixel_grid_cache = {}
        self._timing_enabled = bool(timing_enabled)

    def _load_next(self, expected_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cursor >= len(self._frame_rows):
            raise RuntimeError(
                f"SAM3 cache exhausted at mapper frame {self._cursor}; manifest has {len(self._frame_rows)} frames"
            )
        row = self._frame_rows[self._cursor]
        self._cursor += 1
        path = self.manifest_path.parent / str(row["path"])
        with np.load(path, allow_pickle=False) as saved:
            shape = tuple(int(value) for value in saved["mask_shape"].tolist())
            track_ids = np.asarray(saved["track_ids"], dtype=np.int64)
            class_ids = np.asarray(saved["class_ids"], dtype=np.int64)
            scores = np.asarray(saved["scores"], dtype=np.float32)
            boxes = np.asarray(saved["boxes_xyxy"], dtype=np.float32)
            bits = np.asarray(saved["mask_bits"], dtype=np.uint8)
        count = len(track_ids)
        if count:
            masks_np = np.unpackbits(bits, axis=1, count=shape[0] * shape[1], bitorder="little").reshape(count, *shape).astype(bool)
            masks = torch.from_numpy(masks_np).to(self.device)
        else:
            masks = torch.empty((0, *shape), dtype=torch.bool, device=self.device)
        if shape != expected_hw:
            masks = F.interpolate(masks.float().unsqueeze(1), size=expected_hw, mode="nearest").squeeze(1) > 0.5
            if count:
                scale_x = expected_hw[1] / max(1, shape[1])
                scale_y = expected_hw[0] / max(1, shape[0])
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
        return (
            masks,
            torch.from_numpy(boxes).to(self.device),
            torch.from_numpy(scores).to(self.device),
            torch.from_numpy(class_ids).to(self.device),
            torch.from_numpy(track_ids).to(self.device),
        )

    @torch.no_grad()
    def __call__(
        self,
        color: torch.Tensor | Sequence[torch.Tensor],
        depth: torch.Tensor | Sequence[torch.Tensor],
        depth_intrinsics: torch.Tensor | Sequence[torch.Tensor],
        *,
        camera_names: list[str] | None = None,
        offline_debug: bool = False,
        **_: object,
    ) -> dict:
        colors, depths, intrinsics, _ = self._normalize_batch_inputs(color, depth, depth_intrinsics)
        all_masks: list[torch.Tensor] = []
        boxes_rows: list[torch.Tensor] = []
        score_rows: list[torch.Tensor] = []
        class_rows: list[torch.Tensor] = []
        track_rows: list[torch.Tensor] = []
        batch_rows: list[torch.Tensor] = []
        mean_rows: list[torch.Tensor] = []
        cov_rows: list[torch.Tensor] = []
        count_rows: list[torch.Tensor] = []
        feature_rows: list[torch.Tensor] = []
        points_rows: list[torch.Tensor] = []
        inlier_masks: list[torch.Tensor] = []
        orig_hw: list[tuple[int, int]] = []

        for batch_index, (rgb, depth_value, intrinsics_value) in enumerate(zip(colors, depths, intrinsics)):
            depth_map = self._prepare_depth(depth_value).to(self.device, dtype=torch.float32)
            height, width = int(depth_map.shape[0]), int(depth_map.shape[1])
            orig_hw.append((height, width))
            masks, boxes, scores, class_ids, track_ids = self._load_next((height, width))
            count = int(masks.shape[0])
            all_masks.extend(masks.unbind(0))
            boxes_rows.append(boxes)
            score_rows.append(scores)
            class_rows.append(class_ids)
            track_rows.append(track_ids)
            batch_rows.append(torch.full((count,), batch_index, dtype=torch.int64, device=self.device))
            if count == 0:
                feature_rows.append(torch.empty((0, self.feature_dim), device=self.device))
                continue
            K = intrinsics_value[:3, :3].to(self.device, dtype=torch.float32)
            gy, gx = self._get_pixel_grid(height, width, self.device, depth_map.dtype)
            z = depth_map
            x = (gx - K[0, 2]) * z / K[0, 0]
            y = (gy - K[1, 2]) * z / K[1, 1]
            x_det = x.unsqueeze(0).expand(count, -1, -1)
            y_det = y.unsqueeze(0).expand(count, -1, -1)
            z_det = z.unsqueeze(0).expand(count, -1, -1)
            eroded = self._erode_masks(masks, self.mask_erosion_px)
            weights = (eroded & (z_det > 0) & torch.isfinite(z_det)).float()
            weights = self._depth_mode_filter(z_det, weights)
            n, means, cov6 = self._compute_weighted_stats(x_det, y_det, z_det, weights)
            if self.mahalanobis_thresh > 0:
                weights = self._mahalanobis_reject(x_det, y_det, z_det, weights, means, cov6)
                n, means, cov6 = self._compute_weighted_stats(x_det, y_det, z_det, weights)
            means = self._compute_mask_medians(x_det, y_det, z_det, weights, self.min_depth_points)
            mean_rows.append(means)
            cov_rows.append(cov6)
            count_rows.append(n)
            inlier_masks.extend((weights > 0).unbind(0))
            for detection_index in range(count):
                valid = weights[detection_index] > 0
                points_rows.append(torch.stack((x[valid], y[valid], z[valid]), dim=1))
            local_batch = torch.zeros((count,), dtype=torch.int64, device=self.device)
            feature_rows.append(self._compute_dino_mask_embeddings([rgb], masks, local_batch))

        total = sum(int(value.shape[0]) for value in boxes_rows)
        empty_float = torch.empty((0,), dtype=torch.float32, device=self.device)
        boxes = torch.cat(boxes_rows) if boxes_rows else torch.empty((0, 4), device=self.device)
        scores = torch.cat(score_rows) if score_rows else empty_float
        class_ids = torch.cat(class_rows) if class_rows else torch.empty((0,), dtype=torch.long, device=self.device)
        track_ids = torch.cat(track_rows) if track_rows else torch.empty((0,), dtype=torch.long, device=self.device)
        batch_ids = torch.cat(batch_rows) if batch_rows else torch.empty((0,), dtype=torch.long, device=self.device)
        means = torch.cat(mean_rows) if mean_rows else torch.empty((0, 3), device=self.device)
        cov6 = torch.cat(cov_rows) if cov_rows else torch.empty((0, 6), device=self.device)
        num_pixels = torch.cat(count_rows) if count_rows else empty_float
        features = torch.cat(feature_rows) if feature_rows else torch.empty((0, self.feature_dim), device=self.device)
        offsets = torch.zeros((total + 1,), dtype=torch.int64, device=self.device)
        if points_rows:
            lengths = torch.as_tensor([len(value) for value in points_rows], dtype=torch.int64, device=self.device)
            offsets[1:] = lengths.cumsum(0)
            points = torch.cat(points_rows) if int(lengths.sum()) else torch.empty((0, 3), device=self.device)
        else:
            points = torch.empty((0, 3), device=self.device)
        detections = []
        boxes_cpu, scores_cpu = boxes.cpu(), scores.cpu()
        for index in range(total):
            class_id = int(class_ids[index].item())
            detections.append(
                {
                    "Camera Name": (camera_names or [f"cam{i}" for i in range(len(colors))])[int(batch_ids[index].item())],
                    "BB": boxes_cpu[index].tolist(),
                    "Confidence": float(scores_cpu[index].item()),
                    "Object Category": self.names[class_id] if 0 <= class_id < len(self.names) else "object",
                    "Instance Track ID": int(track_ids[index].item()),
                }
            )
        result = {
            "batch_ids": batch_ids,
            "boxes_xyxy": boxes,
            "num_pixels": num_pixels,
            "scores": scores,
            "class_ids": class_ids,
            "instance_track_ids": track_ids,
            "masks": all_masks,
            "features": features,
            "means": means,
            "cov6": cov6,
            "det_points_flat": points,
            "det_points_offsets": offsets,
            "orig_hw": torch.as_tensor(orig_hw, dtype=torch.int32, device=self.device),
            "ratios": torch.ones((len(orig_hw), 2), dtype=torch.float32, device=self.device),
            "pads": torch.zeros((len(orig_hw), 2), dtype=torch.float32, device=self.device),
            "detections": detections,
            "extra_vocab_matches": torch.zeros((total,), dtype=torch.bool, device=self.device),
            "timings": {},
        }
        if offline_debug:
            result["masks_inlier"] = inlier_masks
        return result
