"""Lightweight mask-aware identity tracking for ordered YOLOE video frames."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np


def _normalize(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    return array / norm if array.size and norm > 1.0e-8 else None


def cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    left = _normalize(left)
    right = _normalize(right)
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    return float(np.clip(np.dot(left, right), 0.0, 1.0))


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(4)
    right = np.asarray(right, dtype=np.float32).reshape(4)
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    left_area = float(max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1]))
    right_area = float(max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def labels_related(left: str, right: str) -> bool:
    left_words = set(str(left).casefold().replace("-", " ").split())
    right_words = set(str(right).casefold().replace("-", " ").split())
    return bool(left_words and right_words and (left_words <= right_words or right_words <= left_words))


def mask_appearance_descriptor(rgb: np.ndarray, mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Return Custom's compact HSV/shape descriptor for a YOLOE instance."""
    import cv2

    image = np.asarray(rgb, dtype=np.uint8)
    instance_mask = np.asarray(mask, dtype=bool)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.rint(np.asarray(box)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
    crop = image[y1:y2, x1:x2]
    crop_mask = instance_mask[y1:y2, x1:x2].astype(np.uint8)
    if crop.size == 0:
        return np.zeros((99,), dtype=np.float32)
    if not np.any(crop_mask):
        crop_mask = np.ones(crop.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], crop_mask, [12, 8], [0, 180, 0, 256]).reshape(-1)
    histogram = np.sqrt(np.maximum(histogram, 0.0))
    histogram /= max(1.0e-8, float(np.linalg.norm(histogram)))
    shape = np.asarray(
        [
            (x2 - x1) / max(1.0, width),
            (y2 - y1) / max(1.0, height),
            float(np.count_nonzero(crop_mask)) / max(1.0, crop_mask.size),
        ],
        dtype=np.float32,
    )
    descriptor = _normalize(np.r_[histogram.astype(np.float32), shape * 0.35])
    return descriptor if descriptor is not None else np.zeros((99,), dtype=np.float32)


@dataclass
class _Track:
    track_id: int
    label: str
    box: np.ndarray
    appearance: np.ndarray | None
    feature: np.ndarray | None
    last_frame: int
    hits: int = 1
    velocity: np.ndarray | None = None
    confidence: float = 1.0


class TemporalInstanceTracker:
    """Assign stable IDs without making camera-frame position an identity gate.

    Consecutive frames primarily use 2-D overlap plus appearance. After a long
    occlusion, only a very strong and unambiguous appearance match may revive a
    track. This is deliberately stricter than short-term association so two
    co-visible people or cups never collapse merely because they share a class.
    """

    def __init__(
        self,
        *,
        max_dormant_frames: int = 256,
        high_confidence_threshold: float = 0.35,
        low_confidence_threshold: float = 0.05,
    ) -> None:
        self.max_dormant_frames = max(8, int(max_dormant_frames))
        self.high_confidence_threshold = float(high_confidence_threshold)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self._next_track_id = 1
        self._frame_index: dict[str, int] = {}
        self._tracks: dict[str, dict[int, _Track]] = {}

    def update(
        self,
        camera: str,
        boxes: np.ndarray,
        labels: Sequence[str],
        appearances: np.ndarray,
        features: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        image: np.ndarray | None = None,
    ) -> np.ndarray:
        camera = str(camera or "camera")
        frame_index = self._frame_index.get(camera, -1) + 1
        self._frame_index[camera] = frame_index
        tracks = self._tracks.setdefault(camera, {})
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            expired = [
                track_id
                for track_id, track in tracks.items()
                if frame_index - track.last_frame > self.max_dormant_frames
            ]
            for track_id in expired:
                del tracks[track_id]
            return np.empty((0,), dtype=np.int64)
        appearances = np.asarray(appearances, dtype=np.float32).reshape(len(boxes), -1)
        feature_rows = (
            np.asarray(features, dtype=np.float32).reshape(len(boxes), -1)
            if features is not None and np.asarray(features).size
            else None
        )
        score_rows = (
            np.asarray(scores, dtype=np.float32).reshape(len(boxes))
            if scores is not None and np.asarray(scores).size
            else np.ones((len(boxes),), dtype=np.float32)
        )
        assigned = np.full((len(boxes),), -1, dtype=np.int64)
        candidates: list[tuple[float, int, int, float, int]] = []
        dormant_alternatives: dict[int, list[float]] = {}
        for detection_index, (box, label, appearance) in enumerate(zip(boxes, labels, appearances)):
            confidence = float(score_rows[detection_index])
            if confidence < self.low_confidence_threshold:
                continue
            feature = feature_rows[detection_index] if feature_rows is not None else None
            for track_id, track in tracks.items():
                age = frame_index - track.last_frame
                if age <= 0 or age > self.max_dormant_frames:
                    continue
                label_agrees = labels_related(label, track.label)
                velocity = track.velocity if track.velocity is not None else np.zeros((4,), dtype=np.float32)
                predicted_box = track.box + velocity * min(age, 8)
                overlap = box_iou(box, predicted_box)
                appearance_score = cosine_similarity(appearance, track.appearance)
                feature_score = cosine_similarity(feature, track.feature)
                if age <= 5:
                    # ByteTrack-style second association: low-confidence
                    # detections may continue a recent trajectory but never
                    # revive a dormant identity or create a new one.
                    if confidence >= self.high_confidence_threshold:
                        allowed = (
                            (label_agrees and (overlap >= 0.015 or appearance_score >= 0.74))
                            or (
                                not label_agrees
                                and overlap >= 0.20
                                and appearance_score >= 0.78
                                and feature_score >= 0.55
                            )
                        )
                    else:
                        allowed = age <= 2 and (
                            (label_agrees and (overlap >= 0.08 or appearance_score >= 0.82))
                            or (
                                not label_agrees
                                and overlap >= 0.35
                                and appearance_score >= 0.86
                                and feature_score >= 0.65
                            )
                        )
                    # A currently visible trajectory must win over an old
                    # dormant look-alike. Re-identification is only a fallback
                    # when no recent association claims the detection.
                    score = (
                        1.0
                        + 0.46 * overlap
                        + 0.34 * appearance_score
                        + 0.12 * feature_score
                        + 0.08 * confidence
                        + (0.04 if label_agrees else 0.0)
                    )
                else:
                    allowed = (
                        confidence >= self.high_confidence_threshold
                        and appearance_score >= 0.94
                        and feature_score >= 0.70
                        and (label_agrees or (appearance_score >= 0.975 and feature_score >= 0.85))
                    )
                    score = 0.72 * appearance_score + 0.28 * feature_score - min(age, 200) * 0.00015
                    dormant_alternatives.setdefault(detection_index, []).append(score)
                if allowed:
                    candidates.append((score, detection_index, track_id, appearance_score, age))

        used_tracks: set[int] = set()
        # Match high-confidence detections first, then recover occluded
        # objects from low-confidence detections as ByteTrack does.
        for score, detection_index, track_id, appearance_score, age in sorted(
            candidates,
            key=lambda row: (score_rows[row[1]] >= self.high_confidence_threshold, row[0]),
            reverse=True,
        ):
            if assigned[detection_index] >= 0 or track_id in used_tracks:
                continue
            if age > 5:
                alternatives = sorted(dormant_alternatives.get(detection_index, []), reverse=True)
                second = alternatives[1] if len(alternatives) > 1 else -1.0
                if appearance_score < 0.95 or score - second < 0.025:
                    continue
            assigned[detection_index] = track_id
            used_tracks.add(track_id)

        for detection_index, (box, label, appearance) in enumerate(zip(boxes, labels, appearances)):
            feature = feature_rows[detection_index] if feature_rows is not None else None
            track_id = int(assigned[detection_index])
            if track_id < 0:
                if float(score_rows[detection_index]) < self.high_confidence_threshold:
                    continue
                track_id = self._next_track_id
                self._next_track_id += 1
                tracks[track_id] = _Track(
                    track_id=track_id,
                    label=str(label),
                    box=box.copy(),
                    appearance=_normalize(appearance),
                    feature=_normalize(feature),
                    last_frame=frame_index,
                    velocity=np.zeros((4,), dtype=np.float32),
                    confidence=float(score_rows[detection_index]),
                )
                assigned[detection_index] = track_id
                continue
            track = tracks[track_id]
            elapsed = max(1, frame_index - track.last_frame)
            observed_velocity = (box - track.box) / float(elapsed)
            track.velocity = (
                observed_velocity.astype(np.float32)
                if track.velocity is None
                else (0.75 * track.velocity + 0.25 * observed_velocity).astype(np.float32)
            )
            new_appearance = _normalize(appearance)
            new_feature = _normalize(feature)
            if new_appearance is not None:
                track.appearance = _normalize(
                    new_appearance if track.appearance is None else 0.85 * track.appearance + 0.15 * new_appearance
                )
            if new_feature is not None:
                track.feature = _normalize(
                    new_feature if track.feature is None else 0.90 * track.feature + 0.10 * new_feature
                )
            track.box = box.copy()
            track.label = str(label)
            track.last_frame = frame_index
            track.hits += 1
            track.confidence = float(score_rows[detection_index])

        expired = [track_id for track_id, track in tracks.items() if frame_index - track.last_frame > self.max_dormant_frames]
        for track_id in expired:
            del tracks[track_id]
        return assigned


class _PrecomputedReIDEncoder:
    """Expose AL-FARM's mask embeddings through Ultralytics' ReID interface."""

    def __init__(self) -> None:
        self._features = np.empty((0, 0), dtype=np.float32)

    def set_features(self, features: np.ndarray) -> None:
        rows = np.asarray(features, dtype=np.float32)
        if rows.ndim != 2:
            raise ValueError(f"ReID features must be a matrix, got shape {rows.shape}")
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        self._features = np.divide(
            rows,
            np.maximum(norms, 1.0e-8),
            out=np.zeros_like(rows),
            where=norms > 1.0e-8,
        )

    def inference(self, image: np.ndarray | None, detections: np.ndarray) -> np.ndarray:
        # BYTETracker appends each detection's input-row index to xywh. This
        # lets BoT-SORT consume the embeddings already pooled from the exact
        # instance masks instead of running a second crop-level ReID model.
        rows = np.asarray(detections)
        if not len(rows):
            return np.empty((0, self._features.shape[1]), dtype=np.float32)
        indices = np.rint(rows[:, -1]).astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self._features)):
            raise IndexError("BoT-SORT detection index is outside the ReID feature matrix")
        return self._features[indices].copy()


class BoTSORTReIDTracker:
    """Category-agnostic BoT-SORT with GMC and precomputed DINOv3 ReID.

    The vendored Ultralytics release contains BoT-SORT's Kalman filter,
    ByteTrack-style two-stage association and global motion compensation, but
    leaves its ReID encoder unset. This adapter supplies AL-FARM's DINOv3 mask
    embeddings through that interface. Class labels deliberately do not gate
    association because open-vocabulary labels commonly change over a track.
    """

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.35,
        track_low_thresh: float = 0.05,
        new_track_thresh: float = 0.35,
        track_buffer: int = 90,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        label_flip_reid_similarity: float = 0.85,
        gmc_method: str = "sparseOptFlow",
        frame_rate: int = 30,
    ) -> None:
        try:
            from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
            from ultralytics.trackers.basetrack import BaseTrack
            from ultralytics.trackers.utils import matching
        except Exception as exc:
            raise RuntimeError(
                "BoT-SORT is unavailable. Install the vendored Ultralytics tracking "
                "dependency (lapx) in the FARM environment."
            ) from exc

        class _DinoBOTSORT(BOTSORT):
            # The vendored BOTrack initializes its feature history after
            # consuming the first feature. Construct without a feature and
            # then update it, avoiding that upstream initialization bug.
            def init_track(inner_self, dets, scores, classes, image=None):
                if len(dets) == 0:
                    return []
                if inner_self.args.with_reid and inner_self.encoder is not None:
                    feature_rows = inner_self.encoder.inference(image, dets)
                    tracks = [BOTrack(box, score, cls) for box, score, cls in zip(dets, scores, classes)]
                    for track, feature in zip(tracks, feature_rows):
                        track.update_features(feature)
                    return tracks
                return [BOTrack(box, score, cls) for box, score, cls in zip(dets, scores, classes)]

            def get_dists(inner_self, tracks, detections):
                distances = super().get_dists(tracks, detections)
                if not tracks or not detections:
                    return distances
                track_classes = np.asarray([int(track.cls) for track in tracks], dtype=np.int64)
                detection_classes = np.asarray([int(track.cls) for track in detections], dtype=np.int64)
                label_changed = track_classes[:, None] != detection_classes[None, :]
                # Label is a soft guard, not an identity key: an adjacent-frame
                # label flip remains legal when mask appearance is compelling.
                # It blocks the common failure where two overlapping objects
                # with unrelated labels inherit one another's Kalman track.
                embedding_distance = matching.embedding_distance(tracks, detections) / 2.0
                weak_reid = embedding_distance > (1.0 - float(label_flip_reid_similarity)) / 2.0
                distances[label_changed & weak_reid] = 1.0
                return distances

            @staticmethod
            def remove_duplicate_stracks(tracked, lost):
                # Upstream removes spatially overlapping new/lost tracks even
                # when their semantics and ReID disagree. That prevents a
                # person emerging in front of a lost chair track from ever
                # starting. Only de-duplicate same-label tracks here.
                pairwise = matching.iou_distance(tracked, lost)
                pairs = np.where(pairwise < 0.15)
                duplicate_tracked: list[int] = []
                duplicate_lost: list[int] = []
                for left, right in zip(*pairs):
                    if int(tracked[left].cls) != int(lost[right].cls):
                        continue
                    tracked_age = tracked[left].frame_id - tracked[left].start_frame
                    lost_age = lost[right].frame_id - lost[right].start_frame
                    if tracked_age > lost_age:
                        duplicate_lost.append(right)
                    else:
                        duplicate_tracked.append(left)
                return (
                    [track for index, track in enumerate(tracked) if index not in duplicate_tracked],
                    [track for index, track in enumerate(lost) if index not in duplicate_lost],
                )

        self.backend_name = "botsort_dinov3_reid"
        self.gmc_method = str(gmc_method)
        self.label_flip_reid_similarity = float(label_flip_reid_similarity)
        if not 0.0 <= self.label_flip_reid_similarity <= 1.0:
            raise ValueError("label_flip_reid_similarity must be in [0, 1]")
        self._botsort_class = _DinoBOTSORT
        self._base_track_class = BaseTrack
        self._frame_rate = max(1, int(frame_rate))
        self._args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=float(track_high_thresh),
            track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(new_track_thresh),
            track_buffer=max(1, int(track_buffer)),
            match_thresh=float(match_thresh),
            fuse_score=True,
            proximity_thresh=float(proximity_thresh),
            appearance_thresh=float(appearance_thresh),
            with_reid=True,
            gmc_method=self.gmc_method,
        )
        if not 0.0 <= self._args.track_low_thresh <= self._args.track_high_thresh <= 1.0:
            raise ValueError("expected 0 <= track_low_thresh <= track_high_thresh <= 1")
        if self._args.new_track_thresh < self._args.track_high_thresh:
            raise ValueError("new_track_thresh must be at least track_high_thresh")
        self._trackers: dict[str, object] = {}
        self._encoders: dict[str, _PrecomputedReIDEncoder] = {}
        self._global_ids: dict[tuple[str, int], int] = {}
        self._label_ids: dict[str, dict[str, int]] = {}
        self._next_global_id = 1

    def _camera_tracker(self, camera: str) -> tuple[object, _PrecomputedReIDEncoder]:
        tracker = self._trackers.get(camera)
        encoder = self._encoders.get(camera)
        if tracker is None or encoder is None:
            previous_counter = int(self._base_track_class._count)
            tracker = self._botsort_class(self._args, frame_rate=self._frame_rate)
            # BYTETracker resets Ultralytics' process-global counter in every
            # constructor. Restore it so lazily creating a second camera does
            # not recycle IDs later in the first camera.
            self._base_track_class._count = previous_counter
            encoder = _PrecomputedReIDEncoder()
            # The old vendored release initializes this field to None even
            # with with_reid=True. Supplying it activates BOTrack features and
            # embedding_distance without modifying third-party source.
            tracker.encoder = encoder
            self._trackers[camera] = tracker
            self._encoders[camera] = encoder
        return tracker, encoder

    @staticmethod
    def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        xywh = boxes.copy()
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
        xywh[:, 2] = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        xywh[:, 3] = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        return xywh

    def update(
        self,
        camera: str,
        boxes: np.ndarray,
        labels: Sequence[str],
        appearances: np.ndarray,
        features: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        image: np.ndarray | None = None,
    ) -> np.ndarray:
        camera = str(camera or "camera")
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        count = len(boxes)
        if features is None and count:
            raise ValueError("BoT-SORT ReID requires one DINO feature vector per detection")
        if count:
            feature_rows = np.asarray(features, dtype=np.float32).reshape(count, -1)
        else:
            feature_array = np.asarray(features, dtype=np.float32) if features is not None else np.empty((0, 1))
            width = int(feature_array.shape[1]) if feature_array.ndim == 2 and feature_array.shape[1] else 1
            feature_rows = np.empty((0, width), dtype=np.float32)
        score_rows = (
            np.asarray(scores, dtype=np.float32).reshape(count)
            if scores is not None
            else np.ones((count,), dtype=np.float32)
        )
        tracker, encoder = self._camera_tracker(camera)
        encoder.set_features(feature_rows)
        label_ids = self._label_ids.setdefault(camera, {})
        class_rows = np.empty((count,), dtype=np.float32)
        for index, label in enumerate(labels):
            key = str(label).strip().casefold() or "object"
            if key not in label_ids:
                label_ids[key] = len(label_ids)
            class_rows[index] = float(label_ids[key])
        results = SimpleNamespace(
            xywh=self._xyxy_to_xywh(boxes),
            conf=score_rows,
            cls=class_rows,
        )
        bgr = None
        if image is not None:
            rgb = np.asarray(image, dtype=np.uint8)
            bgr = np.ascontiguousarray(rgb[..., ::-1]) if rgb.ndim == 3 else rgb
        rows = np.asarray(tracker.update(results, bgr), dtype=np.float32)
        assigned = np.full((count,), -1, dtype=np.int64)
        if not rows.size:
            return assigned
        rows = rows.reshape(-1, 8)
        for row in rows:
            detection_index = int(round(float(row[7])))
            local_id = int(round(float(row[4])))
            if not 0 <= detection_index < count:
                continue
            key = (camera, local_id)
            global_id = self._global_ids.get(key)
            if global_id is None:
                global_id = self._next_global_id
                self._next_global_id += 1
                self._global_ids[key] = global_id
            assigned[detection_index] = global_id
        return assigned
