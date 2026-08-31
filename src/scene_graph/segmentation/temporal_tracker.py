"""Lightweight mask-aware identity tracking for ordered YOLOE video frames."""

from __future__ import annotations

from dataclasses import dataclass
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
