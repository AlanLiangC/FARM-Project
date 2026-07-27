"""Frame source backed by a ``frames.json`` index: a JSON list of per-frame
entries, each resolving to one JPEG/PNG (RGB) and one depth file — either a
``.npy`` (float32 metres, NaN for invalid) or a ``.png`` (uint16, 0 = invalid,
decoded via the scene-level ``depth_encoding`` block; defaults to millimetres
when the block is absent). FARM-Scenes releases use the uint16-PNG form with
``depth_encoding.scale_to_metres`` per scene.

RGB and depth often have different on-disk resolutions (e.g. 1920x1280 RGB
JPEGs against 480x320 .npy depths sharing the same fisheye intrinsics ``K``,
with depth-scale principal points). We resize RGB to depth resolution so the
pinhole unprojection inside :class:`StreamingMapper` lines up with ``K``.

Equidistant (fisheye) distortion is *ignored* — pixels are treated as if
they came from a plain pinhole camera. This is a good approximation near the
principal point but degrades toward the image edges for wide-FOV/fisheye
sources.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
from PIL import Image

from .base import FrameItem, FrameSource

LOGGER = logging.getLogger("scene_graph.offline.frame_sources.frames_json")


def _resize_rgb_to(target_h: int, target_w: int, rgb: np.ndarray) -> np.ndarray:
    """Resize an HxWx3 uint8 RGB to (target_h, target_w) using PIL."""
    if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
        return rgb
    img = Image.fromarray(rgb)
    img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class FramesJsonFrameSource(FrameSource):
    """Yields pre-decoded dicts from a ``frames.json`` scene directory.

    Args:
        scene_dir: Path containing ``frames.json`` plus the rgb/depth subdirs
            referenced by relative ``rgb_path`` / ``depth_path`` entries.
        cameras: Optional whitelist (e.g. ``("hdr_front",)``). Defaults to
            every camera listed in ``frames.json``.
        stride: Take every Nth frame (per camera, applied after grouping).
        start: Skip the first ``start`` frames per camera.
        end: Optional exclusive end index per camera (after stride).
        nominal_hz: Used to synthesise stamp_ns when ``timestamp_ns`` is
            absent from the index.
        depth_clip_m: Optionally clip depth values above this distance to 0
            (treated as invalid). Useful for outdoor LiDAR with far hits.
        nan_to_zero: Replace NaN depth values with 0.0 (the unprojection
            convention used by ``StreamingMapper``).
    """

    def __init__(
        self,
        scene_dir: str | Path,
        *,
        cameras: Optional[Sequence[str]] = None,
        stride: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        nominal_hz: float = 10.0,
        depth_clip_m: Optional[float] = None,
        nan_to_zero: bool = True,
    ) -> None:
        self.scene_dir = Path(scene_dir).expanduser()
        index_path = self.scene_dir / "frames.json"
        if not index_path.exists():
            raise FileNotFoundError(f"frames.json not found at {index_path}")
        raw = json.loads(index_path.read_text(encoding="utf-8"))

        all_cams: List[str] = list(raw.get("cameras") or [])
        chosen = list(cameras) if cameras else all_cams
        for c in chosen:
            if c not in all_cams:
                raise ValueError(f"camera {c!r} not present in {index_path} (have {all_cams})")
        self.cameras: List[str] = chosen

        self.stride = max(1, int(stride))
        self.start = max(0, int(start))
        self._nominal_hz = float(nominal_hz) if nominal_hz > 0 else 10.0
        self._end = end
        self._depth_clip_m = depth_clip_m
        self._nan_to_zero = bool(nan_to_zero)

        # Scene-level uint16-PNG depth quantisation (metres per count). FARM-
        # Scenes writes ``depth_encoding: {format: png16, scale_to_metres: s}``
        # because outdoor LiDAR depth exceeds the 65.535 m reach of plain
        # millimetres; absent block -> legacy 1 mm/count convention.
        self._png_depth_scale = 1.0e-3
        enc = raw.get("depth_encoding")
        if isinstance(enc, dict):
            scale = enc.get("scale_to_metres")
            if scale is not None and float(scale) > 0:
                self._png_depth_scale = float(scale)

        # Group + sort frames by camera so we iterate in (frame_id, camera) order.
        per_cam: dict[str, list[dict]] = {c: [] for c in self.cameras}
        for entry in raw.get("frames") or []:
            cam = entry.get("camera")
            if cam not in per_cam:
                continue
            per_cam[cam].append(entry)
        for cam, entries in per_cam.items():
            entries.sort(key=lambda e: str(e.get("frame_id", "")))

        # Apply stride/start/end per camera, then interleave so different
        # cameras' frames are seen in temporal order (driven by frame_id).
        kept: list[dict] = []
        for cam, entries in per_cam.items():
            sliced = entries[self.start :]
            if self._end is not None and self._end >= 0:
                sliced = sliced[: max(0, int(self._end) - self.start)]
            sliced = sliced[:: self.stride]
            kept.extend(sliced)
        kept.sort(key=lambda e: (str(e.get("frame_id", "")), str(e.get("camera", ""))))
        self._frames = kept
        LOGGER.info(
            "FramesJsonFrameSource: %d frames over %s (stride=%d, start=%d)",
            len(kept), self.cameras, self.stride, self.start,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def _read_rgb(self, rel_path: str, target_h: int, target_w: int) -> np.ndarray:
        p = (self.scene_dir / rel_path).resolve()
        with Image.open(p) as img:
            img = img.convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8)
        return _resize_rgb_to(target_h, target_w, rgb)

    def _read_depth(self, rel_path: str) -> np.ndarray:
        p = (self.scene_dir / rel_path).resolve()
        if rel_path.endswith(".npy"):
            arr = np.load(p)
        elif rel_path.endswith(".png"):
            # uint16 counts -> metres via the scene's depth_encoding scale
            # (1 mm/count when frames.json carries no depth_encoding block).
            with Image.open(p) as img:
                arr = np.asarray(img)
            arr = arr.astype(np.float32) * self._png_depth_scale
        else:
            raise ValueError(f"unsupported depth extension: {p.name}")
        depth = np.asarray(arr, dtype=np.float32)
        if self._nan_to_zero:
            depth = np.where(np.isfinite(depth), depth, 0.0)
        if self._depth_clip_m is not None:
            depth = np.where(depth <= float(self._depth_clip_m), depth, 0.0)
        depth = np.where(depth < 0, 0.0, depth)
        return depth

    @staticmethod
    def _intrinsics_dict(K: np.ndarray, height: int, width: int) -> dict:
        K = np.asarray(K, dtype=np.float32).reshape(3, 3)
        return {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "width": int(width),
            "height": int(height),
        }

    def __iter__(self) -> Iterator[FrameItem]:
        dt_ns = int(1_000_000_000 / self._nominal_hz)
        for global_idx, entry in enumerate(self._frames):
            cam = str(entry["camera"])
            depth_size = tuple(entry.get("depth_size") or [])
            if len(depth_size) != 2:
                raise ValueError(f"frame {entry.get('frame_id')} missing depth_size")
            d_h, d_w = int(depth_size[0]), int(depth_size[1])
            rgb = self._read_rgb(entry["rgb_path"], d_h, d_w)
            depth = self._read_depth(entry["depth_path"])
            if depth.shape != (d_h, d_w):
                raise ValueError(
                    f"depth shape {depth.shape} != frames.json depth_size ({d_h}, {d_w})"
                )
            T = np.asarray(entry["T_world_cam"], dtype=np.float32).reshape(4, 4)
            K = np.asarray(entry["K"], dtype=np.float32).reshape(3, 3)
            intrinsics = self._intrinsics_dict(K, d_h, d_w)
            stamp_ns = int(entry.get("timestamp_ns") or (global_idx * dt_ns))
            rgb_abs = (self.scene_dir / entry["rgb_path"]).resolve()
            yield {
                "camera": cam,
                "rgb": rgb,
                "depth_f32": depth,
                "T_world_cam": T,
                "rgb_instrinsics": intrinsics,
                "depth_instrinsics": intrinsics,
                "stamp_ns": np.int64(stamp_ns),
                "frame_id": f"{cam}_{entry.get('frame_id')}",
                "received_time": float(stamp_ns) / 1e9,
                # rgb is already a JPEG on disk; absolute path is enough for
                # ``load_scene_state_image`` to recover it later.
                "source_ref": str(rgb_abs),
            }

    def close(self) -> None:
        return None
