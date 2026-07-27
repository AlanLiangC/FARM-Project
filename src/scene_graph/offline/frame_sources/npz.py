"""Frame source for sequences stored as consolidated NPZ archives.

Mirrors :class:`SensFrameSource` and :class:`RosbagFrameSource`, but reads
NPZ chunks produced by tools like ``scripts/render_hm3d_trajectory.py``.

Each NPZ archive must contain::

    images      : (N, H, W, 3) uint8
    depths      : (N, H, W) float32, metric depth in metres
    camtoworlds : (N, 4, 4) float32   (or (N, 3, 4))
    K           : (3, 3) float32

Optional scalars::

    pose_convention : 'opengl' | 'opencv'   (default: 'opengl')
    nominal_hz      : float                 (default: 30.0)

Frames yielded match the pre-decoded shape that
``StreamingMapper._decode_batch`` accepts when ``rgbd_msg is None``::

    {
      "camera": str,
      "rgb": (H, W, 3) uint8,
      "depth_f32": (H, W) float32,
      "T_world_cam": (4, 4) float32        # OpenCV convention
      "rgb_instrinsics": {"fx", "fy", "cx", "cy", "width", "height"},
      "depth_instrinsics": ...,
      "stamp_ns": int,
      "frame_id": str,
      "received_time": float,
    }

The OpenGL → OpenCV conversion (if needed) is applied here so downstream
consumers always see OpenCV camera-to-world poses.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .base import FrameItem, FrameSource

# (3, 3) flip — match scene_graph.datasets._gradslam_base.OPENGL_TO_OPENCV.
# OpenGL has +Y up, +X right, -Z forward. OpenCV has +Y down, +X right,
# +Z forward. Negating Y and Z columns of the 3x3 rotation does that.
_OPENGL_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
)


class NPZFrameSource(FrameSource):
    """Yields pre-decoded dicts from one or more NPZ archives in a directory."""

    def __init__(
        self,
        npz_dir: str | Path,
        *,
        camera: str = "hm3d",
        stride: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        nominal_hz: float = 30.0,
        pattern: str = "*.npz",
    ) -> None:
        self.npz_dir = Path(npz_dir).expanduser()
        if not self.npz_dir.is_dir():
            raise FileNotFoundError(f"NPZ directory not found: {self.npz_dir}")
        self.camera = camera
        self.stride = max(1, int(stride))
        self.start = max(0, int(start))
        self._nominal_hz = float(nominal_hz) if nominal_hz > 0 else 30.0
        self._pattern = pattern

        paths = sorted(glob.glob(str(self.npz_dir / pattern)))
        if not paths:
            raise FileNotFoundError(
                f"No NPZ archives matching '{pattern}' in {self.npz_dir}"
            )
        self._npz_paths: List[Path] = [Path(p) for p in paths]

        # Probe each archive for frame counts.  The HM3D renderer writes
        # compressed NPZ files, so per-frame indexed access would re-inflate the
        # large image/depth arrays.  The iterator below therefore loads one
        # chunk into RAM at a time and then returns cheap views.
        self._chunks: List[dict] = []
        total = 0
        for path in self._npz_paths:
            with np.load(path) as data:
                n = int(data["images"].shape[0])
                self._chunks.append({"path": path, "n": n, "offset": total})
                total += n
        self.total = total
        self.end = total if end is None or end < 0 else min(int(end), total)
        self._indices = list(range(self.start, self.end, self.stride))

    def __len__(self) -> int:
        return len(self._indices)

    def _locate(self, global_idx: int) -> tuple[int, int]:
        """Return ``(chunk_index, local_index)`` for a global frame index."""
        for ci, chunk in enumerate(self._chunks):
            if global_idx < chunk["offset"] + chunk["n"]:
                return ci, global_idx - chunk["offset"]
        raise IndexError(f"frame {global_idx} out of range (total={self.total})")

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

    @staticmethod
    def _normalise_pose(pose: np.ndarray, convention: str) -> np.ndarray:
        """Coerce pose to (4, 4) OpenCV camera-to-world float32."""
        pose = np.asarray(pose, dtype=np.float32)
        if pose.ndim == 2 and pose.shape == (3, 4):
            pad = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=0)
        if pose.shape != (4, 4):
            raise ValueError(f"Unexpected pose shape {pose.shape}")
        if convention == "opengl":
            pose = pose.copy()
            pose[:3, :3] = pose[:3, :3] @ _OPENGL_TO_OPENCV
        elif convention != "opencv":
            raise ValueError(f"Unsupported pose convention '{convention}'")
        return np.ascontiguousarray(pose, dtype=np.float32)

    def __iter__(self) -> Iterator[FrameItem]:
        # Reset cursor for a fresh iteration. (Re-entering __iter__ is fine for
        # `for x in source:` but rare — the offline runner takes one iter().)
        self._iter_pos = 0
        return self

    def skip(self, n: int) -> int:
        """Cheaply advance past ``n`` upcoming items without decoding.
        Returns the number actually skipped.  Used by the offline runner's
        ``--drop-when-late`` path so that dropped frames don't pay full NPZ
        decode cost (~290 ms/frame on these 480x640 chunks).
        """
        if n <= 0:
            return 0
        pos = getattr(self, "_iter_pos", 0)
        actual = min(int(n), len(self._indices) - pos)
        self._iter_pos = pos + actual
        return actual

    def __next__(self) -> FrameItem:
        if not hasattr(self, "_iter_pos"):
            self._iter_pos = 0
        if not hasattr(self, "_cur_chunk_state"):
            self._cur_chunk_state = {"chunk": -1, "data": None}
        if self._iter_pos >= len(self._indices):
            raise StopIteration
        dt_ns = int(1_000_000_000 / self._nominal_hz)
        global_idx = self._indices[self._iter_pos]
        self._iter_pos += 1
        _decode_state = self._cur_chunk_state
        cur_chunk = _decode_state["chunk"]
        cur_data = _decode_state["data"]
        chunk_idx, local_idx = self._locate(global_idx)
        if chunk_idx != cur_chunk:
            if cur_data is not None:
                cur_data["_npz"].close()
            cur_chunk = chunk_idx
            npz = np.load(self._chunks[chunk_idx]["path"])
            convention = "opengl"
            if "pose_convention" in npz.files:
                raw = npz["pose_convention"]
                convention = str(raw.tolist() if isinstance(raw, np.ndarray) else raw).lower()
            cur_data = {
                "_npz": npz,
                "images": np.asarray(npz["images"]),
                "depths": np.asarray(npz["depths"]),
                "camtoworlds": np.asarray(npz["camtoworlds"]),
                "K": np.asarray(npz["K"], dtype=np.float32),
                "convention": convention,
            }
            self._cur_chunk_state = {"chunk": cur_chunk, "data": cur_data}
        assert cur_data is not None  # for type-checkers
        convention = cur_data["convention"]
        rgb = np.asarray(cur_data["images"][local_idx], dtype=np.uint8)
        depth_f32 = np.asarray(cur_data["depths"][local_idx], dtype=np.float32)
        pose = self._normalise_pose(cur_data["camtoworlds"][local_idx], convention)
        K = cur_data["K"]
        H, W = rgb.shape[0], rgb.shape[1]

        intrinsics = self._intrinsics_dict(K, H, W)
        stamp_ns = global_idx * dt_ns
        chunk_path = self._chunks[chunk_idx]["path"]
        return {
            "camera": self.camera,
            "rgb": rgb,
            "depth_f32": depth_f32,
            "T_world_cam": pose,
            "rgb_instrinsics": intrinsics,
            "depth_instrinsics": intrinsics,  # rendered: same camera for both
            "stamp_ns": np.int64(stamp_ns),
            "frame_id": f"{self.camera}_frame_{global_idx}",
            "received_time": float(stamp_ns) / 1e9,
            # ``<chunk>.npz#frame=<local_idx>`` — resolved by
            # ``load_scene_state_image`` so the eval viser can recover RGB.
            "source_ref": f"{chunk_path}#frame={local_idx}",
        }

    def close(self) -> None:
        st = getattr(self, "_cur_chunk_state", None)
        if st is not None and st.get("data") is not None:
            try:
                st["data"]["_npz"].close()
            except Exception:
                pass
            self._cur_chunk_state = {"chunk": -1, "data": None}
        return None
