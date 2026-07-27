"""Lazy frame source for ScanNet ``.sens`` archives.

Format reference: ScanNet SensReader (https://github.com/ScanNet/ScanNet/tree/master/SensReader/python).
This is a Python 3 reimplementation that avoids extracting frames to disk —
frames are indexed once on init (byte offsets + poses only), then decoded on
demand during iteration.
"""

from __future__ import annotations

import io
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import imageio
import numpy as np

from .base import FrameItem, FrameSource

COLOR_COMPRESSION = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


@dataclass
class _FrameEntry:
    color_offset: int
    color_size: int
    depth_offset: int
    depth_size: int


@dataclass
class SensHeader:
    version: int
    sensor_name: str
    intrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    color_compression: str
    depth_compression: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float


def _scan_sens(path: str) -> Tuple[SensHeader, List[_FrameEntry], List[np.ndarray]]:
    """Parse header + frame offsets + poses without decoding image bytes."""
    with open(path, "rb") as f:
        version = struct.unpack("I", f.read(4))[0]
        if version != 4:
            raise ValueError(f"Unsupported .sens version {version} in {path}")
        strlen = struct.unpack("Q", f.read(8))[0]
        sensor_name = f.read(strlen).decode("utf-8", errors="replace")
        intrinsic_color = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
        f.read(64)  # extrinsic_color
        intrinsic_depth = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
        f.read(64)  # extrinsic_depth
        color_compression = COLOR_COMPRESSION[struct.unpack("i", f.read(4))[0]]
        depth_compression = DEPTH_COMPRESSION[struct.unpack("i", f.read(4))[0]]
        color_width = struct.unpack("I", f.read(4))[0]
        color_height = struct.unpack("I", f.read(4))[0]
        depth_width = struct.unpack("I", f.read(4))[0]
        depth_height = struct.unpack("I", f.read(4))[0]
        depth_shift = struct.unpack("f", f.read(4))[0]
        num_frames = struct.unpack("Q", f.read(8))[0]

        header = SensHeader(
            version=version,
            sensor_name=sensor_name,
            intrinsic_color=intrinsic_color,
            intrinsic_depth=intrinsic_depth,
            color_compression=color_compression,
            depth_compression=depth_compression,
            color_width=color_width,
            color_height=color_height,
            depth_width=depth_width,
            depth_height=depth_height,
            depth_shift=float(depth_shift),
        )

        entries: List[_FrameEntry] = []
        poses: List[np.ndarray] = []
        for _ in range(num_frames):
            pose = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
            f.read(16)  # timestamp_color + timestamp_depth
            color_size = struct.unpack("Q", f.read(8))[0]
            depth_size = struct.unpack("Q", f.read(8))[0]
            color_offset = f.tell()
            f.seek(color_size, os.SEEK_CUR)
            depth_offset = f.tell()
            f.seek(depth_size, os.SEEK_CUR)
            entries.append(_FrameEntry(color_offset, color_size, depth_offset, depth_size))
            poses.append(pose)

        return header, entries, poses


def _decode_color(raw: bytes, compression: str) -> np.ndarray:
    if compression == "jpeg":
        return np.asarray(imageio.imread(io.BytesIO(raw)))
    raise NotImplementedError(f"Unsupported color compression: {compression}")


def _decode_depth(raw: bytes, compression: str, height: int, width: int) -> np.ndarray:
    if compression == "zlib_ushort":
        buf = zlib.decompress(raw)
        return np.frombuffer(buf, dtype=np.uint16).reshape(height, width).copy()
    if compression == "raw_ushort":
        return np.frombuffer(raw, dtype=np.uint16).reshape(height, width).copy()
    raise NotImplementedError(f"Unsupported depth compression: {compression}")


def read_sens_frame_color(sens_path: str, frame_idx: int) -> np.ndarray:
    """Decode a single color frame from a ``.sens`` file.

    Used by ``scene_state_io.load_scene_state_image`` to resolve
    ``'<path>#frame=<N>'`` references in ``ImageRecord.storage_path``.
    """
    header, entries, _ = _scan_sens(sens_path)
    if frame_idx < 0 or frame_idx >= len(entries):
        raise IndexError(f"frame {frame_idx} out of range for {sens_path}")
    entry = entries[frame_idx]
    with open(sens_path, "rb") as f:
        f.seek(entry.color_offset)
        raw = f.read(entry.color_size)
    return _decode_color(raw, header.color_compression)


class SensFrameSource(FrameSource):
    """Yields pre-decoded dicts for ``StreamingMapper._run_mapping_batch``."""

    def __init__(
        self,
        sens_path: str | Path,
        *,
        camera: str = "scannet",
        stride: int = 1,
        start: int = 0,
        end: Optional[int] = None,
        nominal_hz: float = 30.0,
    ) -> None:
        self.sens_path = str(Path(sens_path).expanduser())
        if not os.path.isfile(self.sens_path):
            raise FileNotFoundError(f"No .sens file at {self.sens_path}")
        self.camera = camera
        self.stride = max(1, int(stride))
        self.start = max(0, int(start))
        self._nominal_hz = float(nominal_hz) if nominal_hz > 0 else 30.0

        self.header, self._entries, self._poses = _scan_sens(self.sens_path)
        total = len(self._entries)
        self.end = total if end is None or end < 0 else min(int(end), total)

        if self.header.color_compression != "jpeg":
            raise NotImplementedError(
                f"SensFrameSource only supports JPEG color (got {self.header.color_compression})"
            )
        if self.header.depth_compression not in ("zlib_ushort", "raw_ushort"):
            raise NotImplementedError(
                f"SensFrameSource only supports zlib/raw ushort depth (got {self.header.depth_compression})"
            )

        self._indices = list(range(self.start, self.end, self.stride))
        self._fh = open(self.sens_path, "rb")

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[FrameItem]:
        color_intr = {
            "fx": float(self.header.intrinsic_color[0, 0]),
            "fy": float(self.header.intrinsic_color[1, 1]),
            "cx": float(self.header.intrinsic_color[0, 2]),
            "cy": float(self.header.intrinsic_color[1, 2]),
            "width": int(self.header.color_width),
            "height": int(self.header.color_height),
        }
        depth_intr = {
            "fx": float(self.header.intrinsic_depth[0, 0]),
            "fy": float(self.header.intrinsic_depth[1, 1]),
            "cx": float(self.header.intrinsic_depth[0, 2]),
            "cy": float(self.header.intrinsic_depth[1, 2]),
            "width": int(self.header.depth_width),
            "height": int(self.header.depth_height),
        }
        dt_ns = int(1_000_000_000 / self._nominal_hz)

        for frame_idx in self._indices:
            entry = self._entries[frame_idx]
            self._fh.seek(entry.color_offset)
            color_raw = self._fh.read(entry.color_size)
            self._fh.seek(entry.depth_offset)
            depth_raw = self._fh.read(entry.depth_size)

            rgb = _decode_color(color_raw, self.header.color_compression)
            depth_u16 = _decode_depth(
                depth_raw,
                self.header.depth_compression,
                self.header.depth_height,
                self.header.depth_width,
            )
            depth_f32 = depth_u16.astype(np.float32) / float(self.header.depth_shift)

            stamp_ns = frame_idx * dt_ns
            pose = self._poses[frame_idx].astype(np.float32, copy=False)

            yield {
                "camera": self.camera,
                "rgb": rgb.astype(np.uint8, copy=False),
                "depth_f32": depth_f32,
                "T_world_cam": pose,
                "rgb_instrinsics": color_intr,
                "depth_instrinsics": depth_intr,
                "stamp_ns": np.int64(stamp_ns),
                "frame_id": f"{self.camera}_frame_{frame_idx}",
                "received_time": float(stamp_ns) / 1e9,
                # ``source_ref`` lets ``load_scene_state_image`` recover the
                # original RGB without a saved JPEG copy on disk.
                "source_ref": f"{self.sens_path}#frame={frame_idx}",
                # ``rgb_msg`` and ``depth_msg`` are not set; ``_decode_batch``
                # sees ``rgbd_msg is None`` and passes the dict through.
            }

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
