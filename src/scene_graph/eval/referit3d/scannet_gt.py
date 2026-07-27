"""ScanNet ground-truth instance loader for ReferIt3D evaluation.

Each ScanNet scene ships with three artifacts that together define the GT
instance segmentation:

- ``<scan>.aggregation.json`` — list of ``segGroups``, each with
  ``{id, objectId, segments: [seg_id, ...], label}``. ``objectId`` is what the
  ReferIt3D CSVs reference as ``target_id``.
- ``<scan>_vh_clean_2.0.010000.segs.json`` — per-vertex ``segIndices`` array
  mapping each mesh vertex to a seg id.
- ``<scan>_vh_clean_2.ply`` — the canonical mesh; we only need vertex XYZ.

For each instance we collect the vertices whose seg id appears in the
instance's ``segments`` list, then compute an axis-aligned bounding box. Cached
to ``<scan>/<scan>_referit3d_bboxes.npz`` so subsequent loads are O(K).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .dataset import default_scans_dir


@dataclass(frozen=True)
class GTInstance:
    """A ScanNet GT instance (one ``segGroup`` from ``aggregation.json``)."""

    instance_id: int
    label: str
    bbox_min: np.ndarray  # (3,) float32, axis-aligned in scene coordinates
    bbox_max: np.ndarray  # (3,) float32
    n_vertices: int

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.bbox_min + self.bbox_max)

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass(frozen=True)
class GTInstancePoints:
    """A ScanNet GT instance with its annotated mesh vertices."""

    instance_id: int
    label: str
    points: np.ndarray  # (N, 3) float32 scene coordinates


def scene_dir(scan_id: str, scans_dir: Optional[Path] = None) -> Path:
    return (Path(scans_dir) if scans_dir else default_scans_dir()) / scan_id


def cache_path(scan_id: str, scans_dir: Optional[Path] = None) -> Path:
    return scene_dir(scan_id, scans_dir) / f"{scan_id}_referit3d_bboxes.npz"


def aggregation_path(scan_id: str, scans_dir: Optional[Path] = None) -> Path:
    return scene_dir(scan_id, scans_dir) / f"{scan_id}.aggregation.json"


def segs_path(scan_id: str, scans_dir: Optional[Path] = None) -> Path:
    return scene_dir(scan_id, scans_dir) / f"{scan_id}_vh_clean_2.0.010000.segs.json"


def mesh_path(scan_id: str, scans_dir: Optional[Path] = None) -> Path:
    return scene_dir(scan_id, scans_dir) / f"{scan_id}_vh_clean_2.ply"


def _read_ply_vertices_xyz(path: Path) -> np.ndarray:
    """Read XYZ vertex coordinates from a binary little-endian PLY.

    ScanNet's ``vh_clean_2.ply`` header is fixed:
        property float x / y / z
        property uchar red / green / blue / alpha
    16 bytes per vertex (3 × float32 + 4 × uint8). Faces follow but we ignore
    them. We parse the header to find ``element vertex N`` and then read N
    records of 16 bytes.

    A small inline parser keeps the package free of plyfile/trimesh/open3d
    dependencies, which matters for host-side smoke testing outside docker.
    """
    with path.open("rb") as fp:
        # Header is ASCII, terminated by 'end_header\n'.
        header_lines = []
        while True:
            line = fp.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF in PLY header")
            decoded = line.decode("ascii", errors="replace").rstrip("\n").rstrip("\r")
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        n_vertices: Optional[int] = None
        vertex_props: list[str] = []
        in_vertex_element = False
        is_binary_le = False
        for line in header_lines:
            if line.startswith("format "):
                is_binary_le = "binary_little_endian" in line
            elif line.startswith("element "):
                _, kind, count = line.split()
                in_vertex_element = (kind == "vertex")
                if in_vertex_element:
                    n_vertices = int(count)
            elif line.startswith("property ") and in_vertex_element:
                # 'property float x' or 'property uchar red'
                vertex_props.append(line)

        if not is_binary_le:
            raise ValueError(f"{path}: only binary_little_endian PLY supported")
        if n_vertices is None:
            raise ValueError(f"{path}: no 'element vertex' in header")

        # Build a struct format string for one vertex record.
        type_to_fmt = {
            "char": "b", "uchar": "B",
            "short": "h", "ushort": "H",
            "int": "i", "uint": "I",
            "float": "f", "double": "d",
        }
        type_to_size = {k: struct.calcsize(v) for k, v in type_to_fmt.items()}
        fmt_chars: list[str] = []
        prop_names: list[str] = []
        for line in vertex_props:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}: unexpected vertex property line: {line!r}")
            _, ply_type, name = parts
            if ply_type not in type_to_fmt:
                raise ValueError(f"{path}: unsupported vertex property type {ply_type!r}")
            fmt_chars.append(type_to_fmt[ply_type])
            prop_names.append(name)

        record_fmt = "<" + "".join(fmt_chars)
        record_size = struct.calcsize(record_fmt)

        try:
            x_idx = prop_names.index("x")
            y_idx = prop_names.index("y")
            z_idx = prop_names.index("z")
        except ValueError as exc:
            raise ValueError(f"{path}: vertex element missing x/y/z") from exc

        # Stream-decode all vertices.
        buf = fp.read(record_size * n_vertices)
        if len(buf) < record_size * n_vertices:
            raise ValueError(f"{path}: truncated vertex payload")

        xyz = np.empty((n_vertices, 3), dtype=np.float32)
        for i in range(n_vertices):
            record = struct.unpack_from(record_fmt, buf, i * record_size)
            xyz[i, 0] = record[x_idx]
            xyz[i, 1] = record[y_idx]
            xyz[i, 2] = record[z_idx]
    return xyz


def _build_instance_table(
    scan_id: str,
    scans_dir: Optional[Path],
) -> Dict[int, GTInstance]:
    agg = json.loads(aggregation_path(scan_id, scans_dir).read_text())
    segs = json.loads(segs_path(scan_id, scans_dir).read_text())
    seg_indices = np.asarray(segs["segIndices"], dtype=np.int64)
    vertices = _read_ply_vertices_xyz(mesh_path(scan_id, scans_dir))
    if vertices.shape[0] != seg_indices.shape[0]:
        raise ValueError(
            f"{scan_id}: vertex count mismatch (ply={vertices.shape[0]} vs "
            f"segs={seg_indices.shape[0]})"
        )

    out: Dict[int, GTInstance] = {}
    for sg in agg["segGroups"]:
        instance_id = int(sg["objectId"])
        label = str(sg.get("label", ""))
        member_segs = np.asarray(sg.get("segments", []), dtype=np.int64)
        if member_segs.size == 0:
            continue
        mask = np.isin(seg_indices, member_segs)
        verts = vertices[mask]
        if verts.shape[0] == 0:
            continue
        bbox_min = verts.min(axis=0).astype(np.float32)
        bbox_max = verts.max(axis=0).astype(np.float32)
        out[instance_id] = GTInstance(
            instance_id=instance_id,
            label=label,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            n_vertices=int(verts.shape[0]),
        )
    return out


def load_scene_gt_points(
    scan_id: str,
    *,
    scans_dir: Optional[Path] = None,
) -> Dict[int, GTInstancePoints]:
    """Return ``{instance_id: GTInstancePoints}`` for visible-mask scoring.

    This intentionally does not cache the full vertex arrays on disk: it is
    used by score-time code once per scene and the mesh/segs parse is already
    cheap relative to RGB-D mask projection.
    """

    agg = json.loads(aggregation_path(scan_id, scans_dir).read_text())
    segs = json.loads(segs_path(scan_id, scans_dir).read_text())
    seg_indices = np.asarray(segs["segIndices"], dtype=np.int64)
    vertices = _read_ply_vertices_xyz(mesh_path(scan_id, scans_dir))
    if vertices.shape[0] != seg_indices.shape[0]:
        raise ValueError(
            f"{scan_id}: vertex count mismatch (ply={vertices.shape[0]} vs "
            f"segs={seg_indices.shape[0]})"
        )

    out: Dict[int, GTInstancePoints] = {}
    for sg in agg["segGroups"]:
        instance_id = int(sg["objectId"])
        label = str(sg.get("label", ""))
        member_segs = np.asarray(sg.get("segments", []), dtype=np.int64)
        if member_segs.size == 0:
            continue
        mask = np.isin(seg_indices, member_segs)
        pts = vertices[mask].astype(np.float32, copy=False)
        if pts.shape[0] == 0:
            continue
        out[instance_id] = GTInstancePoints(
            instance_id=instance_id,
            label=label,
            points=pts,
        )
    return out


def _save_cache(path: Path, table: Dict[int, GTInstance]) -> None:
    instance_ids = np.array(sorted(table.keys()), dtype=np.int64)
    labels = np.array([table[i].label for i in instance_ids], dtype=object)
    bbox_min = np.stack([table[i].bbox_min for i in instance_ids]).astype(np.float32)
    bbox_max = np.stack([table[i].bbox_max for i in instance_ids]).astype(np.float32)
    n_vertices = np.array([table[i].n_vertices for i in instance_ids], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        instance_ids=instance_ids,
        labels=labels,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_vertices=n_vertices,
    )


def _load_cache(path: Path) -> Dict[int, GTInstance]:
    with np.load(path, allow_pickle=True) as data:
        ids = data["instance_ids"]
        labels = data["labels"]
        bbox_min = data["bbox_min"]
        bbox_max = data["bbox_max"]
        n_vertices = data["n_vertices"]
    return {
        int(ids[k]): GTInstance(
            instance_id=int(ids[k]),
            label=str(labels[k]),
            bbox_min=np.asarray(bbox_min[k], dtype=np.float32),
            bbox_max=np.asarray(bbox_max[k], dtype=np.float32),
            n_vertices=int(n_vertices[k]),
        )
        for k in range(len(ids))
    }


def load_scene_gt(
    scan_id: str,
    *,
    scans_dir: Optional[Path] = None,
    use_cache: bool = True,
    rebuild: bool = False,
) -> Dict[int, GTInstance]:
    """Return ``{instance_id: GTInstance}`` for ``scan_id``.

    First call parses aggregation/segs/ply (~1s/scene) and writes a small NPZ
    cache next to the scan; subsequent calls read the cache (~ms). A read-only
    scans directory just skips the cache write.
    """
    cache = cache_path(scan_id, scans_dir)
    if use_cache and not rebuild and cache.exists():
        return _load_cache(cache)
    table = _build_instance_table(scan_id, scans_dir)
    if use_cache:
        try:
            _save_cache(cache, table)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "GT cache write skipped (scans dir not writable): %s", exc
            )
    return table


def build_caches(
    scan_ids: list[str],
    *,
    scans_dir: Optional[Path] = None,
    rebuild: bool = False,
) -> Tuple[int, list[Tuple[str, str]]]:
    """Build NPZ caches for many scans. Returns ``(n_built, failures)``."""
    n_built = 0
    failures: list[Tuple[str, str]] = []
    for sid in scan_ids:
        try:
            load_scene_gt(sid, scans_dir=scans_dir, use_cache=True, rebuild=rebuild)
            n_built += 1
        except Exception as exc:  # noqa: BLE001
            failures.append((sid, repr(exc)))
    return n_built, failures
