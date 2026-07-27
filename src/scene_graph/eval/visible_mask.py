"""Occlusion-aware visible-mask scoring utilities.

The saved scene graph does not always retain raw detector masks. For offline
evaluation we therefore reconstruct an observation-space mask from the evidence
that is persisted: per-object voxel support plus the image ids where the object
was observed. Ground-truth objects are represented by their annotated point
cloud or mesh vertices. Both are projected into the same RGB-D frame and kept
only where their projected depth agrees with the observed depth.
"""

from __future__ import annotations

import contextlib
import csv
import json
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scene_graph.map_update.mask_observations import load_mask_observation

_VOXEL_BITS = 21
_VOXEL_MASK = (1 << _VOXEL_BITS) - 1
_VOXEL_BIAS = 1 << (_VOXEL_BITS - 1)
_VOXEL_BASE_V = 0.005

_GLTF_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_GLTF_NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


def _hm3d_short_id(scene_id: str) -> str:
    sid = str(scene_id)
    return sid.split("-", 1)[1] if "-" in sid else sid


def resolve_hm3d_mesh_path(scene_id: str, hm3d_root: Path, *, semantic: bool = True) -> Path:
    """Resolve an HM3D ``.glb`` for a scene under common download layouts."""

    root = Path(hm3d_root).expanduser()
    short = _hm3d_short_id(scene_id)
    suffixes = ["semantic.glb", "basis.glb"] if semantic else ["basis.glb", "semantic.glb"]
    candidate_roots = [
        root,
        root / "versioned_data" / "hm3d-0.2",
        root / "hm3d",
        root / "versioned_data" / "hm3d-0.2" / "hm3d",
    ]
    for base in candidate_roots:
        for split in ("train", "val", "minival"):
            for suffix in suffixes:
                cand = base / split / str(scene_id) / f"{short}.{suffix}"
                if cand.exists():
                    return cand
                cand = base / "hm3d" / split / str(scene_id) / f"{short}.{suffix}"
                if cand.exists():
                    return cand
    for suffix in suffixes:
        matches = sorted(root.rglob(f"{short}.{suffix}")) if root.exists() else []
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no HM3D mesh for {scene_id} under {hm3d_root}")


def resolve_hm3d_semantic_txt(scene_id: str, hm3d_root: Path) -> Optional[Path]:
    root = Path(hm3d_root).expanduser()
    short = _hm3d_short_id(scene_id)
    candidate_roots = [
        root,
        root / "versioned_data" / "hm3d-0.2",
        root / "hm3d",
        root / "versioned_data" / "hm3d-0.2" / "hm3d",
    ]
    for base in candidate_roots:
        for split in ("train", "val", "minival"):
            for cand in (
                base / split / str(scene_id) / f"{short}.semantic.txt",
                base / "hm3d" / split / str(scene_id) / f"{short}.semantic.txt",
            ):
                if cand.exists():
                    return cand
    matches = sorted(root.rglob(f"{short}.semantic.txt")) if root.exists() else []
    return matches[0] if matches else None


def load_hm3d_semantic_labels(scene_id: str, hm3d_root: Path) -> Dict[int, str]:
    path = resolve_hm3d_semantic_txt(scene_id, hm3d_root)
    if path is None:
        return {}
    labels: Dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row or row[0].startswith("HM3D"):
                continue
            try:
                sem_id = int(str(row[0]).strip())
            except Exception:
                continue
            label = str(row[2]).strip() if len(row) > 2 else ""
            labels[sem_id - 1] = label
    return labels


def _read_glb_chunks(path: Path) -> Tuple[Dict[str, Any], bytes]:
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError(f"not a GLB file: {path}")
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"unsupported GLB header for {path}")
    offset = 12
    gltf: Optional[Dict[str, Any]] = None
    bin_chunk = b""
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = bytes(chunk)
    if gltf is None:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    return gltf, bin_chunk


def _gltf_accessor_array(gltf: Mapping[str, Any], bin_chunk: bytes, accessor_idx: int) -> np.ndarray:
    accessors = gltf.get("accessors") or []
    buffer_views = gltf.get("bufferViews") or []
    accessor = accessors[int(accessor_idx)]
    view = buffer_views[int(accessor["bufferView"])]
    dtype = _GLTF_COMPONENT_DTYPES[int(accessor["componentType"])]
    ncomp = int(_GLTF_NUM_COMPONENTS[str(accessor["type"])])
    count = int(accessor["count"])
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dtype.itemsize * ncomp))
    if stride == dtype.itemsize * ncomp:
        arr = np.frombuffer(
            bin_chunk,
            dtype=dtype,
            count=count * ncomp,
            offset=byte_offset,
        ).reshape(count, ncomp)
    else:
        arr = np.ndarray(
            (count, ncomp),
            dtype=dtype,
            buffer=bin_chunk,
            offset=byte_offset,
            strides=(stride, dtype.itemsize),
        )
    if ncomp == 1:
        return arr.reshape(count).copy()
    return arr.copy()


def _object_label_tokens(label: str) -> List[str]:
    text = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    return [tok for tok in text.split() if tok]


@dataclass(frozen=True)
class EvalFrame:
    """RGB-D frame data needed for visible-mask projection."""

    image_id: int
    pose_world_cam: np.ndarray
    K: np.ndarray
    width: int
    height: int
    depth: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    source_ref: str = ""


@dataclass(frozen=True)
class MaskOverlap:
    """Binary-mask overlap statistics for one view."""

    iou: float
    precision: float
    recall: float
    intersection: int
    union: int
    pred_pixels: int
    gt_pixels: int


@dataclass(frozen=True)
class VisibleMaskMatch:
    """Aggregated visible-mask agreement for one predicted node and GT object."""

    candidate_object_id: int
    evidence_object_id: int
    gt_object_id: int
    best_iou: float
    mean_topk_iou: float
    weighted_iou: float
    best_precision: float
    best_recall: float
    n_valid_views: int
    best_image_id: Optional[int] = None

    def score(self, aggregation: str = "best_iou") -> float:
        key = str(aggregation or "best_iou").strip().lower()
        if key in {"mean_topk", "mean_topk_iou", "topk"}:
            return float(self.mean_topk_iou)
        if key in {"weighted", "weighted_iou"}:
            return float(self.weighted_iou)
        return float(self.best_iou)


def discover_scene_state_paths(scene_state_dir: Path) -> Dict[str, Path]:
    """Return ``{scene_id: path}`` for ``*.pt`` files or nested scene_state.pt."""

    root = Path(scene_state_dir).expanduser()
    out: Dict[str, Path] = {}
    if not root.exists():
        return out
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix == ".pt":
            out[entry.stem] = entry
        elif entry.is_dir():
            cand = entry / "scene_state.pt"
            if cand.exists():
                out[entry.name] = cand
    return out


def _to_numpy(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if value is None:
        arr = np.empty((0,), dtype=np.float32 if dtype is None else dtype)
    else:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def decode_voxel_keys(
    keys: np.ndarray,
    level: int,
    *,
    base_v: float = _VOXEL_BASE_V,
) -> np.ndarray:
    """Decode packed scene-graph voxel keys into world-frame voxel centers."""

    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    if keys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    qz = (keys & _VOXEL_MASK) - _VOXEL_BIAS
    qy = ((keys >> _VOXEL_BITS) & _VOXEL_MASK) - _VOXEL_BIAS
    qx = ((keys >> (2 * _VOXEL_BITS)) & _VOXEL_MASK) - _VOXEL_BIAS
    v = float(base_v) * float(1 << int(level))
    pts = np.stack([qx, qy, qz], axis=-1).astype(np.float64) * v + (0.5 * v)
    return pts.astype(np.float32)


def sample_aabb_surface_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    spacing: float = 0.03,
    max_points: int = 50000,
) -> np.ndarray:
    """Sample points on an AABB surface for datasets without GT meshes."""

    mn = np.asarray(bbox_min, dtype=np.float32).reshape(3)
    mx = np.asarray(bbox_max, dtype=np.float32).reshape(3)
    if not (np.all(np.isfinite(mn)) and np.all(np.isfinite(mx))):
        return np.zeros((0, 3), dtype=np.float32)
    extent = np.maximum(mx - mn, 1e-4)
    step = max(float(spacing), 1e-4)
    if max_points > 0:
        approx = 2.0 * (
            (extent[0] / step + 1.0) * (extent[1] / step + 1.0)
            + (extent[0] / step + 1.0) * (extent[2] / step + 1.0)
            + (extent[1] / step + 1.0) * (extent[2] / step + 1.0)
        )
        if approx > float(max_points) and approx > 0.0:
            step *= float(np.sqrt(approx / float(max_points)))
    axes = [
        np.linspace(float(mn[i]), float(mx[i]), max(2, int(np.ceil(float(extent[i]) / step)) + 1))
        for i in range(3)
    ]
    faces: List[np.ndarray] = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        grid_a, grid_b = np.meshgrid(axes[other[0]], axes[other[1]], indexing="xy")
        for value in (float(mn[axis]), float(mx[axis])):
            pts = np.zeros((grid_a.size, 3), dtype=np.float32)
            pts[:, axis] = value
            pts[:, other[0]] = grid_a.reshape(-1)
            pts[:, other[1]] = grid_b.reshape(-1)
            faces.append(pts)
    if not faces:
        return np.zeros((0, 3), dtype=np.float32)
    pts_all = np.concatenate(faces, axis=0)
    if max_points > 0 and pts_all.shape[0] > int(max_points):
        idx = np.linspace(0, pts_all.shape[0] - 1, int(max_points)).astype(np.int64)
        pts_all = pts_all[idx]
    return pts_all.astype(np.float32, copy=False)


def mask_overlap(pred_mask: np.ndarray, gt_mask: np.ndarray) -> MaskOverlap:
    """Compute IoU, precision, and recall between two binary masks."""

    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"mask shape mismatch: pred={pred.shape} gt={gt.shape}")
    inter = int(np.logical_and(pred, gt).sum())
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    return MaskOverlap(
        iou=(float(inter) / float(union)) if union > 0 else 0.0,
        precision=(float(inter) / float(pred_pixels)) if pred_pixels > 0 else 0.0,
        recall=(float(inter) / float(gt_pixels)) if gt_pixels > 0 else 0.0,
        intersection=inter,
        union=union,
        pred_pixels=pred_pixels,
        gt_pixels=gt_pixels,
    )


def _iref_to_habitat_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1).astype(np.float32)


@dataclass
class MeshObjectSurface:
    """Selected GT object triangles in the evaluator/world coordinate frame."""

    vertices_world: np.ndarray
    faces: np.ndarray
    n_source_faces: int


class HM3DSceneMesh:
    """Lazy HM3D mesh index used to extract GT object surfaces.

    IRef-VLA object ids line up with ``semantic.txt`` ids minus one, but the
    GLB geometry itself is not reliably split into one mesh per object.  We
    therefore extract the GT object's surface by selecting real scene-mesh
    triangles whose centroids/vertices fall inside the annotated oriented GT
    box, then render those triangles with a depth test.  This still evaluates
    a projected mesh surface; the box is only used to identify the object
    region in the scene mesh.
    """

    def __init__(self, scene_id: str, mesh_path: Path, *, semantic_labels: Optional[Dict[int, str]] = None) -> None:
        self.scene_id = str(scene_id)
        self.mesh_path = Path(mesh_path)
        self.semantic_labels = dict(semantic_labels or {})
        self.vertices_raw, self.faces = self._load_glb_mesh(self.mesh_path)
        self.vertices_world = _iref_to_habitat_points(self.vertices_raw)
        self._object_cache: Dict[Tuple[int, float], MeshObjectSurface] = {}

    @classmethod
    def from_hm3d_root(cls, scene_id: str, hm3d_root: Path) -> "HM3DSceneMesh":
        mesh_path = resolve_hm3d_mesh_path(scene_id, hm3d_root, semantic=True)
        labels = load_hm3d_semantic_labels(scene_id, hm3d_root)
        return cls(scene_id, mesh_path, semantic_labels=labels)

    @staticmethod
    def _load_glb_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        gltf, bin_chunk = _read_glb_chunks(path)
        meshes = gltf.get("meshes") or []
        nodes = gltf.get("nodes") or []
        vertices_all: List[np.ndarray] = []
        faces_all: List[np.ndarray] = []
        mesh_to_node_transforms: Dict[int, List[np.ndarray]] = {}

        for node in nodes:
            if "mesh" not in node:
                continue
            mesh_idx = int(node["mesh"])
            transform = np.eye(4, dtype=np.float32)
            if "matrix" in node:
                mat = np.asarray(node["matrix"], dtype=np.float32).reshape(4, 4)
                transform = mat.T if mat.shape == (4, 4) else transform
            else:
                if "scale" in node:
                    scale = np.asarray(node["scale"], dtype=np.float32).reshape(3)
                    transform[:3, :3] = transform[:3, :3] * scale.reshape(1, 3)
                if "translation" in node:
                    transform[:3, 3] = np.asarray(node["translation"], dtype=np.float32).reshape(3)
            mesh_to_node_transforms.setdefault(mesh_idx, []).append(transform)
        if not mesh_to_node_transforms:
            mesh_to_node_transforms = {idx: [np.eye(4, dtype=np.float32)] for idx in range(len(meshes))}

        for mesh_idx, mesh in enumerate(meshes):
            transforms = mesh_to_node_transforms.get(mesh_idx) or [np.eye(4, dtype=np.float32)]
            for prim in mesh.get("primitives") or []:
                attrs = prim.get("attributes") or {}
                pos_idx = attrs.get("POSITION")
                ind_idx = prim.get("indices")
                if pos_idx is None or ind_idx is None:
                    continue
                verts = _gltf_accessor_array(gltf, bin_chunk, int(pos_idx)).astype(np.float32, copy=False).reshape(-1, 3)
                inds = _gltf_accessor_array(gltf, bin_chunk, int(ind_idx)).astype(np.int64, copy=False).reshape(-1)
                if verts.size == 0 or inds.size < 3:
                    continue
                faces = inds.reshape(-1, 3)
                for transform in transforms:
                    if not np.allclose(transform, np.eye(4, dtype=np.float32)):
                        verts_h = np.concatenate(
                            [verts, np.ones((verts.shape[0], 1), dtype=np.float32)],
                            axis=1,
                        )
                        verts_t = (verts_h @ transform.T)[:, :3].astype(np.float32, copy=False)
                    else:
                        verts_t = verts
                    offset = sum(v.shape[0] for v in vertices_all)
                    vertices_all.append(verts_t.copy())
                    faces_all.append((faces + int(offset)).astype(np.int64, copy=False))
        if not vertices_all or not faces_all:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(vertices_all, axis=0), np.concatenate(faces_all, axis=0)

    @staticmethod
    def _points_inside_obb(points_raw: np.ndarray, gt_instance: Any, *, margin_m: float) -> np.ndarray:
        pts = np.asarray(points_raw, dtype=np.float32).reshape(-1, 3)
        center = np.asarray(getattr(gt_instance, "obb_center"), dtype=np.float32).reshape(3)
        extent = np.asarray(getattr(gt_instance, "obb_extent"), dtype=np.float32).reshape(3)
        heading = float(getattr(gt_instance, "obb_heading"))
        rel = pts - center.reshape(1, 3)
        c = float(np.cos(heading))
        s = float(np.sin(heading))
        local_x = c * rel[:, 0] + s * rel[:, 1]
        local_y = -s * rel[:, 0] + c * rel[:, 1]
        local_z = rel[:, 2]
        half = 0.5 * extent + float(margin_m)
        return (
            (np.abs(local_x) <= float(half[0]))
            & (np.abs(local_y) <= float(half[1]))
            & (np.abs(local_z) <= float(half[2]))
        )

    def object_surface(self, gt_instance: Any, *, margin_m: float = 0.02) -> MeshObjectSurface:
        object_id = int(getattr(gt_instance, "instance_id"))
        key = (object_id, round(float(margin_m), 4))
        cached = self._object_cache.get(key)
        if cached is not None:
            return cached
        if self.faces.size == 0 or self.vertices_raw.size == 0:
            surface = MeshObjectSurface(
                vertices_world=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int64),
                n_source_faces=0,
            )
            self._object_cache[key] = surface
            return surface

        tri_raw = self.vertices_raw[self.faces]
        centers = tri_raw.mean(axis=1)
        center_inside = self._points_inside_obb(centers, gt_instance, margin_m=margin_m)
        # Thin objects can have centroids just outside the annotation; keep
        # triangles that touch the expanded OBB as well.
        vert_inside = self._points_inside_obb(tri_raw.reshape(-1, 3), gt_instance, margin_m=margin_m).reshape(-1, 3)
        keep = center_inside | vert_inside.any(axis=1)
        kept_faces = self.faces[keep]
        if kept_faces.size == 0:
            surface = MeshObjectSurface(
                vertices_world=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int64),
                n_source_faces=0,
            )
            self._object_cache[key] = surface
            return surface

        unique_vertices, inverse = np.unique(kept_faces.reshape(-1), return_inverse=True)
        vertices_world = self.vertices_world[unique_vertices].astype(np.float32, copy=False)
        faces = inverse.reshape(-1, 3).astype(np.int64, copy=False)
        surface = MeshObjectSurface(
            vertices_world=vertices_world,
            faces=faces,
            n_source_faces=int(kept_faces.shape[0]),
        )
        self._object_cache[key] = surface
        return surface


def project_world_points(
    points_world: np.ndarray,
    pose_world_cam: np.ndarray,
    K: np.ndarray,
    *,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points with a T_world_cam pose and OpenCV intrinsics."""

    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.float32),
        )
    pose = np.asarray(pose_world_cam, dtype=np.float32).reshape(4, 4)
    R = pose[:3, :3]
    t = pose[:3, 3]
    cam = (pts - t.reshape(1, 3)) @ R
    z = cam[:, 2]
    valid_z = z > 1e-5
    z_safe = np.where(valid_z, z, 1.0)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    u = float(K[0, 0]) * (cam[:, 0] / z_safe) + float(K[0, 2])
    v = float(K[1, 1]) * (cam[:, 1] / z_safe) + float(K[1, 2])
    uv = np.stack([u, v], axis=1).astype(np.float32)
    valid = (
        valid_z
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    return uv, valid, z.astype(np.float32)


def rasterize_mesh_visible_mask(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    frame: EvalFrame,
    *,
    depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
    require_depth: bool = True,
) -> np.ndarray:
    """Render mesh triangles into ``frame`` and apply observed-depth occlusion."""

    height = int(frame.height)
    width = int(frame.width)
    out = np.zeros((height, width), dtype=bool)
    verts = np.asarray(vertices_world, dtype=np.float32).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if verts.size == 0 or tri.size == 0:
        return out

    pose = np.asarray(frame.pose_world_cam, dtype=np.float32).reshape(4, 4)
    R = pose[:3, :3]
    t = pose[:3, 3]
    cam = (verts - t.reshape(1, 3)) @ R
    z = cam[:, 2]
    K = np.asarray(frame.K, dtype=np.float32).reshape(3, 3)
    z_safe = np.where(z > 1e-6, z, 1.0)
    u = float(K[0, 0]) * (cam[:, 0] / z_safe) + float(K[0, 2])
    v = float(K[1, 1]) * (cam[:, 1] / z_safe) + float(K[1, 2])
    uv = np.stack([u, v], axis=1).astype(np.float32)

    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    eps = 1e-6
    for f0, f1, f2 in tri:
        if f0 < 0 or f1 < 0 or f2 < 0 or f0 >= len(verts) or f1 >= len(verts) or f2 >= len(verts):
            continue
        z0, z1, z2 = float(z[f0]), float(z[f1]), float(z[f2])
        if z0 <= eps or z1 <= eps or z2 <= eps:
            continue
        p0 = uv[f0]
        p1 = uv[f1]
        p2 = uv[f2]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        min_x = max(0, int(np.floor(min(float(p0[0]), float(p1[0]), float(p2[0])))))
        max_x = min(width - 1, int(np.ceil(max(float(p0[0]), float(p1[0]), float(p2[0])))))
        min_y = max(0, int(np.floor(min(float(p0[1]), float(p1[1]), float(p2[1])))))
        max_y = min(height - 1, int(np.ceil(max(float(p0[1]), float(p1[1]), float(p2[1])))))
        if max_x < min_x or max_y < min_y:
            continue
        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) <= eps:
            continue
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not bool(inside.any()):
            continue
        z_interp = (w0 * z0 + w1 * z1 + w2 * z2).astype(np.float32, copy=False)
        yy_i = np.arange(min_y, max_y + 1, dtype=np.int64)[:, None]
        xx_i = np.arange(min_x, max_x + 1, dtype=np.int64)[None, :]
        current = zbuf[yy_i, xx_i]
        update = inside & (z_interp < current)
        if bool(update.any()):
            patch = current.copy()
            patch[update] = z_interp[update]
            zbuf[yy_i, xx_i] = patch

    obj_pixels = np.isfinite(zbuf)
    if not bool(obj_pixels.any()):
        return out
    if not require_depth:
        return obj_pixels.astype(bool, copy=False)
    if frame.depth is None:
        return out
    depth = np.asarray(frame.depth, dtype=np.float32)
    if depth.shape[:2] != (height, width):
        return out
    observed = depth
    valid_depth = np.isfinite(observed) & (observed > 1e-5)
    visible = obj_pixels & valid_depth & (np.abs(zbuf - observed) <= float(depth_tolerance_m))
    return visible.astype(bool, copy=False)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    radius = int(radius_px)
    if radius <= 0 or not bool(np.asarray(mask, dtype=bool).any()):
        return np.asarray(mask, dtype=bool)
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        try:
            from scipy import ndimage

            yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            structure = (xx * xx + yy * yy) <= radius * radius
            return ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=structure)
        except Exception:
            return np.asarray(mask, dtype=bool)


def resize_bool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    h = max(1, int(height))
    w = max(1, int(width))
    if src.shape == (h, w):
        return src
    if src.ndim != 2 or src.size == 0:
        return np.zeros((h, w), dtype=bool)
    try:
        import cv2

        resized = cv2.resize(src.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)
    except Exception:
        yy = np.minimum((np.arange(h) * (src.shape[0] / float(h))).astype(np.int64), src.shape[0] - 1)
        xx = np.minimum((np.arange(w) * (src.shape[1] / float(w))).astype(np.int64), src.shape[1] - 1)
        return src[yy[:, None], xx[None, :]].astype(bool, copy=False)


def visible_points_mask(
    points_world: np.ndarray,
    frame: EvalFrame,
    *,
    depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
    point_radius_px: int = 3,
    max_points: int = 50000,
    require_depth: bool = True,
    rng_seed: int = 0,
) -> np.ndarray:
    """Project points into a frame and keep only points visible in observed depth."""

    height = int(frame.height)
    width = int(frame.width)
    mask = np.zeros((height, width), dtype=bool)
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return mask
    if max_points > 0 and pts.shape[0] > int(max_points):
        rng = np.random.default_rng(int(rng_seed))
        keep = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[keep]

    uv, valid, z = project_world_points(
        pts,
        frame.pose_world_cam,
        frame.K,
        width=width,
        height=height,
    )
    if not bool(valid.any()):
        return mask

    valid_idx = np.nonzero(valid)[0]
    xy = np.rint(uv[valid_idx]).astype(np.int64)
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    z_valid = z[valid_idx]

    if frame.depth is not None:
        depth = np.asarray(frame.depth, dtype=np.float32)
        if depth.shape[:2] != (height, width):
            return mask
        observed = depth[y, x]
        depth_valid = np.isfinite(observed) & (observed > 1e-5)
        if float(depth_tolerance_m) >= 0.0:
            depth_valid &= np.abs(z_valid - observed) <= float(depth_tolerance_m)
        x = x[depth_valid]
        y = y[depth_valid]
    elif require_depth:
        return mask

    if x.size == 0:
        return mask
    mask[y, x] = True
    return _dilate_mask(mask, int(point_radius_px))


def _parse_source_ref(source_ref: str) -> Optional[Tuple[Path, int]]:
    if not source_ref:
        return None
    path_part, sep, frag = str(source_ref).partition("#")
    if not sep:
        return None
    frame_idx: Optional[int] = None
    for item in frag.split("&"):
        key, eq, value = item.partition("=")
        if eq and key == "frame":
            with contextlib.suppress(Exception):
                frame_idx = int(value)
            break
    if frame_idx is None:
        return None
    return _prefer_local_frame_path(Path(path_part)), int(frame_idx)


def _prefer_local_frame_path(path: Path) -> Path:
    """Prefer a local mirror of HM3D rendered trajectory frames when present."""

    if not path.is_absolute():
        return path
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "rendered_trajectory_magnet":
            local_root = Path("/tmp/hm3d_projected_top5_s10_rendered_trajectory_magnet")
            local_path = local_root.joinpath(*parts[idx + 1 :])
            if local_path.exists():
                return local_path
            break
    return path


def _record_get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _pose_from_record(record: Any) -> Optional[np.ndarray]:
    pose = _record_get(record, "pose")
    if pose is None:
        return None
    pose_np = _to_numpy(pose, dtype=np.float32)
    if pose_np.shape == (4, 4) and np.all(np.isfinite(pose_np)):
        return pose_np
    return None


def _default_K(width: int, height: int) -> np.ndarray:
    focal = 0.5 * float(width)
    return np.asarray(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class SceneFrameResolver:
    """Lazy resolver for ImageRecord entries stored inside a scene_state."""

    def __init__(self, images: Sequence[Any], *, max_cached_frames: int = 128) -> None:
        self._images_by_id: Dict[int, Any] = {}
        for idx, record in enumerate(images or []):
            image_id = _record_get(record, "image_id", idx)
            with contextlib.suppress(Exception):
                self._images_by_id[int(image_id)] = record
        self._frame_cache: "OrderedDict[int, Optional[EvalFrame]]" = OrderedDict()
        self._max_cached_frames = max(0, int(max_cached_frames))
        self._npz_cache: "OrderedDict[Path, Dict[str, np.ndarray]]" = OrderedDict()
        self._max_cached_npz = 4
        self._sens_cache: Dict[Path, Any] = {}

    def load(self, image_id: int) -> Optional[EvalFrame]:
        image_id_int = int(image_id)
        if image_id_int in self._frame_cache:
            frame = self._frame_cache[image_id_int]
            self._frame_cache.move_to_end(image_id_int)
            return frame

        record = self._images_by_id.get(image_id_int)
        if record is None:
            self._remember_frame(image_id_int, None)
            return None
        pose = _pose_from_record(record)
        if pose is None:
            self._remember_frame(image_id_int, None)
            return None

        source_ref = str(_record_get(record, "source_ref", "") or "")
        parsed = _parse_source_ref(source_ref)
        frame: Optional[EvalFrame] = None
        if parsed is not None:
            path, frame_idx = parsed
            suffix = path.suffix.lower()
            if suffix == ".npz":
                frame = self._load_npz_frame(path, frame_idx, image_id_int, pose, source_ref)
            elif suffix == ".sens":
                frame = self._load_sens_frame(path, frame_idx, image_id_int, pose, source_ref)

        if frame is None:
            frame = self._load_storage_frame(record, image_id_int, pose, source_ref)
        self._remember_frame(image_id_int, frame)
        return frame

    def _remember_frame(self, image_id: int, frame: Optional[EvalFrame]) -> None:
        if self._max_cached_frames <= 0:
            return
        self._frame_cache[int(image_id)] = frame
        self._frame_cache.move_to_end(int(image_id))
        while len(self._frame_cache) > self._max_cached_frames:
            self._frame_cache.popitem(last=False)

    def close(self) -> None:
        self._npz_cache.clear()
        self._frame_cache.clear()

    def frame_with_rgb(self, frame: EvalFrame) -> EvalFrame:
        """Return ``frame`` with RGB attached when the backing source has it.

        Scoring intentionally keeps RGB out of the hot path. Debug rendering
        calls this method for the handful of frames it saves.
        """

        if frame.rgb is not None:
            return frame
        rgb = self.load_rgb(int(frame.image_id))
        if rgb is None:
            return frame
        return EvalFrame(
            image_id=int(frame.image_id),
            pose_world_cam=frame.pose_world_cam,
            K=frame.K,
            width=int(frame.width),
            height=int(frame.height),
            depth=frame.depth,
            rgb=rgb,
            source_ref=frame.source_ref,
        )

    def load_rgb(self, image_id: int) -> Optional[np.ndarray]:
        """Best-effort RGB load for one frame, used only for debug artifacts."""

        record = self._images_by_id.get(int(image_id))
        if record is None:
            return None
        source_ref = str(_record_get(record, "source_ref", "") or "")
        parsed = _parse_source_ref(source_ref)
        if parsed is not None:
            path, frame_idx = parsed
            if path.suffix.lower() == ".npz":
                try:
                    with np.load(str(path), allow_pickle=False) as data:
                        if "images" not in set(data.files):
                            return None
                        image = np.asarray(data["images"][int(frame_idx)])
                    if image.ndim == 2:
                        image = np.repeat(image[..., None], 3, axis=2)
                    if image.ndim == 3 and image.shape[2] > 3:
                        image = image[:, :, :3]
                    if image.ndim == 3 and image.shape[2] == 3:
                        return image.astype(np.uint8, copy=False)
                except Exception:
                    return None

        storage_path = str(_record_get(record, "storage_path", "") or "")
        if storage_path:
            try:
                from PIL import Image

                return np.asarray(Image.open(storage_path).convert("RGB"), dtype=np.uint8)
            except Exception:
                return None
        return None

    def _load_npz_arrays(self, path: Path) -> Dict[str, np.ndarray]:
        cached = self._npz_cache.get(path)
        if cached is not None:
            self._npz_cache.move_to_end(path)
            return cached

        arrays: Dict[str, np.ndarray] = {}
        with np.load(str(path), allow_pickle=False) as data:
            files = set(data.files)
            # Depth and intrinsics are required for occlusion-aware scoring.
            # Do not eagerly load RGB frames here; random mask scoring can touch
            # many frames and RGB is only cosmetic for debug overlays.
            for key in ("depths", "K", "intrinsics"):
                if key in files:
                    arrays[key] = np.asarray(data[key])
            if "depths" not in arrays and "images" in files:
                arrays["images"] = np.asarray(data["images"])

        self._npz_cache[path] = arrays
        self._npz_cache.move_to_end(path)
        while len(self._npz_cache) > self._max_cached_npz:
            self._npz_cache.popitem(last=False)
        return arrays

    def _load_npz_frame(
        self,
        path: Path,
        frame_idx: int,
        image_id: int,
        pose: np.ndarray,
        source_ref: str,
    ) -> Optional[EvalFrame]:
        try:
            data = self._load_npz_arrays(path)
            depth = np.asarray(data["depths"][frame_idx], dtype=np.float32) if "depths" in data else None
            image = None
            if depth is not None:
                height, width = int(depth.shape[0]), int(depth.shape[1])
            elif "images" in data:
                image = np.asarray(data["images"][frame_idx])
                height, width = int(image.shape[0]), int(image.shape[1])
            else:
                return None
            if "K" in data:
                K_raw = np.asarray(data["K"], dtype=np.float32)
                K = K_raw[frame_idx] if K_raw.ndim == 3 else K_raw
                K = np.asarray(K, dtype=np.float32).reshape(3, 3)
            elif "intrinsics" in data:
                K_raw = np.asarray(data["intrinsics"], dtype=np.float32)
                K = K_raw[frame_idx] if K_raw.ndim == 3 else K_raw
                K = np.asarray(K, dtype=np.float32).reshape(3, 3)
            else:
                K = _default_K(width, height)
            return EvalFrame(
                image_id=int(image_id),
                pose_world_cam=pose,
                K=K,
                width=width,
                height=height,
                depth=depth,
                rgb=image.astype(np.uint8, copy=False) if image is not None else None,
                source_ref=source_ref,
            )
        except Exception:
            return None

    def _load_sens_frame(
        self,
        path: Path,
        frame_idx: int,
        image_id: int,
        pose: np.ndarray,
        source_ref: str,
    ) -> Optional[EvalFrame]:
        try:
            from scene_graph.offline.frame_sources.sens import _decode_depth, _scan_sens

            if path not in self._sens_cache:
                header, entries, _poses = _scan_sens(str(path))
                self._sens_cache[path] = (header, entries)
            header, entries = self._sens_cache[path]
            if frame_idx < 0 or frame_idx >= len(entries):
                return None
            entry = entries[frame_idx]
            with path.open("rb") as fp:
                fp.seek(entry.depth_offset)
                raw = fp.read(entry.depth_size)
            depth_u16 = _decode_depth(
                raw,
                header.depth_compression,
                int(header.depth_height),
                int(header.depth_width),
            )
            depth = depth_u16.astype(np.float32) / float(header.depth_shift)
            K = np.asarray(header.intrinsic_depth[:3, :3], dtype=np.float32)
            return EvalFrame(
                image_id=int(image_id),
                pose_world_cam=pose,
                K=K,
                width=int(depth.shape[1]),
                height=int(depth.shape[0]),
                depth=depth,
                rgb=None,
                source_ref=source_ref,
            )
        except Exception:
            return None

    def _load_storage_frame(
        self,
        record: Any,
        image_id: int,
        pose: np.ndarray,
        source_ref: str,
    ) -> Optional[EvalFrame]:
        storage_path = str(_record_get(record, "storage_path", "") or "")
        if not storage_path:
            return None
        try:
            from PIL import Image

            image = Image.open(storage_path)
            width, height = image.size
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return EvalFrame(
                image_id=int(image_id),
                pose_world_cam=pose,
                K=_default_K(width, height),
                width=int(width),
                height=int(height),
                depth=None,
                rgb=rgb,
                source_ref=source_ref,
            )
        except Exception:
            return None


class GTMeshMaskProvider:
    """Caches GT mesh render masks for one scene."""

    def __init__(
        self,
        scene_mesh: HM3DSceneMesh,
        *,
        max_cached_masks: int = 512,
        object_margin_m: float = 0.02,
    ) -> None:
        self.scene_mesh = scene_mesh
        self._cache: "OrderedDict[Tuple[int, int, float, bool, float], np.ndarray]" = OrderedDict()
        self._max_cached_masks = max(0, int(max_cached_masks))
        self.object_margin_m = float(object_margin_m)

    @classmethod
    def from_hm3d_root(
        cls,
        scene_id: str,
        hm3d_root: Path,
        *,
        max_cached_masks: int = 512,
        object_margin_m: float = 0.02,
    ) -> "GTMeshMaskProvider":
        return cls(
            HM3DSceneMesh.from_hm3d_root(scene_id, hm3d_root),
            max_cached_masks=max_cached_masks,
            object_margin_m=object_margin_m,
        )

    def clear(self) -> None:
        self._cache.clear()

    def render_mask(
        self,
        gt_instance: Any,
        frame: EvalFrame,
        *,
        depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
        require_depth: bool = True,
    ) -> np.ndarray:
        gt_id = int(getattr(gt_instance, "instance_id"))
        key = (
            gt_id,
            int(frame.image_id),
            round(float(depth_tolerance_m), 4),
            bool(require_depth),
            round(float(self.object_margin_m), 4),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        surface = self.scene_mesh.object_surface(gt_instance, margin_m=self.object_margin_m)
        mask = rasterize_mesh_visible_mask(
            surface.vertices_world,
            surface.faces,
            frame,
            depth_tolerance_m=depth_tolerance_m,
            require_depth=require_depth,
        )
        if self._max_cached_masks > 0:
            self._cache[key] = mask
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_cached_masks:
                self._cache.popitem(last=False)
        return mask


class SceneStateMaskIndex:
    """Fast access to persisted object evidence for visible-mask scoring."""

    def __init__(
        self,
        state: Dict[str, Any],
        *,
        state_path: Optional[Path] = None,
        max_cached_frames: int = 128,
        max_cached_masks: int = 512,
    ) -> None:
        self.state = state
        self.state_path = Path(state_path) if state_path is not None else None
        self.state_root = self.state_path.parent if self.state_path is not None else None
        object_ids = _to_numpy(state.get("object_id"), dtype=np.int64).reshape(-1)
        self.object_ids = object_ids
        self.object_id_to_index: Dict[int, int] = {int(oid): idx for idx, oid in enumerate(object_ids.tolist())}
        self.object_image_ids = state.get("object_image_ids") or []
        self.viewpoint_image_ids = state.get("viewpoint_image_ids") or []
        self.object_mask_observations = state.get("object_mask_observations") or []
        self.frame_resolver = SceneFrameResolver(state.get("images") or [], max_cached_frames=max_cached_frames)
        self._points_cache: Dict[int, np.ndarray] = {}
        self._visible_mask_cache: "OrderedDict[Tuple[str, int, int, float, int, int, bool], np.ndarray]" = OrderedDict()
        self._max_cached_masks = max(0, int(max_cached_masks))

        flat = state.get("object_voxel_keys_flat")
        offsets = state.get("object_voxel_keys_offsets")
        levels = state.get("object_voxel_levels")
        self._flat = _to_numpy(flat, dtype=np.int64).reshape(-1) if flat is not None else np.empty((0,), dtype=np.int64)
        self._offsets = (
            _to_numpy(offsets, dtype=np.int64).reshape(-1)
            if offsets is not None
            else np.zeros((len(object_ids) + 1,), dtype=np.int64)
        )
        self._levels = (
            _to_numpy(levels, dtype=np.int64).reshape(-1)
            if levels is not None
            else np.zeros((len(object_ids),), dtype=np.int64)
        )

    @classmethod
    def from_path(cls, path: Path) -> "SceneStateMaskIndex":
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        state = payload.get("state", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError(f"unsupported scene_state payload at {path}")
        return cls(state, state_path=Path(path))

    def close(self) -> None:
        self.frame_resolver.close()
        self._points_cache.clear()
        self._visible_mask_cache.clear()

    def clear_visible_mask_cache(self) -> None:
        """Drop projected mask cache entries while keeping decoded object points."""

        self._visible_mask_cache.clear()

    def candidate_evidence_object_id(self, candidate: Dict[str, Any]) -> int:
        """Use alias source id when a ranked candidate came from alias expansion."""

        label = str(candidate.get("label") or "")
        if label.startswith("alias:"):
            parts = label.split(":")
            if len(parts) >= 2:
                with contextlib.suppress(Exception):
                    return int(parts[1])
        return int(candidate.get("object_id", -1))

    def object_points(self, object_id: int) -> np.ndarray:
        oid = int(object_id)
        cached = self._points_cache.get(oid)
        if cached is not None:
            return cached
        idx = self.object_id_to_index.get(oid)
        if idx is None or idx + 1 >= len(self._offsets):
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            start, end = int(self._offsets[idx]), int(self._offsets[idx + 1])
            if 0 <= start < end <= len(self._flat):
                level = int(self._levels[idx]) if idx < len(self._levels) else 0
                pts = decode_voxel_keys(self._flat[start:end], level)
            else:
                pts = np.zeros((0, 3), dtype=np.float32)
        self._points_cache[oid] = pts
        return pts

    def object_view_ids(self, object_id: int, *, max_views: Optional[int] = None) -> List[int]:
        idx = self.object_id_to_index.get(int(object_id))
        if idx is None:
            return []
        raw: Sequence[Any] = []
        if idx < len(self.object_image_ids) and isinstance(self.object_image_ids[idx], (list, tuple)):
            raw = self.object_image_ids[idx]
        if not raw and idx < len(self.viewpoint_image_ids) and isinstance(self.viewpoint_image_ids[idx], (list, tuple)):
            raw = self.viewpoint_image_ids[idx]

        out: List[int] = []
        seen: set[int] = set()
        for value in raw:
            try:
                image_id = int(value)
            except Exception:
                continue
            if image_id in seen:
                continue
            seen.add(image_id)
            out.append(image_id)
            if max_views is not None and int(max_views) > 0 and len(out) >= int(max_views):
                break
        return out

    def object_mask_observation_records(
        self,
        object_id: int,
        *,
        max_views: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        idx = self.object_id_to_index.get(int(object_id))
        if idx is None or idx >= len(self.object_mask_observations):
            return []
        raw = self.object_mask_observations[idx]
        if not isinstance(raw, (list, tuple)):
            return []
        out: List[Dict[str, Any]] = []
        seen_images: set[int] = set()
        for item in reversed(list(raw)):
            if not isinstance(item, dict):
                continue
            try:
                image_id = int(item.get("image_id"))
            except Exception:
                continue
            if image_id in seen_images:
                continue
            seen_images.add(image_id)
            out.append(dict(item))
            if max_views is not None and int(max_views) > 0 and len(out) >= int(max_views):
                break
        return out

    def load_predicted_observation_mask(
        self,
        observation: Mapping[str, Any],
        *,
        kind: str = "raw",
        frame: Optional[EvalFrame] = None,
    ) -> Optional[np.ndarray]:
        mask = load_mask_observation(observation, kind=kind, root=self.state_root)
        if mask is None:
            return None
        if frame is not None and mask.shape != (int(frame.height), int(frame.width)):
            mask = resize_bool_mask(mask, int(frame.height), int(frame.width))
        return mask

    def _visible_mask_cached(
        self,
        kind: str,
        object_id: int,
        points: np.ndarray,
        frame: EvalFrame,
        *,
        depth_tolerance_m: float,
        point_radius_px: int,
        max_points: int,
        require_depth: bool,
    ) -> np.ndarray:
        key = (
            str(kind),
            int(object_id),
            int(frame.image_id),
            round(float(depth_tolerance_m), 4),
            int(point_radius_px),
            int(max_points),
            bool(require_depth),
        )
        cached = self._visible_mask_cache.get(key)
        if cached is not None:
            self._visible_mask_cache.move_to_end(key)
            return cached
        mask = visible_points_mask(
            points,
            frame,
            depth_tolerance_m=depth_tolerance_m,
            point_radius_px=point_radius_px,
            max_points=max_points,
            require_depth=require_depth,
            rng_seed=int(object_id) * 1000003 + int(frame.image_id),
        )
        if self._max_cached_masks > 0:
            self._visible_mask_cache[key] = mask
            self._visible_mask_cache.move_to_end(key)
            while len(self._visible_mask_cache) > self._max_cached_masks:
                self._visible_mask_cache.popitem(last=False)
        return mask

    def score_candidate(
        self,
        candidate: Dict[str, Any],
        gt_object_id: int,
        gt_points: Optional[np.ndarray] = None,
        *,
        gt_instance: Any = None,
        gt_mask_provider: Optional[GTMeshMaskProvider] = None,
        pred_mask_kind: str = "raw",
        allow_pred_point_projection: bool = False,
        depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
        point_radius_px: int = 3,
        min_gt_pixels: int = 20,
        topk: int = 3,
        max_views: Optional[int] = None,
        max_points: int = 50000,
        require_depth: bool = True,
        chosen_view_image_id: Optional[int] = None,
    ) -> VisibleMaskMatch:
        candidate_object_id = int(candidate.get("object_id", -1))
        evidence_object_id = self.candidate_evidence_object_id(candidate)

        overlaps: List[Tuple[MaskOverlap, int]] = []
        if chosen_view_image_id is not None:
            observations = self.object_mask_observation_records(evidence_object_id, max_views=None)
            observations = [
                obs for obs in observations
                if int(obs.get("image_id", -1)) == int(chosen_view_image_id)
            ]
        else:
            observations = self.object_mask_observation_records(evidence_object_id, max_views=max_views)
        if observations:
            iterable: Iterable[Tuple[int, Optional[Mapping[str, Any]]]] = [
                (int(obs["image_id"]), obs) for obs in observations
            ]
        elif allow_pred_point_projection:
            view_ids = self.object_view_ids(evidence_object_id, max_views=max_views)
            if chosen_view_image_id is not None:
                view_ids = [v for v in view_ids if int(v) == int(chosen_view_image_id)]
            iterable = [(image_id, None) for image_id in view_ids]
        else:
            iterable = []

        pred_points: Optional[np.ndarray] = None
        gt_pts: Optional[np.ndarray] = None
        for image_id, observation in iterable:
            frame = self.frame_resolver.load(image_id)
            if frame is None:
                continue
            if gt_mask_provider is not None and gt_instance is not None:
                gt_mask = gt_mask_provider.render_mask(
                    gt_instance,
                    frame,
                    depth_tolerance_m=depth_tolerance_m,
                    require_depth=require_depth,
                )
            else:
                if gt_pts is None:
                    gt_pts = np.asarray(gt_points, dtype=np.float32).reshape(-1, 3)
                gt_mask = self._visible_mask_cached(
                    "gt",
                    int(gt_object_id),
                    gt_pts,
                    frame,
                    depth_tolerance_m=depth_tolerance_m,
                    point_radius_px=point_radius_px,
                    max_points=max_points,
                    require_depth=require_depth,
                )
            if int(gt_mask.sum()) < int(min_gt_pixels):
                continue
            if observation is not None:
                pred_mask = self.load_predicted_observation_mask(
                    observation,
                    kind=pred_mask_kind,
                    frame=frame,
                )
                if pred_mask is None:
                    continue
            else:
                if pred_points is None:
                    pred_points = self.object_points(evidence_object_id)
                pred_mask = self._visible_mask_cached(
                    "pred",
                    int(evidence_object_id),
                    pred_points,
                    frame,
                    depth_tolerance_m=depth_tolerance_m,
                    point_radius_px=point_radius_px,
                    max_points=max_points,
                    require_depth=require_depth,
                )
            overlaps.append((mask_overlap(pred_mask, gt_mask), int(image_id)))

        if not overlaps:
            return VisibleMaskMatch(
                candidate_object_id=candidate_object_id,
                evidence_object_id=int(evidence_object_id),
                gt_object_id=int(gt_object_id),
                best_iou=0.0,
                mean_topk_iou=0.0,
                weighted_iou=0.0,
                best_precision=0.0,
                best_recall=0.0,
                n_valid_views=0,
                best_image_id=None,
            )

        overlaps_sorted = sorted(overlaps, key=lambda item: item[0].iou, reverse=True)
        k = max(1, min(int(topk), len(overlaps_sorted)))
        best, best_image_id = overlaps_sorted[0]
        top_ious = [item[0].iou for item in overlaps_sorted[:k]]
        total_gt = sum(item[0].gt_pixels for item in overlaps_sorted)
        weighted = (
            sum(item[0].iou * float(item[0].gt_pixels) for item in overlaps_sorted) / float(total_gt)
            if total_gt > 0
            else 0.0
        )
        return VisibleMaskMatch(
            candidate_object_id=candidate_object_id,
            evidence_object_id=int(evidence_object_id),
            gt_object_id=int(gt_object_id),
            best_iou=float(best.iou),
            mean_topk_iou=float(np.mean(top_ious)) if top_ious else 0.0,
            weighted_iou=float(weighted),
            best_precision=float(best.precision),
            best_recall=float(best.recall),
            n_valid_views=len(overlaps_sorted),
            best_image_id=int(best_image_id),
        )

    def masks_for_candidate_view(
        self,
        candidate: Dict[str, Any],
        gt_object_id: int,
        image_id: int,
        *,
        gt_instance: Any = None,
        gt_mask_provider: Optional[GTMeshMaskProvider] = None,
        gt_points: Optional[np.ndarray] = None,
        pred_mask_kind: str = "raw",
        depth_tolerance_m: float = 0.15,  # locked 2026-05-16: matches unified scorer / legacy protocol
        point_radius_px: int = 3,
        max_points: int = 50000,
        require_depth: bool = True,
    ) -> Optional[Tuple[EvalFrame, np.ndarray, np.ndarray]]:
        evidence_object_id = self.candidate_evidence_object_id(candidate)
        frame = self.frame_resolver.load(int(image_id))
        if frame is None:
            return None
        gt_mask: np.ndarray
        if gt_mask_provider is not None and gt_instance is not None:
            gt_mask = gt_mask_provider.render_mask(
                gt_instance,
                frame,
                depth_tolerance_m=depth_tolerance_m,
                require_depth=require_depth,
            )
        else:
            pts = np.asarray(gt_points, dtype=np.float32).reshape(-1, 3)
            gt_mask = self._visible_mask_cached(
                "gt",
                int(gt_object_id),
                pts,
                frame,
                depth_tolerance_m=depth_tolerance_m,
                point_radius_px=point_radius_px,
                max_points=max_points,
                require_depth=require_depth,
            )

        pred_mask = None
        for obs in self.object_mask_observation_records(evidence_object_id, max_views=None):
            with contextlib.suppress(Exception):
                if int(obs.get("image_id")) != int(image_id):
                    continue
                pred_mask = self.load_predicted_observation_mask(obs, kind=pred_mask_kind, frame=frame)
                if pred_mask is not None:
                    break
        if pred_mask is None:
            return None
        return frame, pred_mask, gt_mask


def save_mask_debug_image(
    path: Path,
    frame: EvalFrame,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    title: str = "",
    metadata_lines: Optional[Sequence[str]] = None,
) -> None:
    """Save a compact RGB/GT/pred/combined debug panel."""

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    h, w = int(frame.height), int(frame.width)
    if frame.rgb is not None and frame.rgb.shape[:2] == (h, w):
        base = np.asarray(frame.rgb, dtype=np.uint8)
        if base.ndim == 2:
            base = np.repeat(base[..., None], 3, axis=2)
        if base.shape[2] > 3:
            base = base[:, :, :3]
    elif frame.depth is not None:
        depth = np.asarray(frame.depth, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        gray = np.zeros((h, w), dtype=np.uint8)
        if bool(valid.any()):
            lo, hi = np.percentile(depth[valid], [2, 98])
            denom = max(float(hi - lo), 1e-6)
            gray = np.clip((depth - lo) / denom * 255.0, 0, 255).astype(np.uint8)
        base = np.repeat(gray[..., None], 3, axis=2)
    else:
        base = np.zeros((h, w, 3), dtype=np.uint8)

    pred = resize_bool_mask(pred_mask, h, w)
    gt = resize_bool_mask(gt_mask, h, w)

    def overlay(mask_a: np.ndarray, color: Tuple[int, int, int], mask_b: Optional[np.ndarray] = None) -> np.ndarray:
        canvas = base.copy()
        alpha = 0.55
        if mask_a.any():
            c = np.asarray(color, dtype=np.float32)
            canvas[mask_a] = ((1.0 - alpha) * canvas[mask_a].astype(np.float32) + alpha * c).astype(np.uint8)
        if mask_b is not None and mask_b.any():
            c = np.asarray((255, 0, 255), dtype=np.float32)
            canvas[mask_b] = ((1.0 - alpha) * canvas[mask_b].astype(np.float32) + alpha * c).astype(np.uint8)
        both = mask_a & mask_b if mask_b is not None else np.zeros_like(mask_a)
        if both.any():
            canvas[both] = np.asarray((255, 230, 0), dtype=np.uint8)
        return canvas

    panels = [
        base,
        overlay(gt, (0, 220, 80)),
        overlay(pred, (255, 0, 255)),
        overlay(gt, (0, 220, 80), pred),
    ]
    gap = 6
    label_h = 24
    out_w = w * 4 + gap * 3
    wrapped_meta: List[str] = []
    if metadata_lines:
        max_chars = max(48, (out_w - 20) // 7)
        for raw_line in metadata_lines:
            text = str(raw_line).encode("ascii", "replace").decode("ascii")
            while len(text) > max_chars:
                cut = text.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                wrapped_meta.append(text[:cut].rstrip())
                text = text[cut:].lstrip()
            wrapped_meta.append(text)
    meta_h = 10 + 16 * len(wrapped_meta) if wrapped_meta else 0
    out = Image.new("RGB", (out_w, h + label_h + meta_h), (20, 20, 20))
    labels = ["rgb/depth", "gt mesh", "our mask", "combined"]
    for idx, panel in enumerate(panels):
        x = idx * (w + gap)
        out.paste(Image.fromarray(panel), (x, label_h))
    draw = ImageDraw.Draw(out)
    for idx, label in enumerate(labels):
        draw.text((idx * (w + gap) + 5, 5), label, fill=(240, 240, 240))
    if title:
        draw.text((w * 2, 5), str(title)[:120], fill=(240, 240, 240))
    if wrapped_meta:
        y0 = label_h + h + 5
        for line in wrapped_meta:
            draw.text((8, y0), line, fill=(240, 240, 240))
            y0 += 16
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def first_hit_rank_from_scores(
    scores: Sequence[float],
    threshold: float,
) -> Optional[int]:
    """Return first 1-based rank with score >= threshold."""

    thr = float(threshold)
    for idx, value in enumerate(scores, start=1):
        if float(value) >= thr:
            return idx
    return None


def summarize_visible_mask_matches(matches: Sequence[VisibleMaskMatch], *, topk: int = 3) -> Dict[str, Any]:
    """Small helper for debug payloads and smoke tests."""

    vals = list(matches)
    if not vals:
        return {
            "n": 0,
            "best_iou": 0.0,
            "mean_topk_iou": 0.0,
            "weighted_iou": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
        }
    vals_sorted = sorted(vals, key=lambda item: item.best_iou, reverse=True)
    k = max(1, min(int(topk), len(vals_sorted)))
    return {
        "n": int(len(vals_sorted)),
        "best_iou": float(vals_sorted[0].best_iou),
        "mean_topk_iou": float(np.mean([m.best_iou for m in vals_sorted[:k]])),
        "weighted_iou": float(np.mean([m.weighted_iou for m in vals_sorted])),
        "best_precision": float(vals_sorted[0].best_precision),
        "best_recall": float(vals_sorted[0].best_recall),
        "best_image_id": vals_sorted[0].best_image_id,
    }
