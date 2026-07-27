#!/usr/bin/env python3
"""Convert a raw LiDAR + fisheye-camera rosbag into a ``frames.json`` scene.

The offline pipeline (``--source frames-json``) and the online ROS nodes both
need per-frame **RGB + metric depth + pose**. A LiDAR rig (e.g. Berkeley Odin1)
records RGB + a LiDAR ``PointCloud2`` + odometry, but **no depth image** — so
its raw bag is not directly ingestible. This standalone tool bridges that gap:
for each RGB frame it interpolates the pose, gathers nearby LiDAR scans,
projects them into the camera, and writes a synthesized metric depth image,
producing a ``frames.json`` directory the pipeline consumes unchanged:

    <out>/frames.json
    <out>/_rgb/<cam>/NNNNNN.jpeg      # rectified pinhole RGB
    <out>/depths/<cam>/NNNNNN.npy     # float32 metres, NaN where no LiDAR hit
    <out>/cloud.npz                   # aggregated world cloud (viz)

Then reconstruct:

    python -m scene_graph.offline.run --source frames-json \\
        --frames-json-dir <out> --save-path scene.pt --covisibility

Assumptions (Odin1 defaults, overridable): the cloud topic is already in the
world/``odom`` frame; the calibration YAML provides ``Tcl_0`` (camera-from-LiDAR
extrinsic) and a FishPoly fisheye model whose ``A11/A22/u0/v0`` double as the
pinhole intrinsics after rectification. Camera model math is ported cleanly
from the paper's Odin1 adapter.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# --------------------------------------------------------------------------- #
# Calibration + FishPoly camera model
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    image_width: int
    image_height: int
    A11: float
    A12: float
    A22: float
    u0: float
    v0: float
    k: List[float]              # k2..k7 FishPoly coeffs
    T_camera_base: np.ndarray   # 4x4, camera-from-LiDAR/base (Tcl_0)

    @staticmethod
    def from_yaml(path: Path) -> "Calibration":
        raw = yaml.safe_load(Path(path).read_text())
        cam = raw["cam_0"]
        tcl = raw.get("Tcl_0")
        if tcl is None:
            raise ValueError(f"{path}: missing 'Tcl_0' extrinsic")
        return Calibration(
            image_width=int(cam["image_width"]),
            image_height=int(cam["image_height"]),
            A11=float(cam["A11"]),
            A12=float(cam.get("A12", 0.0)),
            A22=float(cam["A22"]),
            u0=float(cam["u0"]),
            v0=float(cam["v0"]),
            k=[float(cam[f"k{i}"]) for i in (2, 3, 4, 5, 6, 7)],
            T_camera_base=np.asarray(tcl, dtype=np.float64).reshape(4, 4),
        )

    def pinhole_K(self) -> np.ndarray:
        return np.array(
            [[self.A11, 0.0, self.u0], [0.0, self.A22, self.v0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def project_fishpoly(pts_cam: np.ndarray, calib: Calibration) -> np.ndarray:
    """Camera-frame points -> FishPoly pixel coords (NaN where behind camera)."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    theta = np.arctan2(np.sqrt(X * X + Y * Y), Z)
    phi = np.arctan2(Y, X)
    theta_d = theta.copy()
    power = theta.copy()
    for k in calib.k:
        power = power * theta
        theta_d = theta_d + float(k) * power
    xd, yd = theta_d * np.cos(phi), theta_d * np.sin(phi)
    u = calib.A11 * xd + calib.A12 * yd + calib.u0
    v = calib.A22 * yd + calib.v0
    behind = Z <= 0
    if np.any(behind):
        u, v = u.copy(), v.copy()
        u[behind] = np.nan
        v[behind] = np.nan
    return np.stack([u, v], axis=-1)


def make_fishpoly_rectification_maps(calib: Calibration, K: np.ndarray):
    """Maps from pinhole output pixels -> source FishPoly pixels (cv2.remap)."""
    h, w = calib.image_height, calib.image_width
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    rays = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    pix = project_fishpoly(rays, calib).reshape(h, w, 2)
    return pix[..., 0].astype(np.float32), pix[..., 1].astype(np.float32)


def project_world_pinhole(pts_world, T_world_cam, K):
    """World points -> pinhole pixels + camera-frame z depth."""
    if pts_world.size == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    T_cam_world = np.linalg.inv(T_world_cam)
    ones = np.ones((pts_world.shape[0], 1))
    pc = (T_cam_world @ np.concatenate([pts_world, ones], axis=1).T).T[:, :3]
    z = pc[:, 2]
    safe = z > 1e-6
    u = np.full_like(z, np.nan)
    v = np.full_like(z, np.nan)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u[safe] = fx * (pc[safe, 0] / z[safe]) + cx
    v[safe] = fy * (pc[safe, 1] / z[safe]) + cy
    return np.stack([u, v], axis=-1), z


def rasterize_zbuffer(pixels, depth_z, W, H, *, min_depth=0.1, max_depth=60.0) -> np.ndarray:
    """Splat (pixels, z) into an (H, W) float32 z-buffer; nearest point wins."""
    out = np.full((H, W), np.nan, dtype=np.float32)
    if pixels.shape[0] == 0:
        return out
    u, v, z = pixels[:, 0], pixels[:, 1], depth_z
    keep = (np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
            & (z >= float(min_depth)) & (z <= float(max_depth)))
    u, v, z = u[keep], v[keep], z[keep]
    iu, iv = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    inb = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    iu, iv, z = iu[inb], iv[inb], z[inb]
    if iu.size == 0:
        return out
    flat = iv * W + iu
    order = np.lexsort((-z, flat))          # nearest (smallest z) written last -> wins
    out.reshape(-1)[flat[order]] = z[order]
    return out


def dilate_sparse_depth(depth: np.ndarray, radius_px: int) -> np.ndarray:
    """Fill invalid pixels from the nearest valid depth within radius_px."""
    if radius_px <= 0:
        return depth
    src = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(src) & (src > 0.0)
    if not np.any(valid) or np.all(valid):
        return src
    dist, labels = cv2.distanceTransformWithLabels(
        np.where(valid, 0, 255).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    max_label = int(labels.max())
    if max_label <= 0:
        return src
    label_depth = np.full(max_label + 1, np.nan, dtype=np.float32)
    label_depth[labels[valid]] = src[valid]
    nearest = label_depth[labels]
    fill = (~valid) & (dist <= float(radius_px)) & np.isfinite(nearest) & (nearest > 0.0)
    out = src.copy()
    out[fill] = nearest[fill]
    return out


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 -> (N,3) float32 XYZ (offset-based, NaN-filtered)."""
    fields = {f.name: f.offset for f in msg.fields}
    if not all(c in fields for c in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)
    buf = bytes(msg.data)
    step = int(msg.point_step)
    n = int(msg.width) * int(msg.height)
    if n == 0 or step <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.frombuffer(buf, dtype=np.uint8)
    n = min(n, arr.size // step)
    arr = arr[: n * step].reshape(n, step)

    def col(name):
        o = fields[name]
        return np.frombuffer(arr[:, o : o + 4].tobytes(), dtype=np.float32)

    pts = np.stack([col("x"), col("y"), col("z")], axis=-1)
    return pts[np.isfinite(pts).all(axis=1)].astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Pose interpolation (slerp)
# --------------------------------------------------------------------------- #
def quat_to_rotmat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n <= 1e-24:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def slerp_xyzw(q0, q1, a):
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
    t0 = math.acos(dot)
    st0 = math.sin(t0)
    t = t0 * a
    return (math.cos(t) - dot * math.sin(t) / st0) * q0 + (math.sin(t) / st0) * q1


def interpolate_pose(times, positions, quats, t) -> np.ndarray:
    if not times:
        raise ValueError("no odometry samples")
    if len(times) == 1 or t <= times[0]:
        pos, quat = np.asarray(positions[0]), np.asarray(quats[0])
    elif t >= times[-1]:
        pos, quat = np.asarray(positions[-1]), np.asarray(quats[-1])
    else:
        i1 = bisect.bisect_left(times, t)
        i0 = max(0, i1 - 1)
        t0, t1 = times[i0], times[i1]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        pos = (1 - a) * np.asarray(positions[i0]) + a * np.asarray(positions[i1])
        quat = slerp_xyzw(np.asarray(quats[i0]), np.asarray(quats[i1]), a)
    T = np.eye(4)
    T[:3, :3] = quat_to_rotmat_xyzw(quat)
    T[:3, 3] = pos
    return T


def rotmat_to_quat_xyzw(R) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw, qx, qy, qz = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


# --------------------------------------------------------------------------- #
# sqlite3 rosbag streaming
# --------------------------------------------------------------------------- #
def resolve_db(bag_dir: Path) -> Path:
    dbs = sorted(Path(bag_dir).glob("*.db3"))
    if not dbs:
        raise FileNotFoundError(f"no *.db3 under {bag_dir} (mcap bags not supported by this tool)")
    return dbs[0]


def decode_image_bgr(msg) -> Optional[np.ndarray]:
    if msg.__class__.__name__ == "CompressedImage":
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    enc = (getattr(msg, "encoding", "") or "").lower()
    h, w = int(msg.height), int(msg.width)
    data = bytes(msg.data)
    if enc in ("bgr8", "rgb8"):
        step = int(msg.step) if int(msg.step) > 0 else w * 3
        arr = np.frombuffer(data, np.uint8, count=h * step).reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if enc == "rgb8" else arr.copy()
    if enc == "mono8":
        step = int(msg.step) if int(msg.step) > 0 else w
        arr = np.frombuffer(data, np.uint8, count=h * step).reshape(h, step)[:, :w]
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding {enc!r}")


def stamp_s(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag-dir", type=Path, required=True, help="rosbag2 dir containing *.db3 + metadata.yaml")
    ap.add_argument("--calibration", type=Path, required=True, help="Odin calib YAML (Tcl_0 + FishPoly cam_0)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--camera", default="odin1")
    ap.add_argument("--image-topic", default="/odin1/image/compressed")
    ap.add_argument("--cloud-topic", default="/odin1/cloud_slam", help="PointCloud2 in the world/odom frame")
    ap.add_argument("--odometry-topic", default="/odin1/odometry")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth image message")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scan-window-s", type=float, default=0.1)
    ap.add_argument("--max-scans-per-frame", type=int, default=3)
    ap.add_argument("--depth-max-m", type=float, default=60.0)
    ap.add_argument("--depth-dilate-px", type=int, default=2)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args(argv)

    calib = Calibration.from_yaml(args.calibration)
    K = calib.pinhole_K()
    map_x, map_y = make_fishpoly_rectification_maps(calib, K)
    T_base_cam = np.linalg.inv(calib.T_camera_base)
    W, H = calib.image_width, calib.image_height

    out = args.out_dir.expanduser().resolve()
    rgb_dir = out / "_rgb" / args.camera
    depth_dir = out / "depths" / args.camera
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    db = resolve_db(args.bag_dir.expanduser().resolve())
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    topics = {name: (tid, typ) for tid, name, typ in conn.execute("SELECT id, name, type FROM topics")}
    for t in (args.image_topic, args.cloud_topic, args.odometry_topic):
        if t not in topics:
            raise KeyError(f"topic {t!r} not in bag; available: {sorted(topics)}")
    id2name = {topics[t][0]: t for t in (args.image_topic, args.cloud_topic, args.odometry_topic)}
    msg_cls = {topics[t][0]: get_message(topics[t][1]) for t in id2name.values()}
    ids = tuple(id2name)

    times: List[float] = []
    positions: List[Tuple[float, float, float]] = []
    quats: List[Tuple[float, float, float, float]] = []
    cloud_buf: Deque[Tuple[float, np.ndarray]] = deque()
    pending: Deque[Tuple[object, float]] = deque()
    agg_cloud: List[np.ndarray] = []
    frames: List[dict] = []
    kept_img = 0
    last_t = 0.0

    def flush(force: bool):
        nonlocal frames
        while pending:
            msg, t = pending[0]
            if not force and (t + args.scan_window_s > last_t):
                break
            if not force and not (len(times) >= 2 and times[-1] >= t):
                break
            pending.popleft()
            if args.max_frames is not None and len(frames) >= args.max_frames:
                continue
            T_world_cam = interpolate_pose(times, positions, quats, t) @ T_base_cam
            bgr = decode_image_bgr(msg)
            if bgr is None:
                continue
            if bgr.shape[1] != W or bgr.shape[0] != H:
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
            bgr = cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            near = [(tt, pp) for tt, pp in cloud_buf if abs(tt - t) <= args.scan_window_s]
            if len(near) > args.max_scans_per_frame:
                s = max(1, len(near) // args.max_scans_per_frame)
                near = near[::s][: args.max_scans_per_frame]
            pts = (np.concatenate([pp for _t, pp in near], axis=0).astype(np.float64)
                   if near else np.zeros((0, 3)))
            pix, z = project_world_pinhole(pts, T_world_cam, K)
            depth = rasterize_zbuffer(pix, z, W, H, min_depth=0.1, max_depth=args.depth_max_m)
            depth = dilate_sparse_depth(depth, args.depth_dilate_px)
            if near:
                agg_cloud.append(pts.astype(np.float32))

            fid = f"{len(frames):06d}"
            cv2.imwrite(str(rgb_dir / f"{fid}.jpeg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            np.save(depth_dir / f"{fid}.npy", depth.astype(np.float32))
            frames.append({
                "frame_id": fid, "camera": args.camera,
                "rgb_path": f"_rgb/{args.camera}/{fid}.jpeg",
                "depth_path": f"depths/{args.camera}/{fid}.npy",
                "T_world_cam": T_world_cam.tolist(), "K": K.tolist(),
                "rgb_size": [H, W], "depth_size": [H, W],
                "timestamp_ns": int(round(t * 1e9)),
                "depth_valid_pixels": int(np.count_nonzero(np.isfinite(depth) & (depth > 0))),
            })
            if len(frames) % 50 == 0:
                print(f"[lidar2frames] {len(frames)} frames "
                      f"(last valid px={frames[-1]['depth_valid_pixels']}, scans={len(near)})", flush=True)
        # bound the cloud buffer
        cutoff = (pending[0][1] - args.scan_window_s) if pending else (last_t - args.scan_window_s)
        while cloud_buf and cloud_buf[0][0] < cutoff:
            cloud_buf.popleft()

    sql = f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({','.join('?' * len(ids))}) ORDER BY id"
    img_seen = 0
    for tid, _ts, data in conn.execute(sql, ids):
        name = id2name[tid]
        msg = deserialize_message(data, msg_cls[tid])
        if name == args.cloud_topic:
            t = stamp_s(msg.header)
            last_t = max(last_t, t)
            cloud_buf.append((t, pointcloud2_to_xyz(msg)))
        elif name == args.odometry_topic:
            t = stamp_s(msg.header)
            last_t = max(last_t, t)
            p, o = msg.pose.pose.position, msg.pose.pose.orientation
            times.append(t)
            positions.append((p.x, p.y, p.z))
            quats.append((o.x, o.y, o.z, o.w))
        else:  # image
            img_seen += 1
            if (img_seen - 1) % args.stride != 0:
                continue
            t = stamp_s(msg.header)
            last_t = max(last_t, t)
            pending.append((msg, t))
        flush(force=False)
        if args.max_frames is not None and len(frames) >= args.max_frames and not pending:
            break
    flush(force=True)

    cloud_xyz = (np.concatenate(agg_cloud, axis=0) if agg_cloud else np.zeros((0, 3), np.float32))
    if cloud_xyz.shape[0] > 400_000:
        cloud_xyz = cloud_xyz[np.random.default_rng(0).choice(cloud_xyz.shape[0], 400_000, replace=False)]
    np.savez_compressed(out / "cloud.npz", xyz=cloud_xyz.astype(np.float32))

    index = {
        "schema_version": 1, "dataset": "odin1", "scene_id": args.out_dir.name,
        "cameras": [args.camera], "cloud_path": "cloud.npz", "frames": frames,
        "notes": f"bag={db.parent.name};calib={args.calibration.name};lidar_projection",
    }
    (out / "frames.json").write_text(json.dumps(index, indent=1))
    med = int(np.median([f["depth_valid_pixels"] for f in frames])) if frames else 0
    print(f"[lidar2frames] wrote {len(frames)} frames to {out} (median valid depth px/frame={med})", flush=True)
    print(f"[lidar2frames] reconstruct:  python -m scene_graph.offline.run --source frames-json "
          f"--frames-json-dir {out} --save-path scene.pt --covisibility", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
