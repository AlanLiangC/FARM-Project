#!/usr/bin/env python3
"""View a saved ``scene_state.pt`` in the browser (viser) — no re-mapping needed.

Loads a scene graph produced by ``python -m scene_graph.offline.run`` (or one
of the prebuilt graphs shipped with FARM-Scenes) and serves it interactively:
per-object voxel clouds and 3D boxes with captions, click-to-inspect, and the
**Query** panel, which runs the full relational retrieval pipeline
(``parse_query`` -> ``execute_spatial_query``) against the loaded scene.

Scene-state files use PyTorch pickle serialization. Only open files produced
by FARM or downloaded from a source you trust.

Examples (inside the container)::

    # Objects only
    python scripts/view_scene_state.py --pt /data/out/warehouse.pt

    # With the dataset's accumulated point cloud as background context
    python scripts/view_scene_state.py \
        --pt /data/scene_graphs/grandtour/2024-11-25_warehouse.pt \
        --cloud /data/scenes/grandtour/2024-11-25_warehouse/cloud.npz

Then open http://localhost:8080. For language queries, either click
"Start vLLM retrieval backend" in the Query panel (launches the servers on
this machine) or run ``./run.sh vllm`` first / point ``VLLM_BASE_URL`` +
``VLLM_EMBED_BASE_URL`` at running servers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def load_state(pt_path: Path) -> dict:
    """Load a scene_state payload, normalising through the library loader."""
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    feature_dim = None
    if isinstance(payload, dict):
        feature_dim = payload.get("feature_dim")
        if feature_dim is None and isinstance(payload.get("state"), dict):
            feats = payload["state"].get("features")
            if isinstance(feats, torch.Tensor) and feats.ndim == 2:
                feature_dim = int(feats.shape[1])
    if feature_dim is not None:
        from scene_graph.scene_state_io import load_scene_state

        return load_scene_state(pt_path, feature_dim=int(feature_dim), device="cpu")
    # Fallback: raw dict without the save_scene_state wrapper.
    return payload["state"] if isinstance(payload, dict) and "state" in payload else payload


def _record_field(rec: object, key: str) -> object:
    """Field access for image records, which are dicts in raw payloads and
    ``ImageRecord`` dataclasses after ``load_scene_state`` normalisation."""
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)


def remap_image_refs(state: dict, frames_dir: Path) -> int:
    """Re-point saved image references at *frames_dir* (a scene directory with
    ``rgb/<camera>/`` frames) so click-to-inspect can show each object's anchor
    view. Saved graphs reference the reconstruction machine's paths; the same
    frames ship with the dataset."""
    remapped = 0
    for rec in state.get("images") or []:
        ref = str(_record_field(rec, "source_ref") or _record_field(rec, "storage_path") or "")
        base = Path(ref).name
        if not base:
            continue
        camera = str(_record_field(rec, "camera_id") or "")
        candidates = [frames_dir / "rgb" / camera / base] if camera else []
        candidates += [frames_dir / "rgb" / base, frames_dir / base]
        for cand in candidates:
            if cand.is_file():
                if isinstance(rec, dict):
                    rec["source_ref"] = str(cand)
                else:
                    rec.source_ref = str(cand)
                remapped += 1
                break
    return remapped


def load_cloud(cloud_path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Read points (+ optional colors) from a ``cloud.npz``-style archive."""
    with np.load(cloud_path) as data:
        pts = None
        for key in ("points", "xyz", "cloud"):
            if key in data.files:
                pts = np.asarray(data[key], dtype=np.float32)
                break
        if pts is None:
            pts = np.asarray(data[data.files[0]], dtype=np.float32)
        pts = pts.reshape(-1, pts.shape[-1])[:, :3]
        cols = None
        for key in ("colors", "rgb", "color"):
            if key in data.files:
                cols = np.asarray(data[key], dtype=np.float32).reshape(-1, 3)
                break
    if max_points > 0 and pts.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[keep]
        if cols is not None and cols.shape[0] >= keep.max() + 1:
            cols = cols[keep]
    return pts, cols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve a saved scene_state.pt in viser (objects, captions, retrieval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pt", type=Path, required=True, help="Path to scene_state.pt")
    parser.add_argument("--cloud", type=Path, default=None, help="Optional cloud.npz shown as static background")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Scene directory with rgb/<camera>/ frames for click-to-inspect anchor views (default: the --cloud directory)")
    parser.add_argument("--host", default="127.0.0.1", help="Interface for the viser server")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viser server")
    parser.add_argument("--point-size", type=float, default=0.02, help="Background cloud point size (m)")
    parser.add_argument("--max-cloud-points", type=int, default=2_000_000, help="Random-subsample the background cloud to this many points (0 = keep all)")
    parser.add_argument("--voxel-points-per-object", type=int, default=0, help="Per-object voxel evidence points to render (0 = clean default: boxes + cloud + trajectory only)")
    parser.add_argument("--query-examples", type=Path, default=None, help="Text file with one query per line for the Query panel's Examples dropdown (default: derive examples from the scene's own captioned objects)")
    args = parser.parse_args()

    from scene_graph.visualization.viser_visualizer import PipelineViserVisualizer

    state = load_state(args.pt.expanduser())
    frames_dir = args.frames_dir
    if frames_dir is None and args.cloud is not None:
        frames_dir = args.cloud.expanduser().parent
    if frames_dir is not None:
        remapped = remap_image_refs(state, Path(frames_dir).expanduser())
        if remapped:
            print(f"Resolved {remapped} image references into {frames_dir}")
    means = state.get("means")
    n_objects = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
    captions = [c for c in (state.get("object_caption") or []) if isinstance(c, str) and c.strip()]
    print(f"Loaded {n_objects} objects ({len(captions)} captioned) from {args.pt}")

    query_examples = None
    if args.query_examples is not None:
        lines = args.query_examples.expanduser().read_text().splitlines()
        query_examples = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        print(f"Loaded {len(query_examples)} query examples from {args.query_examples}")

    visualizer = PipelineViserVisualizer(
        enabled=True,
        host=args.host,
        port=args.port,
        query_examples=query_examples,
        live_rgb_enabled=False,
        image_pose_axes_enabled=False,
        object_image_connections_enabled=False,
        covisibility_connections_enabled=False,
        regions_enabled=False,
        object_voxel_cloud_enabled=args.voxel_points_per_object > 0,
        object_voxel_max_points_per_object=max(0, args.voxel_points_per_object),
        object_box_from_voxels=True,
    )
    if not visualizer.enabled:
        print("viser is not available in this environment — aborting.")
        return 1

    frame_points = None
    if args.cloud is not None:
        try:
            points, colors = load_cloud(args.cloud.expanduser(), args.max_cloud_points)
            visualizer.add_background_point_cloud(points, colors, point_size=args.point_size)
            print(f"Background cloud: {points.shape[0]:,} points from {args.cloud}")
            frame_points = points
        except Exception as exc:  # noqa: BLE001 - cloud is optional context
            print(f"Skipping background cloud ({exc})")

    visualizer.update(colors=[], depths=[], intrinsics=[], poses=[], scene_state=state)
    poses = []
    for rec in state.get("images") or []:
        pose = _record_field(rec, "pose")
        if pose is None:
            continue
        try:
            arr = np.asarray(pose.cpu().numpy() if hasattr(pose, "cpu") else pose, dtype=np.float32)
        except Exception:
            continue
        if arr.shape == (4, 4) and np.isfinite(arr).all():
            poses.append(arr)
    if poses:
        try:
            visualizer.add_trajectory(np.stack(poses))
        except Exception as exc:  # noqa: BLE001 - trajectory is optional context
            print(f"Skipping trajectory ({exc})")
    if frame_points is None and isinstance(means, torch.Tensor) and means.numel():
        frame_points = means.detach().cpu().numpy().reshape(-1, 3)
    visualizer.set_home_view(frame_points)
    print(f"Serving on http://localhost:{args.port} — Ctrl+C to stop.")
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
