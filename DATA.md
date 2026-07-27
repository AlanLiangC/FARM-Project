# Datasets & data formats

This page explains **what data the pipeline consumes, how to format your own,
and how to run reconstruction + retrieval on each supported source** — with
download pointers for the public datasets we used.

> On every dataset below you can build a scene graph and query it. For the
> three paper benchmarks (FARM-Scenes, ReferIt3D, IRef-VLA) this repo also
> ships the full evaluation harness — see [`EVALUATION.md`](EVALUATION.md)
> for the end-to-end replication guide and expected numbers.

## The pipeline's input contract

Every source is adapted to a `FrameSource` (`src/scene_graph/offline/frame_sources/`)
that yields one dict per frame into the shared `StreamingMapper`. There are
exactly two accepted dict shapes (see `frame_sources/base.py`):

1. **ROS-native:** `{"camera": str, "rgbd_msg": RGBDFrame, "received_time": float}`
   — used by the rosbag source.
2. **Pre-decoded** (no ROS message): numpy arrays at native resolution —
   `{"camera", "rgb" (H,W,3 uint8), "depth_f32" (H,W float32, metres),
   "T_world_cam" (4×4), "rgb_instrinsics", "depth_instrinsics", "stamp_ns",
   "frame_id", "received_time"}` — used by all dataset readers.

So **anything you can express as `(rgb, metric-depth, intrinsics, camera pose)`
per frame can be fed in** — either by using one of the built-in sources below,
or by writing a ~30-line `FrameSource` subclass.

## Supported sources at a glance

| Source | `--source` | Input format | Sample data available? |
|---|---|---|---|
| **FARM-Scenes** | `frames-json` | `frames.json` index + JPEG + uint16-PNG depth | **Yes — [hf.co/datasets/GoldenGait/FARM-Scenes](https://huggingface.co/datasets/GoldenGait/FARM-Scenes), incl. prebuilt scene graphs** |
| ScanNet | `sens` | one `.sens` archive per scene | Yes — download from ScanNet (ToS) |
| NPZ chunks | `npz` | `.npz` files with `images/depths/camtoworlds/K` | **Yes — generate synthetically, no download (see below)** |
| frames.json | `frames-json` | a `frames.json` index + JPEG/`.npy`/`.png` per frame | Build from your own capture |
| rosbag2 / mcap | `rosbag` | `RGBDFrame` messages on `/mapping/rgbd_frame/<cam>` | Only bags recorded from the online pipeline |
| Replica (config) | `run_pipeline.py` | RGB + depth-PNG sequences | Yes — Replica-Dataset / Nice-SLAM renders |

Every command below assumes you are **inside the container** (`./run.sh shell
/path/to/data`) with models fetched (`./bootstrap_models.sh`). `/data` inside
the container is whatever host dir you passed to `./run.sh shell`.

---

## FARM-Scenes — the quickest real data

[FARM-Scenes](https://huggingface.co/datasets/GoldenGait/FARM-Scenes) is the
dataset released with the paper: seven large-scale real-robot scenes
(47,300 m² across an ANYmal, a Spot, and a handheld LiDAR-camera unit) with
posed RGBD frames, human-verified 3D object annotations, spatial referring
expressions, **and prebuilt scene graphs** — so you can try visualization and
retrieval before running any reconstruction.

```bash
# one scene + its prebuilt graph (~350 MB); drop --include to fetch all 7 scenes (~2 GB)
hf download GoldenGait/FARM-Scenes --repo-type dataset --include "scenes/grandtour/2024-11-25_warehouse/*" --include "scene_graphs/grandtour/2024-11-25_warehouse.pt" --local-dir /path/to/farm_scenes
```

Each scene is a ready-to-map `frames-json` directory (see the README
quickstart for the three copy-paste commands):

```bash
python -m scene_graph.offline.run --source frames-json --frames-json-dir /data/scenes/grandtour/2024-11-25_warehouse --save-path /data/out/warehouse.pt --covisibility
```

Depth is stored as uint16 PNG with a per-scene `depth_encoding` block in
`frames.json` (`depth_m = png * scale_to_metres`, 0 = invalid) because
outdoor LiDAR depth exceeds the 65.535 m reach of plain millimetres — the
`frames-json` source decodes this automatically.

---

## ScanNet (`.sens`)

The headline reconstruction source — one self-contained `.sens` archive per
scene, no pre-extraction.

**Download.** ScanNet requires accepting its Terms of Use. Request access at
<https://github.com/ScanNet/ScanNet>; you'll receive `download-scannet.py`.
Fetch the `.sens` for one scene:

```bash
python download-scannet.py -o /path/to/scannet --type .sens --id scene0000_00
```

**Run:**

```bash
python -m scene_graph.offline.run \
    --source sens \
    --sens-path /data/scans/scene0000_00/scene0000_00.sens \
    --stride 5 \
    --save-path /data/out/scene0000_00.pt \
    --covisibility
```

`--stride` subsamples frames (a full ~5.5k-frame scene at stride 5 finishes in
about a minute on a modern GPU). Output is a `scene_state.pt`.

---

## NPZ chunks — the easiest bring-your-own-data path

A sequence is stored as one or more `.npz` archives. Each archive holds:

| Key | Shape | Notes |
|---|---|---|
| `images` | `(N, H, W, 3)` uint8 | RGB |
| `depths` | `(N, H, W)` float32 | **metric depth in metres**, NaN/0 = invalid |
| `camtoworlds` | `(N, 4, 4)` or `(N, 3, 4)` float32 | camera-to-world pose |
| `K` | `(3, 3)` float32 | pinhole intrinsics |
| `pose_convention` | scalar (optional) | `'opengl'` (default) or `'opencv'` |
| `nominal_hz` | scalar (optional) | default `30.0` |

OpenGL→OpenCV pose conversion is handled for you. This is the format our
Habitat-sim HM3D renders used, but nothing about it is Habitat-specific.

**Self-contained smoke test (no external data):** write a tiny synthetic
sequence and reconstruct it — proves the whole install works end to end
without downloading anything:

```python
# make_sample_npz.py — run inside the container
import numpy as np, os
os.makedirs("/data/out/sample_npz", exist_ok=True)
N, H, W = 30, 240, 320
rng = np.random.default_rng(0)
images = rng.integers(0, 255, (N, H, W, 3), dtype=np.uint8)
depths = np.full((N, H, W), 2.0, np.float32)          # flat wall at 2 m
K = np.array([[160,0,W/2],[0,160,H/2],[0,0,1]], np.float32)
cams = np.stack([np.eye(4, dtype=np.float32) for _ in range(N)])
cams[:, 0, 3] = np.linspace(0, 0.5, N)                 # small pan in +x
np.savez("/data/out/sample_npz/frames_000.npz",
         images=images, depths=depths, camtoworlds=cams, K=K,
         pose_convention="opencv")
print("wrote /data/out/sample_npz/frames_000.npz")
```

```bash
python make_sample_npz.py
python -m scene_graph.offline.run \
    --source npz \
    --npz-dir /data/out/sample_npz \
    --camera sample \
    --save-path /data/out/sample.pt
```

(Random RGB yields few/no real detections — this checks the *plumbing*, not
retrieval quality. Swap in real frames for a meaningful graph.)

---

## frames.json

For captures where one big NPZ is inconvenient — a directory with a
`frames.json` index plus per-frame image/depth files.

`frames.json` schema:

```jsonc
{
  "cameras": ["cam0"],
  "frames": [
    {
      "camera": "cam0",
      "frame_id": "000000",
      "rgb_path": "rgb/000000.jpg",     // relative to the scene dir
      "depth_path": "depth/000000.npy", // float32 metres HxW, or .png (uint16; see depth_encoding)
      "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],   // or give fx/fy/cx/cy/width/height
      "T_world_cam": [[...4x4...]],          // camera-to-world
      "timestamp_ns": 0                       // optional
    }
  ]
}
```

Depth may instead be uint16 PNG: add a top-level
`"depth_encoding": {"format": "png16", "scale_to_metres": 0.001, "invalid_value": 0}`
block to `frames.json` (absent block = 1 mm per count).

```bash
python -m scene_graph.offline.run \
    --source frames-json \
    --frames-json-dir /data/my_capture \
    --save-path /data/out/my_capture.pt
```

---

## rosbag2 / mcap

Which path you use depends on **what the bag contains**.

### A) A depth-capable sensor bag (RGB + depth image + camera_info + TF) → online ROS pipeline

This is how the robot experiments were actually run. A rig like Spot publishes,
per camera, an RGB image, an **aligned depth image**, `camera_info`, and `/tf`
(e.g. `/spot/camera/frontleft/image` + `/spot/depth/frontleft/image` + …). The
online ROS graph consumes these **directly** — `frame_pub` synchronizes each
RGB+depth+TF triple into an `RGBDFrame` and `streaming_mapper` builds the graph:

```bash
# terminal 1 — bring up the 5-camera mapping graph (use_sim_time for bag /clock)
ros2 launch mapping mapping_five_cam.launch.py use_sim_time:=true caption_enabled:=false
# terminal 2 — replay the sensor bag
ros2 bag play /data/bags/spot_run --clock --rate 0.5
```

The camera-name → topic wiring lives in `src/scene_graph/camera_config.py`
(`head_left → /spot/camera/frontleft/*` + `/spot/depth/frontleft/*`, etc.);
edit it for a different rig's topic names. `./run.sh ros2` wraps this (adds the
vLLM servers for online captions).

### B) A bag of pre-fused `RGBDFrame` messages → offline `--source rosbag`

If the bag was recorded **from** the online pipeline (i.e. already contains
`RGBDFrame` on `/mapping/rgbd_frame/<camera>`), replay it offline without ROS
middleware:

```bash
python -m scene_graph.offline.run \
    --source rosbag --bag-path /data/bags/office_run \
    --save-path /data/out/office.pt --covisibility
```

### C) A pure-LiDAR bag (RGB + `PointCloud2`, no depth image) → project first

A LiDAR rig (e.g. Berkeley Odin1) records RGB + a LiDAR `PointCloud2` +
odometry but **no depth image**, so neither (A) nor (B) applies as-is. Convert
it with `scripts/lidar_bag_to_frames.py` (next section), then reconstruct via
`--source frames-json`.

### Raw LiDAR + camera bag → `frames.json` (`scripts/lidar_bag_to_frames.py`)

For a rig that records **RGB + a LiDAR `PointCloud2` (in the world/`odom`
frame) + odometry** but no depth image (e.g. Berkeley Odin1), this tool
synthesizes the missing depth: per RGB frame it interpolates the pose, gathers
nearby LiDAR scans, projects them into the camera (FishPoly fisheye → pinhole),
and writes a `frames.json` scene you then reconstruct normally.

```bash
# 1. project the raw bag into a frames.json scene
python scripts/lidar_bag_to_frames.py \
    --bag-dir /data/odin1/rosbag2_2026_04_13-17_30_31 \
    --calibration /data/odin1/calib.yaml \
    --out-dir /data/odin1/sdh4and7_frames \
    --stride 40                       # keep every Nth image (subsample the trajectory)

# 2. reconstruct from it
python -m scene_graph.offline.run --source frames-json \
    --frames-json-dir /data/odin1/sdh4and7_frames \
    --save-path /data/out/sdh4and7.pt --covisibility
```

The calibration YAML must provide `Tcl_0` (camera-from-LiDAR 4×4 extrinsic) and
a `cam_0` FishPoly block (`A11/A22/u0/v0`, `k2..k7`). Defaults target Odin1
topics (`/odin1/image/compressed`, `/odin1/cloud_slam`, `/odin1/odometry`) —
override `--image-topic` / `--cloud-topic` / `--odometry-topic` for other rigs.
sqlite3 (`.db3`) bags only. Depth is LiDAR-sparse; `--depth-dilate-px` (default
2) fills small gaps.

### ROS 2 online mode

The same `StreamingMapper` also runs live as a ROS 2 node
(`ros2 run mapping streaming_mapper`, or the launch files under
`ros/mapping/launch/`). It subscribes to `RGBDFrame` messages on
`/mapping/rgbd_frame/<camera>`; a `frame_pub` node produces those by
synchronizing an **RGB image topic + an aligned depth image topic + TF**. So
online mode expects a depth-capable rig (or an upstream node that emits aligned
depth). See the README's "Run online" section.

---

## Replica (config-driven)

Replica runs through the dataset-based orchestrator rather than
`offline.run`, driven by a YAML config (`configs/replica.yaml`).

**Download.** The pre-rendered RGB-D trajectories used by most SLAM papers:

```bash
# Nice-SLAM's Replica renders (RGB + depth PNG per frame)
git clone https://github.com/cvg/nice-slam && cd nice-slam
bash scripts/download_replica.sh     # → Replica/office0, room0, ...
```

(Raw meshes: <https://github.com/facebookresearch/Replica-Dataset>.)

**Run:**

```bash
python scripts/run_pipeline.py \
    --config configs/replica.yaml \
    --dataset-root /data/Replica \
    --sequence office0 \
    [--caption] [--viser]
```

Camera intrinsics + depth scale come from the `dataset:` block in
`configs/replica.yaml` — copy it and adjust for other RGB-D-PNG datasets.

---

## Querying what you reconstructed

Any `scene_state.pt` above is queryable (needs an embedding server —
`./run.sh vllm`):

```bash
python scripts/query_scene_graph.py --pt /data/out/scene0000_00.pt --query "a backpack"
```

See the README's "Query a scene graph" section for the Python API.

---

## Datasets used in the paper

We evaluated on the datasets below. This repo lets you **reconstruct and query**
each; the **grounding/QA scoring harnesses are not included here** (they live in
our research repo). Download pointers:

| Benchmark | Underlying data | Reconstruct via | Get the data |
|---|---|---|---|
| **FARM-Scenes** | 3 robot platforms, 7 scenes | `--source frames-json` | [hf.co/datasets/GoldenGait/FARM-Scenes](https://huggingface.co/datasets/GoldenGait/FARM-Scenes) (annotations + referring expressions + prebuilt graphs included) |
| **ReferIt3D** (NR3D + SR3D) | ScanNet | `--source sens` | ScanNet (ToS, above); language: <https://referit3d.github.io> |
| **IRef-VLA** | HM3D (multi-room) | render to NPZ, `--source npz` | Annotations: public AirLab bucket; HM3D meshes: Matterport (ToS) |
| **OpenEQA** | ScanNet, HM3D | `--source sens` / `npz` | <https://github.com/facebookresearch/open-eqa> |

For ScanNet-based benchmarks, reconstruct each scene once into
`/data/out/scannet/<scene_id>.pt` and reuse the `.pt` across queries.
