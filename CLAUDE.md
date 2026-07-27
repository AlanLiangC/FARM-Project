# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Official implementation of **FARM: Find Anything using Relational Spatial Memory** (arXiv:2606.15476) — real-time 3D scene-graph construction and language-based retrieval from RGBD streams. Objects are detected/segmented (YOLOE, open-vocabulary), tracked as 3D Gaussians with sparse voxel evidence, merged across frames (DINOv3 features + Hellinger distance + union-find), captioned asynchronously (vLLM), embedded multi-modally, and queried with relational language (LLM query parsing → spatial predicate execution).

Online (ROS 2) and offline (datasets) share the **same algorithm code** — `StreamingMapper` runs in both; only the frame ingress differs. The layout separates the ROS-free `src/scene_graph/` library from the thin `ros/` integration layer. The Python package imports as `scene_graph`.

## Build & Run Commands

All runtime work happens inside the Docker container; there is no supported host-side pip/uv/conda path. See `README.md` for full setup; summary:

```bash
git submodule update --init --recursive   # (host, once) third_party/yoloe
./bootstrap_models.sh                      # (host, once) YOLOE + MobileCLIP + SigLIP2 from public URLs
./run.sh build                             # CUDA 12.8 + ROS 2 Humble image
./run.sh shell /path/to/data               # container shell; mounts the dir at /data
```

The entrypoint sources ROS, activates `~/.venv`, and colcon-builds `mapping_msgs` + `mapping` against the bind-mounted tree.

### Offline mapping

```bash
python -m scene_graph.offline.run --source sens --sens-path /data/scans/scene0000_00/scene0000_00.sens --stride 5 --save-path /data/out/scene0000_00.pt --covisibility
python -m scene_graph.offline.run --source frames-json --frames-json-dir /data/scenes/grandtour/2024-11-25_warehouse --save-path /data/out/warehouse.pt --covisibility
```

Sources: `sens`, `rosbag`, `npz`, `frames-json` (FARM-Scenes format — uint16-PNG depth decoded via the scene's `depth_encoding` block). Add `--caption` (needs `./run.sh vllm`), `--viser` (live 3D view :8080, ~6× throughput cost), `--regions`, `--extra-param key:=value` passthrough.

For Replica-style PNG-depth datasets driven by a typed `PipelineConfig` YAML: `python scripts/run_pipeline.py --config configs/replica.yaml`.

### View + query a saved scene state

```bash
python scripts/view_scene_state.py --pt /data/out/warehouse.pt [--cloud /data/scenes/.../cloud.npz]   # viser :8080, incl. Query panel
python scripts/query_scene_graph.py --pt /data/out/warehouse.pt --query "a backpack"                  # CLI retrieval (needs embed server)
```

Python API: `SceneGraphRetriever.from_scene_state(pt, embedder=EmbedInterface())` for embedding retrieval; `parse_query` + `execute_spatial_query` (`scene_graph.retrieval.spatial_reasoning`) for the full relational pipeline the paper evaluates.

### Online ROS 2

```bash
./run.sh ros2 caption_enabled:=true    # vLLM servers + scenegraph_validation_exploration.launch.py
```

One `frame_pub` per camera (topics from `CAMERA_CONFIG` in `src/scene_graph/camera_config.py`) feeds the single `streaming_mapper`, which batches by `camera_names` / `expected_batch`. `mapping_five_cam.launch.py` does this for the 5 Spot cameras. Depth-less LiDAR sensors work the same way: `odin1_depth_pub` projects the `PointCloud2` into the camera and republishes RGBD topics (`ros2 launch mapping mapping_odin1.launch.py`).

Registered executables (`ros/mapping/setup.py`): `streaming_mapper`, `frame_pub`, `odin1_depth_pub`, `visual_search_yoloe_text_prompt`. `nodes/tf_listener.py` is a helper import of `frame_pub`, not an executable.

### run.sh / in_docker.sh

- `./run.sh build | shell [dir] | vllm | ros2 [args...] | stop`
- `./run.sh vllm` starts three servers in tmux: **Qwen3.5-9B** :8000 (captioning + query parsing), **Qwen3-Embedding-0.6B** :8002, **Qwen3-VL-Embedding-2B** :8006. GPUs via `GPU_VL8`/`GPU_EMB`/`GPU_VL_EMB`.
- `./scripts/in_docker.sh <cmd>` runs a command in a long-lived container (default name `scene-graph-batch`, override `SG_CONTAINER`); rewrites `python` to the venv python and forwards the vLLM/SigLIP2 env vars.

### Models

`models/dinov3-vits16` (DINOv3 ViT-S/16 merge backbone) is **committed in-repo** with its license, so a fresh clone maps fully offline. `bootstrap_models.sh` fetches the rest from official public sources — YOLOE (`hf.co/jameslahm/yoloe`), MobileCLIP (Apple CDN), SigLIP2 (`scripts/download_siglip2.py`). No Git LFS, no HF account. The gated `dinov3-vits16plus` (paper backbone, tighter merging) is auto-preferred by `resolve_dino_backbone()` whenever `models/dinov3-vits16plus/` exists — see README. Both variants are `hidden_size=384`, so switching changes only merge-feature quality. **Each model has its own license — `THIRD_PARTY_NOTICES.md`.**

## Architecture

- **`src/scene_graph/`** — core library, zero ROS imports.
  - `offline/` — driver (`run.py`) + frame sources (`frame_sources/{sens,rosbag,npz,frames_json}.py`), `viser_recording.py`.
  - `pipeline/` — `PipelineOrchestrator` + pure-function `steps.py` shared with the ROS node.
  - `map_update/` — `object_update.py` (Gaussian fusion + voxel clouds), `covisibility.py` (bitset graph), `union_find.py` (correspondence), `get_neighbors.py` (spatial+Hellinger lookup), `filtering.py`, `pruning.py`, `cannot_link.py`, `mask_observations.py`, `models.py` (`SceneState`).
  - `segmentation/` — `yoloe.py` (the live backend), `dino.py` (merge features), `interfaces.py`, `models.py`, `visualization.py`.
  - `captioning/` — async vLLM captions: `services.py` (`CaptionManager`), `worker.py`, `structured.py`, `crop_util.py`, `models.py`.
  - `retrieval/` — `scene_graph_retriever.py` (`SceneGraphRetriever.from_scene_state(...).retrieve(query)`) + `spatial_reasoning/` (the paper's relational pipeline: `query_parser.py` → `executor.py` `execute_spatial_query`; `predicates.py`, `methods.py`, `semantic_retrieval.py`, `calibration.py`, `fuzzy_ops.py`, `prompts.py`, `models.py`). Covered by `tests/`.
  - `regions/` — multi-room region clustering + LLM labeling (used via `--regions` / `region_enabled`).
  - `storage/` — HDF5 image store + async writer. `datasets/` — Replica/ScanNet/NPZ loaders for the config-driven path. `visualization/` — `viser_visualizer.py` (`PipelineViserVisualizer`, includes the Query retrieval panel). `debug/` — JSONL pipeline tracer (pairs with `scripts/inspect_pipeline_trace.py`). `llm_utils/` — `LLMInterface`/`EmbedInterface` vLLM clients.
  - `config.py` (`PipelineConfig` from YAML), `runtime_paths.py` (model/file resolution, `resolve_dino_backbone()`), `scene_state_io.py` (save/load + image-ref resolution), `camera_config.py`, `scene_graph.py`, `mapping_util.py`.
- **`ros/`** — NOT an ament package itself; colcon recurses into it.
  - `ros/mapping/` — ament-python pkg `mapping`: `nodes/` (`streaming_mapper.py` — both online and offline funnel through `_run_mapping_batch`/`_decode_batch`; `frame_pub.py`, `odin1_depth_pub.py`, `visual_search_yoloe_text_prompt.py`, `tf_listener.py`), `lib/` (pure helpers: frame_sync, persistence, parameters, covisibility, filtering, embedding_io, batch_processing, bbox_utils, geometric_transforms, image_decoding, captioning, scene_graph_io, viser_edit, viser_integration, odin1_projection, paths), `launch/` (`mapping_five_cam`, `mapping_odin1`, `scenegraph_validation_exploration`, `visual_search_yoloe_text_prompt`).
  - `ros/msgs/` — ament-cmake pkg `mapping_msgs` (RGBDFrame, FrameMetadata, DetectedObject(s), LocalCaption(Array), SceneGraphSnapshotSimple, VisualSearchResult(Array); srv Siglip2TextEmbed).
- **`scripts/`** — `view_scene_state.py`, `query_scene_graph.py`, `run_pipeline.py`, `lidar_bag_to_frames.py` (RGB+LiDAR bag → frames.json), `inspect_pipeline_trace.py`, `download_siglip2.py`, `in_docker.sh`, plus the benchmark drivers listed under "Benchmark harness" in Notes.
- **`configs/`** — `replica.yaml`, `indoor.yaml`, `yoloe_vocabulary.txt` (YOLOE open-vocab prompt list).
- **`models/`**, **`docker/`**, **`tests/`**, **`third_party/yoloe`** (submodule).

### Shared pipeline (per batch)

1. `_decode_batch` — `RGBDFrame` msgs decoded; pre-decoded dicts from offline sources pass through (`rgbd_msg is None`).
2. `_prepare_frames` — tensors (colors, depths resized to RGB size, intrinsics, world poses).
3. `_segment_batch` — YOLOE → DINOv3 features → 3D Gaussian per mask (+ voxelized points).
4. `_filter_segmentation_outputs` — border / num-pixels / distance / label / IoU-duplicate filters.
5. `get_neighbors` + `find_object_correspondence` — union-find matching.
6. `update_scene_graph_state` — Gaussian fusion + voxel merge + covisibility merge on losers.
7. `update_covisibility_from_visible_indices` — adjacency by camera neighborhood.
8. Async captions (optional) · periodic pruning · snapshot/save.

### Benchmark evaluation

See `EVALUATION.md` for the full protocol. Quick FARM-Scenes run (dataset mounted at `/data`, vLLM servers up):

```bash
python scripts/eval_farm_scenes.py --dataset odin1 --phase both --eval-root /data/gt --scenes-dir /data/scene_graphs/odin1 --predictions /data/out/farm_odin1_preds.json
```

The score phase needs no GPU services; `--phase score` re-scores existing predictions deterministically.

### Offline frame-source contract

`FrameSource.__iter__()` yields either `{"camera", "rgbd_msg", "received_time"}` (ROS messages) or pre-decoded `{"camera", "rgb", "depth_f32", "T_world_cam", "rgb_instrinsics", "depth_instrinsics", "stamp_ns", "frame_id", "received_time"}` numpy dicts (note the historical `instrinsics` spelling).

### Key environment variables

`SCENE_GRAPH_MODEL_DIR` (default `./models`), `SCENE_GRAPH_MAPPING_DATA_DIR` (default `~/.ros/scene_graph/mapping`), `VLLM_BASE_URL` / `VLLM_EMBED_BASE_URL` / `VLLM_QWEN3_VL_EMBED_BASE_URL` (retrieval + caption backends), `LAM_SIGLIP2_CKPT`, `QWEN3_VL_EMBED_ENABLED`.

## Testing

`tests/` covers the spatial-reasoning retrieval path: `test_spatial_predicates.py`, `test_spatial_covisibility_constraint.py`. Run `pytest tests/` inside the container (`./scripts/in_docker.sh python -m pytest tests/`); CI runs them CPU-only. Elsewhere coverage is not comprehensive — verify changes with a short offline run (synthetic NPZ smoke in `DATA.md`, or a FARM-Scenes scene with `--end 50`).

## Notes

- **Voxel-cloud AABB.** `update_scene_graph_state` accumulates a per-object sparse voxel cloud (`object_voxel_keys_flat/offsets/levels`); `scene_state_io` persists it; `utils/geometry.py` decodes it (`voxel_cloud_aabb`, `voxel_keys_to_world`) into tight boxes used by the viser viewer and retrieval.
- **Two Python installs coexist.** Root `pyproject.toml` installs the `scene_graph` library into the venv; `ros/mapping/setup.py` is colcon-built (`mapping.*` namespace). Do NOT put a `package.xml` at `ros/` root — colcon would stop recursing and miss both packages.
- **`third_party/yoloe`** is a submodule (public HTTPS URL, pinned); the Dockerfile installs it editably with its nested `ml-mobileclip`. A clone without `--recursive` breaks `./run.sh build`. YOLOE is AGPL-3.0 — hence this repo's license.
- **streaming_mapper.py uses CRLF line endings** — keep them (git and editors may otherwise rewrite the whole file).
- **Benchmark harness.** `src/scene_graph/eval/` ships the full evaluation stack: `referit3d/` (NR3D/SR3D+ loaders, ScanNet GT, matching, metrics), `iref_vla/` (HM3D statements/GT + the locked `spatial_runner`), `largescale/` (FARM-Scenes GT loader + annotation schema), `unified_scoring.py` + `visible_mask.py` + `view_selection.py` (canonical scorer with occlusion-aware projected-mask IoU). Drivers: `scripts/eval_farm_scenes.py` (FARM-Scenes, works out of the box with the HF dataset), `scripts/{run_scene_graph_referit3d,eval_referit3d_spatial}.py` (ScanNet), `scripts/{render_hm3d_trajectory,run_scene_graph_iref_vla,eval_iref_vla}.py` (HM3D; render runs on the host in a habitat-sim env), `scripts/{convert_ours_to_canonical,eval_predictions,score_largescale_predictions}.py` (scoring). Curated subset uid lists in `benchmarks/curated_utterances/`. **EVALUATION.md** is the replication guide (per-benchmark commands + expected numbers + research-code parity record).
