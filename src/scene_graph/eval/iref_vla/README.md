# IRef-VLA benchmark — HM3D split

Harness for benchmarking the pipeline on
[IRef-VLA](https://github.com/HaochenZ11/IRef-VLA) (ICRA 2025), a 3D
referential-grounding dataset built on top of the
[VLA-3D](https://github.com/HaochenZ11/VLA-3D) corpus. We use the
**HM3D split** (140 multi-room scenes) to test grounding in multi-room
indoor environments — IRef-VLA's region-level annotations let us pick
scenes with ≥2 regions deterministically.

See the repo-root `EVALUATION.md` for the full replication protocol and the
expected numbers.

## What lives here

| File              | Role                                                                 |
|-------------------|----------------------------------------------------------------------|
| `dataset.py`      | Load `<scene>_referential_statements.json` → `Statement`; multi-room filter |
| `iref_vla_gt.py`  | Parse `<scene>_object_result.csv` / `<scene>_region_result.csv` → `GTInstance` / `RegionInfo`; OBB → AABB |
| `runner.py`       | Iterate statements, call retriever, write predictions JSON          |
| `spatial_runner.py` | The paper's locked predict path (relational query graph per statement) |
| `metrics.py`      | Acc@1@IoU, R@K, MRR, median rank + breakdowns (AABB or visible-mask) |
| `scoring.py`      | CLI wrapper for `metrics.score_predictions`                          |

ReferIt3D's `matching.py` and `retrieval_adapter.py` are reused unchanged —
the bridge from retrieval clusters to flat `PredictedObject` AABB lists is
dataset-agnostic.

## Pipeline

```
HM3D mesh (.glb)
  ↓ scripts/render_hm3d_trajectory.py        (host, habitat-sim env)
NPZ frames (RGB + depth + intrinsics + pose)
  ↓ scripts/run_scene_graph_iref_vla.py      (offline driver, docker)
scene_state.pt
  ↓ scripts/eval_iref_vla.py --phase predict (retrieval, docker)
predictions.json
  ↓ scripts/convert_ours_to_canonical.py + scripts/eval_predictions.py --bench hm3d
metrics.json + per-breakdown table
```

## Step 1 — download IRef-VLA HM3D annotations

The IRef-VLA HM3D zip (~11 GB) is public (no auth), hosted on a CMU AirLab
swift bucket:

```bash
python -c "
import boto3
from botocore import UNSIGNED
from botocore.client import Config
c = boto3.client('s3',
    endpoint_url='https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330',
    config=Config(signature_version=UNSIGNED))
import tqdm
resp = c.get_object(Bucket='iref-vla', Key='HM3D.zip')
total = resp['ContentLength']
with open('HM3D.zip', 'wb') as fp, \
     tqdm.tqdm(total=total, unit='B', unit_scale=True) as bar:
    for chunk in resp['Body'].iter_chunks(chunk_size=1024*1024):
        fp.write(chunk); bar.update(len(chunk))
"
unzip HM3D.zip -d /path/to/iref_vla/
```

Scenes land at `/path/to/iref_vla/HM3D/<scene_id>/...`. Set `IREF_VLA_ROOT`
to that directory (container default: `/data/iref_vla/HM3D`).

## Step 2 — download HM3D meshes

IRef-VLA ships processed point clouds + annotations, not the raw HM3D meshes
needed for trajectory rendering. Grab them via the official habitat-sim
downloader (Matterport ToS-gated — request access first):

```bash
python -m habitat_sim.utils.datasets_download \
    --uids hm3d_val_v0.2 hm3d_val_semantic_annots_v0.2 \
    --data-path /path/to/hm3d
```

(Use `hm3d_train_v0.2` + `hm3d_train_semantic_annots_v0.2` for the full set.)

## Step 3 — render RGBD trajectories

Runs **outside** the docker container, in a habitat-sim (~0.2.5) environment.
For each scene, samples walkable points per region and renders
RGB+depth+pose:

```bash
python scripts/render_hm3d_trajectory.py \
    --scene-id 00238-j6fHrce9pHR \
    --hm3d-root /path/to/hm3d \
    --iref-vla-root /path/to/iref_vla/HM3D \
    --out /path/to/rendered/00238-j6fHrce9pHR \
    --mode magnet
```

Output is a directory of NPZ chunks the pipeline ingests via
`scene_graph.offline.run --source npz`.

## Step 4 — build scene graphs

Inside the container:

```bash
python scripts/run_scene_graph_iref_vla.py \
    --rendered-dir /data/iref_vla/rendered \
    --out-dir      /data/out/iref_vla \
    --skip-existing
```

Each scene becomes `/data/out/iref_vla/<scene_id>.pt`.

## Step 5 — predict and score

```bash
python scripts/eval_iref_vla.py --phase predict \
    --scenes-dir /data/out/iref_vla \
    --predictions-path /data/out/iref_vla/predictions.json
python scripts/convert_ours_to_canonical.py --in /data/out/iref_vla/predictions.json \
    --out /data/out/iref_vla/canonical.json --bench hm3d
python scripts/eval_predictions.py --predictions /data/out/iref_vla/canonical.json \
    --bench hm3d --metrics-out /data/out/iref_vla/metrics.json \
    --hm3d-root /data/iref_vla/HM3D --scene-state-dir /data/out/iref_vla
```

## Output schema

`predictions.json` (list of records):

```json
[
  {
    "uid": "00238-j6fHrce9pHR/0/3a1d7e8c/0",
    "scene_id": "00238-j6fHrce9pHR",
    "region_id": 0,
    "statement": "the picture that is above the coffee table",
    "target_id": 26, "target_class": "picture",
    "distractor_ids": [27, 28, 29, 30, 31, 53],
    "anchor_ids": [25], "anchor_classes": ["coffee table"],
    "relation": "above", "relation_type": "binary",
    "ranked": [
      {"rank": 1, "object_id": 26, "score": 0.41, "bbox_min": [], "bbox_max": [], "label": "picture", "caption": "..."}
    ],
    "elapsed_s": 1.2,
    "error": null
  }
]
```

`<predictions>-metrics.json` holds the aggregate dict (overall + breakdowns
by relation, region_class, difficulty, scene, target_class).

## Coordinate-system note (validated)

IRef-VLA stores object/region bboxes in HM3D's native **Z-up** frame.
habitat-sim runs in **Y-up** internally. Empirically validated on
`00009-vLpv2VX547B`: IRef-VLA `(x, y, z)` → habitat `(x, z, -y)`.

The transform is applied in two places:
- `iref_vla_gt.aabb_from_obb(..., to_habitat_frame=True)` (default) — converts
  GT bboxes into habitat frame so IoU vs scene-graph predictions just works.
- `scripts/render_hm3d_trajectory.py` region sampling — converts IRef-VLA
  region centers `(cx, cy)` → habitat floor-plane `(cx, -cy)` before calling
  `pathfinder.snap_point`.

Without both transforms, region-aware sampling lands on the wrong half of the
building and no GT targets fall inside the rendered scene-graph extent.
