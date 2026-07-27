# ReferIt3D benchmark (ScanNet)

Harness for benchmarking the pipeline on
[ReferIt3D](https://referit3d.github.io/) — NR3D (natural human references) +
SR3D+ (programmatic spatial references), both grounded in ScanNet — in the
open-vocabulary detection setting (no GT boxes at prediction time).

See the repo-root `EVALUATION.md` for the full replication protocol and the
expected numbers.

## What lives here

| File                  | Role                                                               |
|-----------------------|--------------------------------------------------------------------|
| `dataset.py`          | Load NR3D/SR3D+ CSVs → `Utterance`; val-split + local-scans filter |
| `scannet_gt.py`       | Parse ScanNet aggregation/segs/PLY → per-instance AABB; NPZ cache  |
| `matching.py`         | Gaussian/voxel → AABB; 3D IoU; predicted ↔ GT match                |
| `retrieval_adapter.py`| Retrieval clusters → flat ranked `PredictedObject` list            |
| `runner.py`           | Non-spatial predict loop (embedding retrieval only)                |
| `metrics.py`          | Acc@1@IoU, R@K, MRR + breakdowns (AABB or visible-mask)            |
| `scoring.py`          | CLI wrapper for scoring                                            |
| `alias_geometry.py`   | Alias-box expansion used by `--geometry-mode alias_expand`         |
| `partial_scenes.txt`  | Frozen partial-reconstruction scene subset                          |

The paper's locked predict path (relational query graph per utterance) lives
in `scripts/eval_referit3d_spatial.py`, which builds on these modules.

## Data / env vars

- `REFERIT3D_DIR` — directory holding `nr3d.csv` and `sr3d+.csv`
  (download from <https://referit3d.github.io/>; container default
  `/data/_eval/referit3d`)
- `SCANNET_SCANS_DIR` — ScanNet `scans/` directory with `.sens`,
  `.aggregation.json`, `_vh_clean_2.0.010000.segs.json`, `_vh_clean_2.ply`
  per scene (ScanNet ToS; container default `/data/scans`)
- `SCANNETV2_VAL_TXT` — path to `scannetv2_val.txt` from the ScanNet
  benchmark repo (container default `/data/_eval/scannet_v2_val.txt`)
