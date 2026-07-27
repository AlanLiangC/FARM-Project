"""Unified scorer: read canonical predictions JSON → write canonical metrics JSON.

This is the ONE scoring entry point for every method × benchmark in the final
benchmark sweep — the canonical, apples-to-apples pipeline. Converters such as
``scripts/convert_ours_to_canonical.py`` produce the canonical predictions
schema it consumes.

Usage::

    python scripts/eval_predictions.py \\
        --predictions /data/out/<method>/<bench>_preds.json \\
        --bench scannet | hm3d \\
        --metrics-out /data/out/<method>/<bench>_preds-metrics.json \\
        [--scans-dir /data/scans]      # required for ScanNet
        [--hm3d-root /data/iref_vla/HM3D]  # required for HM3D
        [--scene-state-dir /data/out/<method>/scene_states/<bench>]  # required if any candidate
                                                                     # has pred_mask_source=ours_state
        [--uid-filter benchmarks/curated_utterances/<bench>_30.json]
        [--view-picker v1_largest_mask]  # used only when a candidate
                                         # doesn't carry chosen_view_image_id

The scorer is intentionally NOT a thin wrapper around the per-benchmark
``score_predictions`` functions — those have grown method-specific logic and
incompatible metrics schemas. This script is the canonical, apples-to-apples
replacement.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("scene_graph.eval.eval_predictions")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True,
                   help="Canonical predictions JSON file.")
    p.add_argument("--bench", choices=("scannet", "hm3d"), required=True,
                   help="Benchmark — determines GT loader + frame source.")
    p.add_argument("--metrics-out", type=Path, required=True,
                   help="Output metrics JSON path. Per-record breakdown sibling at <stem>-per-record.json.")
    p.add_argument("--scans-dir", type=Path, default=None,
                   help="ScanNet scans dir (e.g. /data/scans). Required for --bench scannet.")
    p.add_argument("--hm3d-root", type=Path, default=None,
                   help="IRef-VLA HM3D root (with per-scene .semantic.glb). Required for --bench hm3d.")
    p.add_argument("--scene-state-dir", type=Path, default=None,
                   help="Dir with per-scene scene_state.pt files. Required when any candidate has pred_mask_source=ours_state.")
    p.add_argument("--uid-filter", type=Path, default=None,
                   help="Optional JSON file restricting which utterances to score.")
    p.add_argument("--view-picker", default="v1_largest_mask",
                   help="Default picker for candidates without chosen_view_image_id. Locked default (2026-05-16 ablation).")
    p.add_argument("--depth-tolerance-m", type=float, default=0.15,
                   help="Depth-tolerance for visible-point projection. Locked default 0.15 "
                        "(2026-05-16), matching the visible-mask protocol numbers. "
                        "Tighter values (0.08) drop too many "
                        "GT projections under realistic camera-pose noise.")
    p.add_argument("--point-radius-px", type=int, default=3)
    p.add_argument("--min-gt-pixels", type=int, default=20)
    p.add_argument("--max-points", type=int, default=50000)
    p.add_argument("--gt-object-margin-m", type=float, default=0.02,
                   help="HM3D-only: AABB margin when extracting GT surface from the scene mesh.")
    p.add_argument("--allow-raw-projection", action="store_true",
                   help="Score frames with no depth by raw projection (default requires depth).")
    p.add_argument("--top-k-cap", type=int, default=10,
                   help="Cap per-utterance ranked candidates scored. Locked default "
                        "is 10 (2026-05-16).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _collect_uids_from_filter_doc(blob: object) -> set[str]:
    """Extract uids from either legacy uid lists or curated subset JSON.

    The final benchmark scripts historically used simple ``["uid", ...]``
    files, while the controlled subset stores richer grouped scene records
    under ``scenes.*.utterances`` plus a flat list of utterance dictionaries.
    """
    uids: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            uid = node.get("uid") or node.get("utterance_id")
            if uid is not None:
                uids.add(str(uid))

            scenes = node.get("scenes")
            if isinstance(scenes, dict):
                for scene_doc in scenes.values():
                    visit(scene_doc)

            for key in ("utterances", "statements", "records", "predictions", "uids"):
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        visit(item)

            # Legacy shape: {"scene0011_00": ["uid1", "uid2"], ...}
            for value in node.values():
                if isinstance(value, list) and all(not isinstance(x, (dict, list)) for x in value):
                    for item in value:
                        uids.add(str(item))
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    visit(item)
                else:
                    uids.add(str(item))

    visit(blob)
    return uids


def _load_uid_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    return _collect_uids_from_filter_doc(blob)


def _scan_id_to_state_path(state_dir: Path, scan_id: str) -> Optional[Path]:
    direct = state_dir / f"{scan_id}.pt"
    if direct.exists():
        return direct
    nested = state_dir / scan_id / "scene_state.pt"
    if nested.exists():
        return nested
    return None


def _group_predictions(
    preds: Sequence[Mapping[str, Any]], uid_filter: Optional[set[str]]
) -> Dict[str, List[Mapping[str, Any]]]:
    by_scene: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for rec in preds:
        uid = str(rec.get("uid", ""))
        if uid_filter is not None and uid not in uid_filter:
            continue
        scan = str(rec.get("scan_id") or rec.get("scene_id") or "")
        if not scan:
            continue
        by_scene[scan].append(rec)
    return by_scene


def _load_gt_for_scene(
    bench: str,
    scan_id: str,
    *,
    scans_dir: Optional[Path],
    hm3d_root: Optional[Path],
    gt_object_margin_m: float,
):
    """Return ``(gt_lookup, gt_mask_provider, gt_points_lookup)``."""
    if bench == "scannet":
        from scene_graph.eval.referit3d.scannet_gt import load_scene_gt, load_scene_gt_points
        gt = load_scene_gt(scan_id, scans_dir=scans_dir)
        gt_points = load_scene_gt_points(scan_id, scans_dir=scans_dir)
        return gt, None, gt_points
    if bench == "hm3d":
        from scene_graph.eval.iref_vla.iref_vla_gt import load_scene_objects
        from scene_graph.eval.visible_mask import GTMeshMaskProvider, sample_aabb_surface_points
        gt = load_scene_objects(scan_id, dataset_root=hm3d_root)
        gt_mask_provider = None
        if hm3d_root is not None:
            try:
                gt_mask_provider = GTMeshMaskProvider.from_hm3d_root(
                    scan_id, Path(hm3d_root), object_margin_m=gt_object_margin_m
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("HM3D mesh provider unavailable for %s: %s — falling back to AABB-surface sample.", scan_id, exc)
        # AABB-surface fallback: precompute per-instance to keep loop fast.
        gt_points: Dict[int, np.ndarray] = {}
        if gt_mask_provider is None:
            for oid, obj in gt.items():
                gt_points[int(oid)] = sample_aabb_surface_points(
                    obj.bbox_min, obj.bbox_max, spacing=0.03, max_points=50000
                )
        return gt, gt_mask_provider, gt_points
    raise ValueError(f"unknown bench={bench!r}")


def _score_scene(
    bench: str,
    scan_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    scans_dir: Optional[Path],
    hm3d_root: Optional[Path],
    scene_state_dir: Optional[Path],
    view_picker: str,
    depth_tolerance_m: float,
    point_radius_px: int,
    min_gt_pixels: int,
    max_points: int,
    require_depth: bool,
    top_k_cap: int,
    gt_object_margin_m: float,
) -> List[Dict[str, Any]]:
    from scene_graph.eval.unified_scoring import score_one_candidate, make_frame_source
    from scene_graph.eval.view_selection import resolve_chosen_view_image_id
    from scene_graph.eval.visible_mask import SceneStateMaskIndex

    # Build mask_index from ours' scene_state — used both as ours' frame source
    # (``ours_state`` pred_mask_source) and as the resolver for ours' V1 picker
    # over mask sidecars. Methods like BBQ supply their own ``frame_source`` per
    # record; we instantiate those lazily and cache by (kind, dir).
    mask_index: Optional[SceneStateMaskIndex] = None
    if scene_state_dir is not None:
        state_path = _scan_id_to_state_path(Path(scene_state_dir), scan_id)
        if state_path is not None:
            try:
                mask_index = SceneStateMaskIndex.from_path(state_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("failed to build mask_index for %s: %s", scan_id, exc)

    frame_source_cache: Dict[Any, Any] = {}

    def _resolve_frame_source(rec: Mapping[str, Any]):
        fs = rec.get("frame_source") if isinstance(rec, dict) else None
        if not fs:
            # Default: ours' scene_state. Requires mask_index.
            if mask_index is None:
                return None
            key = ("ours_scene_state", None)
            cached = frame_source_cache.get(key)
            if cached is None:
                cached = make_frame_source(frame_source_kind="ours_scene_state", mask_index=mask_index)
                frame_source_cache[key] = cached
            return cached
        kind = str(fs.get("kind") or "")
        if kind == "ours_scene_state":
            if mask_index is None:
                return None
            key = ("ours_scene_state", None)
        elif kind == "bbq_extracted_frames":
            key = ("bbq_extracted_frames", str(fs.get("frames_dir")))
        elif kind == "scannet_sens_direct":
            key = ("scannet_sens_direct", scan_id, str(scans_dir))
        else:
            return None
        cached = frame_source_cache.get(key)
        if cached is None:
            cached = make_frame_source(
                frame_source_kind=kind,
                mask_index=mask_index if kind == "ours_scene_state" else None,
                frames_dir=Path(fs.get("frames_dir")) if fs.get("frames_dir") else None,
                scan_id=scan_id if kind == "scannet_sens_direct" else None,
                scans_dir=Path(scans_dir) if kind == "scannet_sens_direct" and scans_dir is not None else None,
            )
            frame_source_cache[key] = cached
        return cached

    gt, gt_mask_provider, gt_points_lookup = _load_gt_for_scene(
        bench, scan_id,
        scans_dir=scans_dir, hm3d_root=hm3d_root, gt_object_margin_m=gt_object_margin_m,
    )

    # --- Cache view -> NPZ-path so we can sort candidate-scoring work by
    # NPZ, giving each NPZ exactly one expensive decompress per scene
    # instead of the cache-thrashed ~2x in the input-order scorer.
    view_to_npz: Dict[int, str] = {}
    if mask_index is not None:
        try:
            images_by_id = dict(mask_index.frame_resolver._images_by_id)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            images_by_id = {}
        for iid, rec_im in images_by_id.items():
            try:
                sr = rec_im.get("source_ref") if isinstance(rec_im, Mapping) else getattr(rec_im, "source_ref", None)
            except Exception:  # noqa: BLE001
                sr = None
            sr = str(sr or "")
            if "#frame=" in sr:
                view_to_npz[int(iid)] = sr.split("#frame=")[0]

    # ------------------------------------------------------------------
    # Pre-pass 1: build the per-record scaffolding (gt, gt_points, frame_source,
    # ranked, per_cand placeholder list) and a flat list of candidate-scoring
    # tasks. Each task carries everything score_one_candidate needs PLUS the
    # NPZ-path key we'll sort by.
    # ------------------------------------------------------------------

    # Per-record skeletons (mutated in pre-pass 3); per-record metadata cache
    # (gt_instance / gt_points / frame_source); flat list of candidate-scoring
    # tasks (one per ranked candidate that survives the pre-pass).
    out_records: List[Optional[Dict[str, Any]]] = [None] * len(records)
    flat_tasks: List[Tuple[str, int, int]] = []  # (npz_key, orig_idx, cand_idx)
    record_meta: Dict[int, Dict[str, Any]] = {}
    task_chosen: List[Optional[int]] = []
    task_cand: List[Mapping[str, Any]] = []

    t_scene = time.time()
    for orig_idx, rec in enumerate(records):
        target_id = int(rec.get("target_id", -1))

        # Mirror original control flow: gt-presence check first.
        if target_id not in gt:
            out_records[orig_idx] = {
                "uid": rec.get("uid"),
                "scan_id": scan_id,
                "target_id": target_id,
                "top1_iou": 0.0,
                "per_candidate_ious": [],
                "top1_best_image_id": None,
                "error": "no_gt_for_target",
            }
            continue

        # Fast path A: empty-prediction with valid GT (byte-identical to the
        # original loop's empty-``ranked`` result).
        ranked_raw = rec.get("ranked")
        if not ranked_raw:
            out_records[orig_idx] = {
                "uid": rec.get("uid"),
                "scan_id": scan_id,
                "target_id": target_id,
                "top1_iou": 0.0,
                "top1_pred_pixels": 0,
                "top1_gt_pixels": 0,
                "top1_best_image_id": None,
                "per_candidate_ious": [],
                "per_candidate": [],
                "error": None,
            }
            continue

        gt_instance = gt[target_id] if bench == "hm3d" else None
        gt_points_for_target: Optional[np.ndarray] = None
        if bench == "scannet":
            gpt = gt_points_lookup.get(target_id) if gt_points_lookup else None
            if gpt is not None:
                gt_points_for_target = np.asarray(getattr(gpt, "points", gpt), dtype=np.float32).reshape(-1, 3)
        elif bench == "hm3d" and gt_mask_provider is None:
            gt_points_for_target = gt_points_lookup.get(target_id)

        frame_source = _resolve_frame_source(rec)

        ranked = list(ranked_raw)[: max(0, int(top_k_cap))]
        ious_slot: List[float] = [0.0] * len(ranked)
        per_cand_slot: List[Dict[str, Any]] = [{} for _ in ranked]
        skeleton: Dict[str, Any] = {
            "uid": rec.get("uid"),
            "scan_id": scan_id,
            "target_id": target_id,
            "top1_iou": 0.0,
            "top1_pred_pixels": 0,
            "top1_gt_pixels": 0,
            "top1_best_image_id": None,
            "per_candidate_ious": ious_slot,
            "per_candidate": per_cand_slot,
            "error": None,
        }
        out_records[orig_idx] = skeleton

        record_meta[orig_idx] = {
            "ranked": ranked,
            "gt_instance": gt_instance,
            "gt_points": gt_points_for_target,
            "frame_source": frame_source,
        }

        for cand_idx, cand in enumerate(ranked):
            chosen: Optional[int]
            if cand.get("chosen_view_image_id") is not None:
                try:
                    chosen = int(cand["chosen_view_image_id"])
                except (TypeError, ValueError):
                    chosen = None
            elif mask_index is not None:
                chosen_raw = resolve_chosen_view_image_id(mask_index, cand, picker_name=view_picker)
                chosen = None if chosen_raw is None else int(chosen_raw)
            else:
                chosen = None

            task_chosen.append(chosen)
            task_cand.append(cand)
            npz_key = view_to_npz.get(int(chosen), "") if chosen is not None else ""
            flat_tasks.append((npz_key, orig_idx, cand_idx))

    # ------------------------------------------------------------------
    # Pre-pass 2: sort tasks by NPZ path (stable on (orig_idx, cand_idx) for
    # determinism). All candidates that need a frame from the same NPZ now
    # cluster together -> the underlying frame-resolver loads each NPZ once.
    # ------------------------------------------------------------------
    flat_tasks_sorted = sorted(
        range(len(flat_tasks)),
        key=lambda i: (flat_tasks[i][0], flat_tasks[i][1], flat_tasks[i][2]),
    )

    # ------------------------------------------------------------------
    # Pre-pass 3: score every task in NPZ-locality order. Results are
    # mutated into the per-record per_cand / per_candidate_ious slots
    # we set up in pre-pass 1.
    # ------------------------------------------------------------------
    last_orig_idx: Optional[int] = None
    for task_id in flat_tasks_sorted:
        _npz_key, orig_idx, cand_idx = flat_tasks[task_id]
        meta = record_meta[orig_idx]
        cand = task_cand[task_id]
        chosen = task_chosen[task_id]
        frame_source = meta["frame_source"]

        # Drop visible-mask projection cache when we leave a record boundary
        # so per-record cache growth doesn't unbounded-creep across the
        # whole scene. This preserves the original loop's semantics
        # (clear_visible_mask_cache after each record) — the unified
        # scorer doesn't actually populate this cache, so it's a no-op
        # in practice, but kept for behaviour parity.
        if mask_index is not None and last_orig_idx is not None and orig_idx != last_orig_idx:
            mask_index.clear_visible_mask_cache()
        last_orig_idx = orig_idx

        if chosen is None:
            out_rec = out_records[orig_idx]
            assert out_rec is not None
            out_rec["per_candidate"][cand_idx] = {"error": "no_chosen_view"}
            out_rec["per_candidate_ious"][cand_idx] = 0.0
            continue

        if frame_source is None:
            out_rec = out_records[orig_idx]
            assert out_rec is not None
            out_rec["per_candidate"][cand_idx] = {"image_id": chosen, "error": "no_frame_source"}
            out_rec["per_candidate_ious"][cand_idx] = 0.0
            continue

        frame = frame_source.load(chosen)
        if frame is None:
            out_rec = out_records[orig_idx]
            assert out_rec is not None
            out_rec["per_candidate"][cand_idx] = {"image_id": chosen, "error": "frame_load_failed"}
            out_rec["per_candidate_ious"][cand_idx] = 0.0
            continue

        single = score_one_candidate(
            cand,
            mask_index=mask_index,
            frame=frame,
            gt_mask_provider=gt_mask_provider,
            gt_instance=meta["gt_instance"],
            gt_points=meta["gt_points"],
            depth_tolerance_m=depth_tolerance_m,
            point_radius_px=point_radius_px,
            min_gt_pixels=min_gt_pixels,
            max_points=max_points,
            require_depth=require_depth,
        )
        out_rec = out_records[orig_idx]
        assert out_rec is not None
        out_rec["per_candidate_ious"][cand_idx] = float(single.iou)
        out_rec["per_candidate"][cand_idx] = {
            "image_id": single.image_id,
            "iou": single.iou,
            "precision": single.precision,
            "recall": single.recall,
            "pred_pixels": single.pred_pixels,
            "gt_pixels": single.gt_pixels,
            "error": single.error,
        }

    # ------------------------------------------------------------------
    # Post-pass: fill the top1_* summary fields. These are derived from
    # per_candidate[0] / per_candidate_ious[0] (always rank-1, never
    # re-ordered), so the values are identical to what the original
    # loop produced.
    # ------------------------------------------------------------------
    for orig_idx, rec in enumerate(records):
        out_rec = out_records[orig_idx]
        if out_rec is None:
            continue
        per_cand = out_rec.get("per_candidate")
        ious = out_rec.get("per_candidate_ious")
        if not isinstance(per_cand, list) or not per_cand:
            # error / empty-pred branches already set top1_* keys upstream.
            continue
        out_rec["top1_iou"] = ious[0] if ious else 0.0
        first = per_cand[0]
        out_rec["top1_pred_pixels"] = int(first.get("pred_pixels")) if first.get("pred_pixels") else 0
        out_rec["top1_gt_pixels"] = int(first.get("gt_pixels")) if first.get("gt_pixels") else 0
        out_rec["top1_best_image_id"] = first.get("image_id")

    # Final clear after the very last task — mirrors the original loop's
    # post-record clear behavior on its final iteration.
    if mask_index is not None:
        mask_index.clear_visible_mask_cache()

    wall = time.time() - t_scene
    final: List[Dict[str, Any]] = [r for r in out_records if r is not None]
    LOGGER.info("scene %s n=%d wallclock=%.1fs", scan_id, len(final), wall)
    return final


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    if args.bench == "scannet" and args.scans_dir is None:
        LOGGER.error("--scans-dir is required for --bench scannet")
        return 2
    if args.bench == "hm3d" and args.hm3d_root is None:
        LOGGER.error("--hm3d-root is required for --bench hm3d")
        return 2

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(preds, list):
        LOGGER.error("predictions JSON must be a list of records; got %s", type(preds).__name__)
        return 2

    uid_filter = _load_uid_filter(args.uid_filter)
    by_scene = _group_predictions(preds, uid_filter)
    n_total = sum(len(v) for v in by_scene.values())
    LOGGER.info("scoring %d predictions across %d scenes (preds=%s, bench=%s, view_picker=%s)",
                n_total, len(by_scene), args.predictions, args.bench, args.view_picker)

    from scene_graph.eval.unified_scoring import aggregate_overall

    all_records: List[Dict[str, Any]] = []
    per_scene_metrics: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    for scan_id in sorted(by_scene):
        records = by_scene[scan_id]
        scored = _score_scene(
            args.bench, scan_id, records,
            scans_dir=args.scans_dir,
            hm3d_root=args.hm3d_root,
            scene_state_dir=args.scene_state_dir,
            view_picker=args.view_picker,
            depth_tolerance_m=args.depth_tolerance_m,
            point_radius_px=args.point_radius_px,
            min_gt_pixels=args.min_gt_pixels,
            max_points=args.max_points,
            require_depth=not args.allow_raw_projection,
            top_k_cap=args.top_k_cap,
            gt_object_margin_m=args.gt_object_margin_m,
        )
        all_records.extend(scored)
        per_scene_metrics[scan_id] = aggregate_overall(scored)

    overall = aggregate_overall(all_records)
    wall = time.time() - t0

    metrics_blob: Dict[str, Any] = {
        "overall": overall,
        "per_scene": per_scene_metrics,
        "n_scenes": len(by_scene),
        "predictions_path": str(args.predictions),
        "bench": args.bench,
        "visible_mask_params": {
            "view_picker": str(args.view_picker),
            "scene_state_dir": str(args.scene_state_dir) if args.scene_state_dir else None,
            "scans_dir": str(args.scans_dir) if args.scans_dir else None,
            "hm3d_root": str(args.hm3d_root) if args.hm3d_root else None,
            "depth_tolerance_m": float(args.depth_tolerance_m),
            "point_radius_px": int(args.point_radius_px),
            "min_gt_pixels": int(args.min_gt_pixels),
            "max_points": int(args.max_points),
            "require_depth": not args.allow_raw_projection,
            "top_k_cap": int(args.top_k_cap),
            "uid_filter": str(args.uid_filter) if args.uid_filter else None,
        },
        "wallclock_s": round(wall, 2),
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics_blob, indent=2))

    per_record_path = args.metrics_out.with_name(args.metrics_out.stem + "-per-record.json")
    per_record_path.write_text(json.dumps(all_records, indent=2))

    LOGGER.info("wrote metrics  : %s", args.metrics_out)
    LOGGER.info("wrote per-record: %s", per_record_path)

    headline = overall.get("acc@1@mask_iou=0.1")
    n = overall.get("n")
    LOGGER.info("OVERALL n=%s  acc@1@mask_iou=0.1=%.4f  acc@1@0.25=%.4f  acc@1@0.5=%.4f  mean_top1_iou=%.4f  wallclock=%.1fs",
                n, headline or 0.0,
                overall.get("acc@1@mask_iou=0.25") or 0.0,
                overall.get("acc@1@mask_iou=0.5") or 0.0,
                overall.get("mean_top1_iou") or 0.0,
                wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
