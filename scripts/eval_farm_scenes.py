#!/usr/bin/env python3
"""Run the paper's grounding benchmark on FARM-Scenes.

Thin driver around ``scripts/eval_referit3d_spatial.py``'s predict loop for
the FARM-Scenes release layout (hf.co/datasets/GoldenGait/FARM-Scenes):

  --eval-root   <farm_scenes>/gt            (utterances.json + _gt_instances/)
  --scenes-dir  <farm_scenes>/scene_graphs/<dataset>   (prebuilt .pt files,
                or a directory of your own reconstructions)

By default it keeps the paper's locked retrieval knobs:

  * spatial_method=unified_soft_w50
  * no VLM candidate rerank / no spatial VLM checks
  * pre_filter_k=-1
  * retrieval_mode=multi
  * candidate_pool_mode=active
  * geometry_mode=alias_expand
  * max_predictions=100

Scoring is delegated to ``scripts/score_largescale_predictions.py`` because
FARM-Scenes GT is 3D-AABB-based rather than ScanNet/HM3D visible-mask GT.

Example (inside the container, vLLM servers up)::

    python scripts/eval_farm_scenes.py --dataset odin1 --phase both \
        --eval-root /data/gt --scenes-dir /data/scene_graphs/odin1 \
        --predictions /data/out/farm_odin1_preds.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_graph.eval.referit3d.dataset import Utterance  # noqa: E402

from eval_referit3d_spatial import run_spatial_predictions  # noqa: E402
from score_largescale_predictions import main as score_largescale_main  # noqa: E402


LOGGER = logging.getLogger("eval_farm_scenes")


def _to_utterance(rec: Dict[str, Any]) -> Utterance:
    return Utterance(
        uid=str(rec["uid"]),
        dataset=str(rec.get("dataset") or "farm"),
        scan_id=str(rec["scan_id"]),
        target_id=int(rec["target_id"]),
        distractor_ids=list(rec.get("distractor_ids") or []),
        instance_type=str(rec.get("instance_type") or ""),
        utterance=str(rec.get("utterance") or ""),
        mentions_target_class=bool(rec.get("mentions_target_class") or False),
        reference_type=rec.get("reference_type"),
        coarse_reference_type=rec.get("coarse_reference_type"),
        anchor_ids=rec.get("anchor_ids"),
        anchors_types=rec.get("anchors_types"),
        uses_object_lang=rec.get("uses_object_lang"),
        uses_spatial_lang=rec.get("uses_spatial_lang"),
        uses_color_lang=rec.get("uses_color_lang"),
        uses_shape_lang=rec.get("uses_shape_lang"),
    )


def _load_utterances(path: Path, scenes: Optional[List[str]], max_utterances: int) -> List[Utterance]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"unexpected utterances shape in {path}")
    utts = [_to_utterance(r) for r in payload]
    if scenes:
        allow = set(scenes)
        utts = [u for u in utts if u.scan_id in allow]
    if max_utterances and max_utterances > 0:
        utts = utts[: int(max_utterances)]
    return utts


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("predict", "score", "both"), default="both")
    p.add_argument("--eval-root", type=Path, required=True,
                   help="FARM-Scenes gt/ directory (contains <dataset>/utterances.json + _gt_instances/).")
    p.add_argument("--dataset", required=True, help="grandtour | spot | odin1")
    p.add_argument("--utterances", type=Path, default=None)
    p.add_argument("--scenes-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--scene", action="append", default=None)
    p.add_argument("--max-utterances", type=int, default=0)
    p.add_argument("--parse-cache-path", type=Path, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--use-vlm", action="store_true",
                   help="Opt into spatial VLM checks. Default is the locked no-VLM setting.")
    p.add_argument("--spatial-method", default="unified_soft_w50",
                   help="Spatial method passed through to eval_referit3d_spatial.")
    p.add_argument("--retrieval-mode", choices=("single", "multi"), default="multi",
                   help="Semantic retrieval mode. Locked FINAL setting is multi.")
    p.add_argument("--candidate-pool-mode", choices=("active", "all", "active_plus_redirect"),
                   default="active",
                   help="Candidate pool for multi retrieval. Locked FINAL setting is active.")
    p.add_argument("--max-predictions", type=int, default=100)
    p.add_argument("--pre-filter-k", type=int, default=-1)
    p.add_argument("--geometry-mode", default="alias_expand",
                   choices=("default", "canonical_source", "alias_expand", "alias_text", "alias_text_expand"))
    p.add_argument("--max-aliases-per-candidate", type=int, default=2)
    p.add_argument("--alias-order", default="source_first", choices=("source_first", "text_first"))
    p.add_argument("--logger-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.logger_level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    utterances_path = args.utterances or (args.eval_root / args.dataset / "utterances.json")
    metrics_out = args.metrics_out or args.predictions.with_name(args.predictions.stem + "-metrics.json")

    if args.phase in {"predict", "both"}:
        utts = _load_utterances(utterances_path, args.scene, int(args.max_utterances))
        LOGGER.info("queued %d utterances from %s", len(utts), utterances_path)
        if not utts:
            LOGGER.error("no utterances matched scene filter %s", args.scene)
            return 2
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        run_spatial_predictions(
            scenes_dir=args.scenes_dir,
            output_path=args.predictions,
            utterances=utts,
            use_vlm=bool(args.use_vlm),
            k_sigma=2.5,
            max_predictions=int(args.max_predictions),
            pre_filter_k=int(args.pre_filter_k),
            max_candidates_for_vlm=10,
            vlm_rerank_candidates=False,
            vlm_rerank_top_k=20,
            vlm_rerank_blend=0.65,
            retrieval_mode=str(args.retrieval_mode),
            candidate_pool_mode=str(args.candidate_pool_mode),
            spatial_method=str(args.spatial_method),
            predicate_calibration_path=None,
            hard_threshold=0.5,
            parse_cache_path=args.parse_cache_path,
            resume=not args.no_resume,
            prefer_voxel_aabb=True,
            geometry_mode=str(args.geometry_mode),
            max_aliases_per_candidate=int(args.max_aliases_per_candidate),
            alias_order=str(args.alias_order),
        )
        LOGGER.info("predictions written to %s", args.predictions)

    if args.phase in {"score", "both"}:
        score_args = [
            "--predictions", str(args.predictions),
            "--eval-root", str(args.eval_root),
            "--dataset", str(args.dataset),
            "--metrics-out", str(metrics_out),
            "--iou-thresholds", "0.1", "0.25", "0.5",
            "--recall-ks", "1", "3", "5", "10",
        ]
        LOGGER.info("scoring %s", args.predictions)
        return int(score_largescale_main(score_args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
