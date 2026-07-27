#!/usr/bin/env python3
"""Score a largescale predictions JSON against the largescale GT cache.

Loads ``predictions.json`` (referit3d-shape, with ``ranked`` lists) and the
per-scene GT npz files shipped in the FARM-Scenes release
(``<eval-root>/<dataset>/_gt_instances/<scene>.npz``), then runs the
referit3d metric stack to produce a sibling ``-metrics.json`` with the
overall headline + per-relation / per-difficulty / per-instance-type
breakdowns. Normally invoked via ``scripts/eval_farm_scenes.py``.

Usage::

    python scripts/score_largescale_predictions.py \\
        --predictions /data/out/farm_odin1_preds.json \\
        --eval-root /data/gt \\
        --dataset odin1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


LOGGER = logging.getLogger("score_largescale_predictions")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True,
                   help="Predictions JSON (referit3d shape).")
    p.add_argument("--eval-root", type=Path, required=True,
                   help="Largescale eval root (parent of <dataset>/_gt_instances/).")
    p.add_argument("--dataset", type=str, required=True,
                   help="One of grandtour | tartanground | odin1.")
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Where to write metrics JSON. Default: alongside predictions.")
    p.add_argument("--iou-thresholds", type=float, nargs="+", default=(0.25, 0.5))
    p.add_argument("--recall-ks", type=int, nargs="+", default=(1, 3, 5, 10))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(name)s: %(message)s")

    # Lazy imports — these depend on the in-tree code being importable.
    from scene_graph.eval.referit3d.metrics import (
        UtteranceScore,
        aggregate,
        score_utterance,
    )
    from scene_graph.eval.largescale.largescale_gt import load_scene_gt

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"unexpected predictions shape in {args.predictions}")

    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in payload:
        if isinstance(rec, dict):
            by_scene[str(rec.get("scan_id") or "")].append(rec)

    scores: List[UtteranceScore] = []
    iou_thresholds = tuple(float(t) for t in args.iou_thresholds)
    recall_ks = tuple(int(k) for k in args.recall_ks)
    for sid, recs in sorted(by_scene.items()):
        try:
            gt = load_scene_gt(args.eval_root, args.dataset, sid)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("could not load GT for %s: %s", sid, exc)
            for r in recs:
                scores.append(score_utterance(
                    {**r, "error": f"GT load failed: {exc}"},
                    gt={},
                    iou_thresholds=iou_thresholds,
                ))
            continue
        for r in recs:
            scores.append(score_utterance(r, gt=gt, iou_thresholds=iou_thresholds))

    aggr = aggregate(scores, iou_thresholds=iou_thresholds, recall_ks=recall_ks)

    out = args.metrics_out or args.predictions.with_name(args.predictions.stem + "-metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggr, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("wrote metrics → %s", out)

    # Pretty-print headline
    overall = aggr.get("overall") or {}
    LOGGER.info("# Overall n=%s/%s", overall.get("n"), overall.get("n_total"))
    for k in (
        "mean_top1_iou",
        "acc@1@iou=0.25",
        "acc@1@iou=0.5",
        "recall@1@iou=0.25",
        "recall@3@iou=0.25",
        "recall@5@iou=0.25",
        "recall@10@iou=0.25",
        "mrr@iou=0.25",
        "median_rank@iou=0.25",
        "hit_rate@any_rank@iou=0.25",
    ):
        if k in overall:
            v = overall[k]
            LOGGER.info("  %-32s %.4f" if isinstance(v, float) else "  %-32s %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
