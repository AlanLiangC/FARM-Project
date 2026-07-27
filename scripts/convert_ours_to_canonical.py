"""Convert ours retrieval-predictions JSON → canonical predictions schema.

The ours predict-phase writes JSON records with shape
``{uid, scan_id|scene_id, target_id, target_description, ranked:[…]}``
already very close to the canonical schema consumed by
``scripts/eval_predictions.py``. This script normalises it:

  * Sets ``method = "ours"``
  * Sets ``dataset`` (from the ``--bench`` flag)
  * Coerces ``scene_id`` → ``scan_id`` so the canonical schema has one
    scene-id field.
  * Per candidate: sets ``pred_mask_source = "ours_state"``, copies the
    label-derived ``evidence_object_id`` (alias-resolved when the
    candidate label begins with ``alias:<id>``), leaves
    ``chosen_view_image_id = null`` so the unified scorer runs the
    locked V1 picker against the candidate's evidence object.

Output is a list-of-records JSON ready for ``eval_predictions.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

LOGGER = logging.getLogger("convert_ours_to_canonical")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", type=Path, required=True,
                   help="Input ours predictions JSON (from eval_referit3d_spatial.py / eval_iref_vla.py).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output canonical predictions JSON.")
    p.add_argument("--bench", choices=("scannet", "hm3d"), required=True)
    p.add_argument("--method-tag", default="ours")
    return p.parse_args()


def _resolve_evidence_object_id(candidate: Mapping[str, Any]) -> int:
    """Extract the alias-resolved evidence object id.

    Per ``scene_graph.eval.visible_mask.SceneStateMaskIndex.candidate_evidence_object_id``:
    a candidate whose ``label`` starts with ``alias:<id>`` reports ``<id>`` as
    the evidence object; otherwise ``object_id`` itself.
    """
    label = str(candidate.get("label") or "")
    if label.startswith("alias:"):
        parts = label.split(":", 2)
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except (TypeError, ValueError):
                pass
    try:
        return int(candidate.get("object_id", -1))
    except (TypeError, ValueError):
        return -1


def convert_one(record: Mapping[str, Any], bench: str, method_tag: str) -> Dict[str, Any]:
    scan_id = str(record.get("scan_id") or record.get("scene_id") or "")
    ranked_in = list(record.get("ranked") or [])
    ranked_out: List[Dict[str, Any]] = []
    for cand in ranked_in:
        ranked_out.append({
            "rank": int(cand.get("rank", len(ranked_out))),
            "object_id": int(cand.get("object_id", -1)),
            "score": float(cand.get("score", 0.0)),
            "bbox_min": list(cand.get("bbox_min")) if cand.get("bbox_min") is not None else None,
            "bbox_max": list(cand.get("bbox_max")) if cand.get("bbox_max") is not None else None,
            # Leave None to let the scorer's V1 picker resolve the chosen view
            # against the candidate's evidence object's mask observations.
            "chosen_view_image_id": cand.get("chosen_view_image_id"),
            "pred_mask_source": "ours_state",
            "pred_mask_path": None,
            "evidence_object_id": _resolve_evidence_object_id(cand),
            "label": cand.get("label"),
            "caption": cand.get("caption"),
            "region_label": cand.get("region_label"),
        })
    return {
        "uid": record.get("uid"),
        "dataset": bench,
        "scan_id": scan_id,
        "target_id": int(record.get("target_id", -1)),
        "target_description": record.get("target_description"),
        "utterance": record.get("utterance") or record.get("statement"),
        "method": method_tag,
        "ranked": ranked_out,
        "method_metadata": {
            "spatial_method": record.get("spatial_method"),
            "predicates": record.get("predicates"),
            "geometry_mode": record.get("geometry_mode"),
            "method": record.get("method"),  # spatial / fallback / error
            "error": record.get("error"),
        },
        "error": record.get("error"),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    args = _parse_args()

    src = json.loads(args.src.read_text(encoding="utf-8"))
    if not isinstance(src, list):
        LOGGER.error("expected list of predictions in %s; got %s", args.src, type(src).__name__)
        return 2

    out = [convert_one(r, args.bench, args.method_tag) for r in src]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    LOGGER.info("converted %d records: %s → %s", len(out), args.src, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
