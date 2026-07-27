"""Iterate IRef-VLA statements and write a predictions JSON.

Mirrors :mod:`scene_graph.eval.referit3d.runner`. For each scene we:

1. Load scene_state.pt once → build :class:`SceneGraphRetriever`.
2. Iterate the scene's statements; per statement, run ``retrieve(query)``
   and flatten clusters into a flat ranked list of :class:`PredictedObject`.
3. Persist incrementally for resume-on-rerun.

Output schema (``predictions.json``):

    [
      {
        "uid": "<scene>/<region>/<text-hash>/<variant>",
        "scene_id": "00238-j6fHrce9pHR",
        "region_id": 0,
        "statement": "the picture that is above the coffee table",
        "target_id": 26,
        "target_class": "picture",
        "distractor_ids": [27, 28, 29, 30, 31, 53],
        "anchor_ids": [25],
        "anchor_classes": ["coffee table"],
        "relation": "above",
        "relation_type": "binary",
        "ranked": [
          {"rank": 1, "object_id": ..., "score": ..., "bbox_min": [...], "bbox_max": [...], "label": ..., "caption": ...},
          ...
        ],
        "elapsed_s": 1.12
      },
      ...
    ]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# We deliberately re-use ReferIt3D's ScenePredictor + ranked_to_dicts —
# the bridge from SceneGraphRetriever clusters to a flat ranked list of
# PredictedObject is dataset-agnostic.
from scene_graph.eval.referit3d.retrieval_adapter import ScenePredictor, ranked_to_dicts

from .dataset import (
    Statement,
    list_local_scenes,
    load_all_statements,
    statements_by_scene,
)

LOGGER = logging.getLogger("scene_graph.eval.iref_vla.runner")


# ---------------------------------------------------------------------
# Scene-state discovery
# ---------------------------------------------------------------------


def discover_scene_states(scenes_dir: Path) -> Dict[str, Path]:
    """Return ``{scene_id: scene_state_path}`` for every .pt under ``scenes_dir``.

    Two layouts are supported:
    * flat   — ``<scene_id>.pt`` directly under ``scenes_dir``
    * nested — ``<scene_id>/scene_state.pt``
    """
    out: Dict[str, Path] = {}
    if not scenes_dir.exists():
        return out
    for entry in sorted(scenes_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".pt":
            out[entry.stem] = entry
        elif entry.is_dir():
            cand = entry / "scene_state.pt"
            if cand.exists():
                out[entry.name] = cand
    return out


# ---------------------------------------------------------------------
# Predictions helpers
# ---------------------------------------------------------------------


def statement_to_record(statement: Statement) -> Dict[str, Any]:
    """Stable JSON shape for one IRef-VLA statement (without predictions)."""
    return {
        "uid": statement.uid,
        "scene_id": statement.scene_id,
        "region_id": int(statement.region_id),
        "statement": statement.statement,
        "target_id": int(statement.target_id),
        "target_class": statement.target_class,
        "distractor_ids": list(statement.distractor_ids),
        "anchor_ids": list(statement.anchor_ids),
        "anchor_classes": list(statement.anchor_classes),
        "relation": statement.relation,
        "relation_type": statement.relation_type,
        "variant_idx": int(statement.variant_idx),
        "is_false_statement": bool(statement.is_false_statement),
    }


def _load_existing(output_path: Path) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not parse existing predictions at %s: %s", output_path, exc)
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in payload or []:
        uid = str(entry.get("uid") or "")
        if uid:
            out[uid] = entry
    return out


def _persist(
    output_path: Path,
    ordered_uids: Sequence[str],
    predictions: Dict[str, Dict[str, Any]],
) -> None:
    out: List[Dict[str, Any]] = [predictions[u] for u in ordered_uids if u in predictions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Knobs that affect how each statement is scored."""

    k_sigma: float = 2.5
    max_predictions: int = 20
    retriever_verbose: bool = False
    embedder_verbose: bool = False
    log_every: int = 50
    prefer_voxel_aabb: bool = True
    voxel_sor_k: int = 8
    voxel_sor_alpha: float = 2.0


def run_predictions(
    *,
    scenes_dir: Path,
    output_path: Path,
    cfg: Optional[RunnerConfig] = None,
    statements: Optional[Iterable[Statement]] = None,
    scene_filter: Optional[Sequence[str]] = None,
    max_statements: Optional[int] = None,
    resume: bool = True,
    dataset_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Iterate statements grouped by scene_id, write predictions JSON.

    Args:
        scenes_dir: Directory containing ``<scene_id>.pt`` files (or
            ``<scene_id>/scene_state.pt`` sub-dirs).
        output_path: Predictions JSON path; written incrementally.
        cfg: Runner knobs (k_sigma, max_predictions, …).
        statements: Iterable of :class:`Statement` to score. Defaults to
            every statement under ``IREF_VLA_ROOT`` (filtered to the scenes
            for which ``scenes_dir`` actually has a saved scene_state.pt).
        scene_filter: Optional whitelist of scene ids.
        max_statements: Cap total work for smoke tests.
        resume: When True, skip statements already present in the JSON.
        dataset_root: IRef-VLA root if defaulting; passed to ``load_all_statements``.

    Returns:
        The persisted prediction list (also on disk at ``output_path``).
    """
    cfg = cfg or RunnerConfig()
    from scene_graph.llm_utils import EmbedInterface
    from scene_graph.retrieval.scene_graph_retriever import SceneGraphRetriever

    scene_states = discover_scene_states(scenes_dir)
    if statements is None:
        # Default: load statements only for scenes we actually have scene_state for.
        scenes_with_states = set(scene_states.keys())
        statements = load_all_statements(
            dataset_root=dataset_root,
            scene_filter=sorted(scenes_with_states),
        )
    statements = list(statements)
    if scene_filter is not None:
        allow = set(scene_filter)
        statements = [s for s in statements if s.scene_id in allow]
    if max_statements is not None:
        statements = statements[: int(max_statements)]

    LOGGER.info(
        "found %d scene_state.pt under %s; %d statements queued",
        len(scene_states), scenes_dir, len(statements),
    )

    grouped: Dict[str, List[Statement]] = statements_by_scene(statements)

    existing: Dict[str, Dict[str, Any]] = _load_existing(output_path) if resume else {}
    if existing:
        LOGGER.info("resuming from %d existing predictions", len(existing))

    predictions: Dict[str, Dict[str, Any]] = dict(existing)
    ordered_uids: List[str] = [s.uid for s in statements]

    embedder = EmbedInterface(verbose=cfg.embedder_verbose)

    n_scenes_done = 0
    n_stmts_done = 0
    n_stmts_ok = 0
    n_stmts_fail = 0
    t_start = time.time()

    for scan_id in sorted(grouped.keys()):
        stmts_for_scene = [s for s in grouped[scan_id] if s.uid not in predictions]
        if not stmts_for_scene:
            continue
        pt = scene_states.get(scan_id)
        if pt is None:
            LOGGER.warning("skipping %s — no scene_state.pt found", scan_id)
            continue
        LOGGER.info(
            "loading scene %s (%d new statements) from %s",
            scan_id, len(stmts_for_scene), pt.name,
        )
        try:
            retriever = SceneGraphRetriever.from_scene_state(
                pt, embedder=embedder, verbose=cfg.retriever_verbose,
            )
            predictor = ScenePredictor(
                retriever,
                k_sigma=cfg.k_sigma,
                max_predictions=cfg.max_predictions,
                prefer_voxel_aabb=cfg.prefer_voxel_aabb,
                voxel_sor_k=cfg.voxel_sor_k,
                voxel_sor_alpha=cfg.voxel_sor_alpha,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("failed to build retriever for %s: %s", scan_id, exc)
            continue

        n_scenes_done += 1
        scene_started = time.time()

        for i, stmt in enumerate(stmts_for_scene, start=1):
            t0 = time.time()
            record = statement_to_record(stmt)
            try:
                ranked = predictor.predict(stmt.statement)
                record["ranked"] = ranked_to_dicts(ranked)
                record["error"] = None
                n_stmts_ok += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("retrieve failed on %s: %s", stmt.uid, exc)
                record["ranked"] = []
                record["error"] = str(exc)
                n_stmts_fail += 1
            record["elapsed_s"] = round(time.time() - t0, 4)
            predictions[stmt.uid] = record
            n_stmts_done += 1

            if i % cfg.log_every == 0 or i == len(stmts_for_scene):
                _persist(output_path, ordered_uids, predictions)
                LOGGER.info(
                    "  %s %d/%d statements (%.1f stmt/s)",
                    scan_id, i, len(stmts_for_scene),
                    i / max(1e-3, time.time() - scene_started),
                )
        _persist(output_path, ordered_uids, predictions)

    _persist(output_path, ordered_uids, predictions)
    LOGGER.info(
        "done. %d scenes, %d statements (%d ok, %d failed) in %.1fs (avg %.2f stmt/s)",
        n_scenes_done, n_stmts_done, n_stmts_ok, n_stmts_fail,
        time.time() - t_start,
        n_stmts_done / max(1e-3, time.time() - t_start),
    )
    return [predictions[u] for u in ordered_uids if u in predictions]
